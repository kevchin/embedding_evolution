#!/usr/bin/env python3
"""
Photo ingestion pipeline — organize + caption + embed.

┌─────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│  INPUT_DIR  │───►│  Phase 0: ORGANIZE       │───►│  ARCHIVE_ROOT/           │
│  (raw dump) │    │  • EXIF / ctime date     │    │  YYYY/MM/DD/photo.jpg    │
└─────────────┘    │  • SHA-256 dedup vs DB   │    └────────────┬─────────────┘
                   │  • copy2 + verify        │                 │
                   └──────────────────────────┘                 ▼
                                                  ┌─────────────────────────────┐
                   GPU 0 ─► Qwen2-VL-7B (4-bit)  │  Phase 1: CAPTION           │
                   GPU 1 ─► CLIP ViT-L/14         │  Phase 2: CLIP image embed  │
                            BGE-large-en-v1.5      │  Phase 3: BGE text embed    │
                                                  └────────────┬────────────────┘
                                                               ▼
                                                  ┌─────────────────────────────┐
                                                  │  Phase 4: DB UPSERT         │
                                                  │  Postgres + pgvector        │
                                                  └─────────────────────────────┘

Date extraction priority (works for HEIC, JPG, PNG):
  1. PIL EXIF  → DateTimeOriginal (36867) / DateTimeDigitized (36868) / DateTime (306)
  2. exifread  → EXIF DateTimeOriginal  (fallback for some HEIC variants)
  3. File      → st_birthtime (macOS/Windows) or st_mtime (Linux)

Deduplication:
  • SHA-256 of raw bytes checked against DB hashes before any copy is made.
  • A filename-collision check inside each YYYY/MM/DD dir appends _001, _002 …
  • Every copy is verified by re-hashing the destination; bad copies are removed.

GPU assignment (--gpu-mode):
  auto     captions → cuda:0  |  CLIP + BGE → cuda:1   (default)
  single   everything on cuda:0
  reverse  captions → cuda:1  |  CLIP + BGE → cuda:0

Usage:
    # Full pipeline — organize then caption/embed/store:
    python ingest_auto.py \\
        --input-dir  /path/to/new/photos \\
        --archive-root /path/to/archive \\
        --dsn "postgresql://claude1:claude1@localhost/photo_archive"

    # Dry-run: preview copies, no writes:
    python ingest_auto.py --input-dir /new --archive-root /archive --dry-run

    # Organize only (copy + dedup, skip ML/DB):
    python ingest_auto.py --input-dir /new --archive-root /archive --organize-only

    # Ingest already-organized archive (skip copy phase):
    python ingest_auto.py --archive-root /archive --dsn "..." --ingest-only

Environment variables (overridden by CLI flags):
    PHOTO_DB_DSN   postgresql://claude1:claude1@localhost/photo_archive
    PHOTO_ROOT     /path/to/archive
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import shutil
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from pillow_heif import register_heif_opener
register_heif_opener()          # enables HEIC/HEIF support throughout PIL
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

CLIP_MODEL_ID    = "openai/clip-vit-large-patch14"   # 768-dim image embedding
CAPTION_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"       # multimodal caption model
BGE_MODEL_ID     = "BAAI/bge-large-en-v1.5"          # 1024-dim text embedding

CAPTION_BATCH_SIZE = 25     # safe at 512×512 on 16 GB; lower to 1-2 if OOM
CHECKPOINT_SIZE    = 200   # commit every N photos
EMBED_BATCH_SIZE   = 64
CLIP_BATCH_SIZE    = 32

MAX_PIXELS = 512 * 512      # Qwen2-VL vision token budget (512×512)
MIN_PIXELS = 4 * 28 * 28   # Qwen2-VL hard minimum

# ── GPU device assignment ─────────────────────────────────────────────────────
def _pick_devices(mode: str = "auto") -> tuple[str, str]:
    """Return (caption_device, embed_device)."""
    n = torch.cuda.device_count()
    if n == 0:
        log.warning("No CUDA GPUs found — running on CPU (slow).")
        return "cpu", "cpu"
    if mode == "single" or n == 1:
        dev = "cuda:0"
        log.info("Single-GPU mode — all models on %s (%s).", dev, torch.cuda.get_device_name(0))
        return dev, dev
    if mode == "reverse":
        log.info("Reverse mode: captions→cuda:1 (%s), embeddings→cuda:0 (%s).",
                 torch.cuda.get_device_name(1), torch.cuda.get_device_name(0))
        return "cuda:1", "cuda:0"
    log.info("2-GPU mode: captions→cuda:0 (%s), embeddings→cuda:1 (%s).",
             torch.cuda.get_device_name(0), torch.cuda.get_device_name(1))
    return "cuda:0", "cuda:1"

CAPTION_DEVICE, EMBED_DEVICE = _pick_devices()

# ── Model singletons ──────────────────────────────────────────────────────────
_clip_model      = None
_clip_processor  = None
_caption_model   = None
_caption_proc    = None
_bge_model       = None
_bge_tokenizer   = None


def _unload(model):
    if model is None:
        return
    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_caption_model():
    global _caption_model, _caption_proc
    if _caption_model is None:
        from transformers import (
            Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
        )
        log.info("Loading Qwen2-VL (4-bit NF4) on %s …", CAPTION_DEVICE)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto" if CAPTION_DEVICE == EMBED_DEVICE else {"": CAPTION_DEVICE}
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            log.info("flash-attn detected — using flash_attention_2.")
        except ImportError:
            attn_impl = "sdpa"
            log.info("flash-attn not installed — using sdpa (pip install flash-attn for ~20%% speedup).")

        _caption_model = Qwen2VLForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_ID,
            quantization_config=bnb_cfg,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            local_files_only=True,
        ).eval()

        try:
            _caption_model = torch.compile(_caption_model, mode="reduce-overhead")
            log.info("Caption model compiled with torch.compile.")
        except Exception as e:
            log.warning("torch.compile skipped: %s", e)

        _caption_proc = AutoProcessor.from_pretrained(
            CAPTION_MODEL_ID,
            trust_remote_code=True,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            local_files_only=True,
        )
    return _caption_model, _caption_proc


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        log.info("Loading CLIP ViT-L/14 on %s …", EMBED_DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
        _clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_ID, torch_dtype=torch.float16, local_files_only=True
        ).eval().to(EMBED_DEVICE)
    return _clip_model, _clip_processor


def get_bge():
    global _bge_model, _bge_tokenizer
    if _bge_model is None:
        from transformers import AutoTokenizer, AutoModel
        log.info("Loading BGE-large-en on %s …", EMBED_DEVICE)
        _bge_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_ID, local_files_only=True)
        _bge_model = AutoModel.from_pretrained(
            BGE_MODEL_ID, torch_dtype=torch.float16, local_files_only=True
        ).eval().to(EMBED_DEVICE)
    return _bge_model, _bge_tokenizer


# ── Embedding helpers ─────────────────────────────────────────────────────────
def embed_images_clip(image_paths: list[str]) -> np.ndarray:
    """Return (N, 768) float32 L2-normalised CLIP image embeddings."""
    model, processor = get_clip()
    all_embs: list[np.ndarray] = []
    t0 = time.time()
    log.info("CLIP embeddings for %d image(s) …", len(image_paths))
    for i in range(0, len(image_paths), CLIP_BATCH_SIZE):
        batch  = image_paths[i : i + CLIP_BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(EMBED_DEVICE)
        with torch.no_grad(), torch.autocast(EMBED_DEVICE):
            out = model.get_image_features(**inputs)
        # transformers may return BaseModelOutputWithPooling instead of a Tensor
        embs = out if isinstance(out, torch.Tensor) else out.pooler_output
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().float().numpy())
    log.info("  CLIP done in %.1fs.", time.time() - t0)
    return np.concatenate(all_embs, axis=0)


def embed_text_bge(texts: list[str]) -> np.ndarray:
    """Return (N, 1024) float32 L2-normalised BGE text embeddings."""
    model, tokenizer = get_bge()
    all_embs: list[np.ndarray] = []
    t0 = time.time()
    log.info("BGE embeddings for %d caption(s) …", len(texts))
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [
            "Represent this sentence for searching relevant passages: " + t
            for t in texts[i : i + EMBED_BATCH_SIZE]
        ]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(EMBED_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        embs = out.last_hidden_state[:, 0, :]   # CLS token
        embs = embs / embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.cpu().float().numpy())
    log.info("  BGE done in %.1fs.", time.time() - t0)
    return np.concatenate(all_embs, axis=0)


# ── Captioning ────────────────────────────────────────────────────────────────
CAPTION_PROMPT = (
    "Describe this photo in 2-3 detailed sentences. "
    "Include: people (count, age, activity), objects, setting, colors, mood. "
    "Be specific and factual. No speculation."
)


def generate_captions_batch(image_paths: list[str]) -> list[str]:
    """Generate Qwen2-VL captions for a list of absolute image paths."""
    from qwen_vl_utils import process_vision_info

    model, processor = get_caption_model()

    # Pre-filter unreadable images
    valid: list[str] = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(p)
        except Exception as exc:
            log.warning("Skipping unreadable image %s: %s", p, exc)
    if len(valid) < len(image_paths):
        log.warning("Skipped %d unreadable file(s).", len(image_paths) - len(valid))
    image_paths = valid

    captions: list[str] = []
    t0 = time.time()
    log.info("Captioning %d image(s) (batch=%d, device=%s) …",
             len(image_paths), CAPTION_BATCH_SIZE, CAPTION_DEVICE)

    for i in range(0, len(image_paths), CAPTION_BATCH_SIZE):
        batch_paths = image_paths[i : i + CAPTION_BATCH_SIZE]
        messages_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{p}"},
                    {"type": "text",  "text":  CAPTION_PROMPT},
                ],
            }]
            for p in batch_paths
        ]
        texts = [
            processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(
            [msg for msgs in messages_batch for msg in msgs]
        )
        inputs = processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(CAPTION_DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                temperature=None, top_p=None,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        captions.extend(processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))

        done    = i + len(batch_paths)
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed else 0
        eta     = (len(image_paths) - done) / rate if rate else 0
        log.info("  Caption %d/%d — %.1fs elapsed — ETA %dm%ds",
                 done, len(image_paths), elapsed, int(eta // 60), int(eta % 60))

    return captions


# ── Database ──────────────────────────────────────────────────────────────────
DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photos (
    id              SERIAL PRIMARY KEY,
    filepath        TEXT        NOT NULL UNIQUE,
    content_hash    TEXT        NOT NULL,
    filename        TEXT        NOT NULL,
    photo_date      DATE,
    caption         TEXT,
    clip_embedding  VECTOR(768),
    bge_embedding   VECTOR(1024),
    file_size_bytes BIGINT,
    width           INT,
    height          INT,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

-- Safe migrations for databases created by earlier versions of this script.
ALTER TABLE photos ADD COLUMN IF NOT EXISTS photo_date DATE;
-- Drop the UNIQUE constraint on content_hash if it exists (it was wrong: the
-- same bytes can live at multiple paths, e.g. archive copy + input leftover).
DO $$ BEGIN
    ALTER TABLE photos DROP CONSTRAINT IF EXISTS photos_content_hash_key;
EXCEPTION WHEN others THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_content_hash ON photos (content_hash);

CREATE INDEX IF NOT EXISTS idx_clip_hnsw
    ON photos USING hnsw (clip_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_bge_hnsw
    ON photos USING hnsw (bge_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_caption_fts
    ON photos USING gin(to_tsvector('english', coalesce(caption, '')));

CREATE INDEX IF NOT EXISTS idx_photo_date ON photos (photo_date);
CREATE INDEX IF NOT EXISTS idx_filepath    ON photos (filepath);
"""


def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DB_SCHEMA)
    conn.commit()
    log.info("Database schema ready.")


def get_existing_hashes(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM photos")
        return {r[0] for r in cur.fetchall()}


def get_existing_paths(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT filepath FROM photos")
        return {r[0] for r in cur.fetchall()}


def upsert_photos(conn, rows: list[dict]) -> None:
    # Pre-filter: fetch hashes already in DB so we never attempt to INSERT a
    # duplicate hash in the same batch (the in-memory set in ingest() is built
    # at startup; rows added during earlier chunks of this run aren't in it yet).
    # This is a cheap round-trip compared to the cost of the ML work already done.
    batch_hashes = tuple(r["content_hash"] for r in rows)
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM photos WHERE content_hash = ANY(%s)", (list(batch_hashes),))
        already_stored = {row[0] for row in cur.fetchall()}

    if already_stored:
        skipped = [r["filepath"] for r in rows if r["content_hash"] in already_stored]
        log.info("  Skipping %d row(s) whose hash is already in DB: %s …",
                 len(skipped), skipped[0] if skipped else "")
        rows = [r for r in rows if r["content_hash"] not in already_stored]

    if not rows:
        conn.commit()
        return

    sql = """
        INSERT INTO photos
            (filepath, content_hash, filename, photo_date, caption,
             clip_embedding, bge_embedding, file_size_bytes, width, height)
        VALUES %s
        ON CONFLICT (filepath) DO UPDATE SET
            content_hash   = EXCLUDED.content_hash,
            photo_date     = EXCLUDED.photo_date,
            caption        = EXCLUDED.caption,
            clip_embedding = EXCLUDED.clip_embedding,
            bge_embedding  = EXCLUDED.bge_embedding,
            ingested_at    = now()
    """
    values = [
        (
            r["filepath"], r["content_hash"], r["filename"], r["photo_date"],
            r["caption"], r["clip_embedding"].tolist(), r["bge_embedding"].tolist(),
            r["file_size_bytes"], r["width"], r["height"],
        )
        for r in rows
    ]
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
    conn.commit()


# ── Utilities ─────────────────────────────────────────────────────────────────
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: str) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


# ── Date extraction ───────────────────────────────────────────────────────────
_EXIF_FMTS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d")


def _parse_exif_str(s: str) -> Optional[datetime]:
    for fmt in _EXIF_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def get_photo_date(path: str) -> datetime:
    """
    Return the best available capture datetime for a photo.

    Priority:
      1. PIL EXIF tags 36867/36868/306 — works for JPG, PNG, most HEIC.
      2. exifread EXIF tags — more robust for certain HEIC variants.
      3. File st_birthtime (macOS/Windows) or st_mtime (Linux).
    """
    # Method 1: PIL
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                for tag_id in (36867, 36868, 306):  # DateTimeOriginal, Digitized, DateTime
                    val = exif.get(tag_id)
                    if val:
                        dt = _parse_exif_str(str(val))
                        if dt:
                            return dt
    except Exception:
        pass

    # Method 2: exifread (robust HEIC fallback)
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
        for key in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
            val = tags.get(key)
            if val:
                dt = _parse_exif_str(str(val))
                if dt:
                    return dt
    except Exception:
        pass

    # Method 3: filesystem timestamps
    try:
        st = os.stat(path)
        ts = getattr(st, "st_birthtime", None)  # macOS / Windows
        if not ts:
            ts = st.st_mtime                     # Linux
        return datetime.fromtimestamp(ts)
    except Exception:
        return datetime.now()


# ── Phase 0: Organize ─────────────────────────────────────────────────────────
def find_input_images(directory: str) -> list[str]:
    """Recursively find all supported images, skipping macOS sidecar files."""
    return sorted(
        str(p)
        for p in Path(directory).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not p.name.startswith("._")
        and not p.name.startswith(".")
    )


def _unique_dest(dest_dir: Path, filename: str, known_hash: str) -> Optional[Path]:
    """
    Return an available destination Path for *filename* inside *dest_dir*.

    Returns None if the same bytes are already there (already organized).
    Appends _001, _002 … on name collisions with *different* content.
    """
    stem      = Path(filename).stem
    suffix    = Path(filename).suffix.lower()
    candidate = dest_dir / f"{stem}{suffix}"

    if not candidate.exists():
        return candidate

    # File exists — is it the same bytes?
    if sha256_file(str(candidate)) == known_hash:
        return None  # already done

    # Different content — find a free slot
    for counter in range(1, 10_000):
        candidate = dest_dir / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        if sha256_file(str(candidate)) == known_hash:
            return None  # found our bytes at a renamed slot

    raise RuntimeError(f"Could not find a free filename for {filename} in {dest_dir}")


def organize_photos(
    input_dir: str,
    archive_root: str,
    existing_hashes: set[str],
    dry_run: bool = False,
) -> list[tuple[str, datetime, str]]:
    """
    Scan *input_dir*, skip content already in *existing_hashes*, and copy unique
    photos into *archive_root*/YYYY/MM/DD/ using the photo's capture date.

    Returns:
        List of (absolute_archive_path, photo_datetime, sha256) for newly
        organized photos — ready to pass directly into ingest().

    *existing_hashes* is mutated in place so that files processed earlier in
    the same run are not re-copied.
    """
    archive_path = Path(archive_root)
    candidates   = find_input_images(input_dir)
    log.info("Organize: %d candidate image(s) in %s", len(candidates), input_dir)

    new_files:      list[tuple[str, datetime, str]] = []
    skipped_dupes:  int = 0
    copy_errors:    int = 0

    for src in tqdm(candidates, desc="Organizing", unit="file"):
        try:
            file_hash = sha256_file(src)

            # ── Dedup: skip if already in DB or seen this run ──────────────────
            if file_hash in existing_hashes:
                log.debug("Duplicate skipped (hash known): %s", src)
                skipped_dupes += 1
                continue

            # ── Date → destination directory ───────────────────────────────────
            photo_dt = get_photo_date(src)
            dest_dir = (
                archive_path
                / photo_dt.strftime("%Y")
                / photo_dt.strftime("%m")
                / photo_dt.strftime("%d")
            )
            dest_path = _unique_dest(dest_dir, Path(src).name, file_hash)

            if dest_path is None:
                # Same bytes already exist in the archive tree (not yet in DB)
                log.debug("Already in archive (not yet in DB): %s", src)
                skipped_dupes += 1
                existing_hashes.add(file_hash)
                continue

            # ── Copy or preview ────────────────────────────────────────────────
            if dry_run:
                log.info("[DRY RUN] %s → %s  [%s]",
                         src, dest_path, photo_dt.strftime("%Y/%m/%d"))
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest_path)   # preserves timestamps and metadata

                # Verify integrity
                if sha256_file(str(dest_path)) != file_hash:
                    log.error("Copy verification FAILED (hash mismatch): %s → %s", src, dest_path)
                    dest_path.unlink(missing_ok=True)
                    copy_errors += 1
                    continue

            existing_hashes.add(file_hash)
            new_files.append((str(dest_path), photo_dt, file_hash))
            log.debug("Organized: %s → %s", src, dest_path)

        except Exception as exc:
            log.error("Error organizing %s: %s", src, exc)
            copy_errors += 1

    log.info(
        "Organize complete — copied: %d  |  duplicates skipped: %d  |  errors: %d",
        len(new_files), skipped_dupes, copy_errors,
    )
    return new_files


# ── Phases 1-4: Caption + Embed + DB ─────────────────────────────────────────
def ingest(
    new_files: list[tuple[str, datetime, str]],
    archive_root: str,
    conn,
    existing_paths: set[str],
    existing_hashes: set[str],
    checkpoint_size: int = CHECKPOINT_SIZE,
) -> None:
    """
    Caption, embed, and store *new_files* in Postgres.

    new_files: list of (absolute_path, photo_datetime, sha256)
               produced by organize_photos() or a directory scan.
    archive_root: used to compute the relative path stored in the DB.
    """
    archive_root_path = Path(archive_root).resolve()

    def _rel(abs_path: str) -> str:
        try:
            r = str(Path(abs_path).resolve().relative_to(archive_root_path))
        except ValueError:
            r = Path(abs_path).name
        return r.replace("\\", "/")

    # Second dedup pass: skip anything already in the DB
    to_process: list[tuple[str, str, datetime, str]] = []
    for abs_path, photo_dt, file_hash in new_files:
        r = _rel(abs_path)
        if r in existing_paths:
            log.debug("Path already in DB — skip: %s", r)
            continue
        if file_hash in existing_hashes:
            log.debug("Hash already in DB — skip: %s", abs_path)
            continue
        to_process.append((abs_path, r, photo_dt, file_hash))

    total = len(to_process)
    log.info("%d photo(s) queued for caption/embed/store.", total)
    if not total:
        return

    # Load all models once up front
    log.info("Pre-loading models …")
    get_caption_model()
    get_clip()
    get_bge()

    t_total   = time.time()
    committed = 0

    for chunk_start in range(0, total, checkpoint_size):
        chunk       = to_process[chunk_start : chunk_start + checkpoint_size]
        chunk_paths = [c[0] for c in chunk]
        log.info("── Chunk %d–%d / %d ──", chunk_start + 1, chunk_start + len(chunk), total)

        captions  = generate_captions_batch(chunk_paths)    # GPU 0
        clip_embs = embed_images_clip(chunk_paths)           # GPU 1
        bge_embs  = embed_text_bge(captions)                 # GPU 1

        rows: list[dict] = []
        for j, (abs_path, rel_path, photo_dt, file_hash) in enumerate(chunk):
            w, h = image_dimensions(abs_path)
            rows.append({
                "filepath":        rel_path,
                "content_hash":    file_hash,
                "filename":        Path(abs_path).name,
                "photo_date":      photo_dt.date() if photo_dt else None,
                "caption":         captions[j] if j < len(captions) else "",
                "clip_embedding":  clip_embs[j],
                "bge_embedding":   bge_embs[j],
                "file_size_bytes": os.path.getsize(abs_path),
                "width":           w,
                "height":          h,
            })

        upsert_photos(conn, rows)
        committed += len(rows)

        elapsed = time.time() - t_total
        rate    = committed / elapsed if elapsed else 0
        eta_s   = (total - chunk_start - len(chunk)) / rate if rate else 0
        log.info(
            "✅ Checkpoint %d/%d (%.0f%%)  —  ETA %dm%ds",
            committed, total, 100.0 * committed / total,
            int(eta_s // 60), int(eta_s % 60),
        )

    # Unload caption model — it holds ~14 GB; CLIP + BGE stay for potential re-runs
    global _caption_model, _caption_proc
    _unload(_caption_model)
    _caption_model = _caption_proc = None

    elapsed = time.time() - t_total
    log.info(
        "✅ Ingest done — %d photo(s) in %dm%ds (avg %.1fs/photo).",
        committed, int(elapsed // 60), int(elapsed % 60),
        elapsed / committed if committed else 0,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize photos into YYYY/MM/DD hierarchy, then caption/embed/store them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline — organize new SD card photos then ingest:
  python ingest_auto.py \\
      --input-dir  /Volumes/SD_Card/DCIM \\
      --archive-root /mnt/archive \\
      --dsn "postgresql://claude1:claude1@localhost/photo_archive"

  # Dry-run: preview what would be copied (no writes):
  python ingest_auto.py --input-dir /new --archive-root /archive --dry-run

  # Organize only — copy and deduplicate, skip ML/DB:
  python ingest_auto.py --input-dir /new --archive-root /archive --organize-only

  # Ingest an already-organized archive (skip copy phase):
  python ingest_auto.py --archive-root /archive --dsn "..." --ingest-only
        """,
    )

    parser.add_argument("--input-dir",
                        default=None,
                        help="Raw input directory to scan. Omit with --ingest-only.")
    parser.add_argument("--archive-root",
                        required=True,
                        help="Root of the organized YYYY/MM/DD archive.")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PHOTO_DB_DSN", "postgresql://claude1:claude1@localhost/photo_archive"),
        help="Postgres DSN (or set PHOTO_DB_DSN env var).",
    )
    parser.add_argument(
        "--gpu-mode", default="auto", choices=["auto", "single", "reverse"],
        help=(
            "auto    = captions→GPU0, embeddings→GPU1 (default). "
            "single  = everything on GPU0 (when GPU1 drives a display). "
            "reverse = captions→GPU1, embeddings→GPU0."
        ),
    )
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Preview organize operations; no files copied, no DB writes.")
    parser.add_argument("--organize-only",
                        action="store_true",
                        help="Copy and deduplicate photos into archive; skip caption/embed/DB.")
    parser.add_argument("--ingest-only",
                        action="store_true",
                        help="Skip organize; caption/embed/store the entire archive-root.")
    parser.add_argument("--checkpoint",
                        type=int, default=CHECKPOINT_SIZE,
                        help=f"Commit every N photos (default {CHECKPOINT_SIZE}).")
    parser.add_argument(
        "--max-pixels", type=int, default=MAX_PIXELS,
        help=(
            f"Qwen2-VL vision token budget (default {MAX_PIXELS} = 512×512). "
            "Lower is faster; use 1048576 for text-heavy images."
        ),
    )

    args = parser.parse_args()

    # Validate
    if args.ingest_only and args.input_dir:
        parser.error("--ingest-only and --input-dir are mutually exclusive.")
    if not args.ingest_only and args.input_dir is None:
        parser.error("--input-dir is required unless --ingest-only is set.")

    # Apply GPU mode before any model load
    # globals() targets __main__'s namespace directly — the one all functions here
    # actually read. `import ingest_auto as _self` creates a shadow module and
    # the assignments would silently be ignored (--gpu-mode reverse bug).
    globals()['CAPTION_DEVICE'], globals()['EMBED_DEVICE'] = _pick_devices(args.gpu_mode)
    if args.max_pixels != MAX_PIXELS:
        globals()['MAX_PIXELS'] = args.max_pixels
        log.info("MAX_PIXELS set to %d.", args.max_pixels)

    # ── Connect to DB (unless preview/organize-only) ───────────────────────────
    need_db = not args.organize_only and not args.dry_run
    if need_db:
        conn            = psycopg2.connect(args.dsn)
        init_db(conn)
        existing_hashes = get_existing_hashes(conn)
        existing_paths  = get_existing_paths(conn)
        log.info("DB: %d known hash(es), %d known path(s).",
                 len(existing_hashes), len(existing_paths))
    else:
        conn            = None
        existing_hashes = set()
        existing_paths  = set()

    # ── Phase 0: Organize (or scan archive for ingest-only) ───────────────────
    if args.ingest_only:
        all_paths         = find_input_images(args.archive_root)
        archive_root_path = Path(args.archive_root).resolve()

        def _rel_path(p: str) -> str:
            try:
                return str(Path(p).resolve().relative_to(archive_root_path)).replace("\\", "/")
            except ValueError:
                return Path(p).name

        # Step 1: filter by relative path — O(1) set lookup, zero disk I/O.
        # Eliminates hashing for files already in the DB (the vast majority on repeat runs).
        path_unknown    = [p for p in all_paths if _rel_path(p) not in existing_paths]
        already_by_path = len(all_paths) - len(path_unknown)
        log.info(
            "Ingest-only: %d total  |  %d already in DB by path  |  %d to hash …",
            len(all_paths), already_by_path, len(path_unknown),
        )
        # Step 2: hash only the path-unknown files, then filter by hash.
        # Catches renamed/moved duplicates whose bytes are already stored.
        new_files = []
        for p in tqdm(path_unknown, desc="Hashing new files", unit="file"):
            h = sha256_file(p)
            if h in existing_hashes:
                log.debug("Hash already in DB (renamed/moved) — skip: %s", p)
                continue
            new_files.append((p, get_photo_date(p), h))
    else:
        new_files = organize_photos(
            input_dir=args.input_dir,
            archive_root=args.archive_root,
            existing_hashes=existing_hashes,
            dry_run=args.dry_run,
        )

    if args.organize_only or args.dry_run:
        log.info("Stopping after organize phase (--organize-only / --dry-run).")
        sys.exit(0)

    # ── Phases 1-4: Caption + Embed + DB ──────────────────────────────────────
    if not new_files:
        log.info("No new photos to process.")
    else:
        ingest(
            new_files=new_files,
            archive_root=args.archive_root,
            conn=conn,
            existing_paths=existing_paths,
            existing_hashes=existing_hashes,
            checkpoint_size=args.checkpoint,
        )

    if conn:
        conn.close()
