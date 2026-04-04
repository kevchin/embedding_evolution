#!/usr/bin/env python3
"""
Phase 2: Walk the photo archive and ingest captions + embeddings.
Idempotent — skips photos that already have a caption.
Resumes automatically if interrupted.

Usage:
    python 02_ingest.py \
        --root /mnt/ssd/photos \
        --conn "postgresql://user:pass@localhost/photos" \
        [--limit 20]          # optional: stop after N photos (for testing)
        [--dry-run]           # scan only, no model loading
        [--workers 1]         # DB write workers (keep at 1 for GPU memory)
"""

import os
import gc
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", message="Unrecognized keys in `rope_parameters`")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import torch
import psycopg2
import psycopg2.extras
import pillow_heif
from PIL import Image
from tqdm import tqdm
from rich.console import Console
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration

pillow_heif.register_heif_opener()
console = Console()

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
QWEN_REPO_ID   = "Qwen/Qwen2-VL-7B-Instruct"
CAPTION_PROMPT = "Describe this image in detail."
SUPPORTED_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def find_local_model(repo_id: str) -> str:
    folder = "models--" + repo_id.replace("/", "--")
    snap_dir = HF_CACHE / folder / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(f"Model not in cache: {snap_dir}")
    return str(sorted(snap_dir.iterdir(), key=lambda p: p.stat().st_mtime)[-1])

def walk_photos(root: Path):
    """Yield (rel_path_str, abs_path) for all supported images under root."""
    for abs_path in sorted(root.rglob("*")):
        if abs_path.suffix.lower() in SUPPORTED_EXTS and abs_path.is_file():
            rel = abs_path.relative_to(root)
            yield str(rel), abs_path

def get_uncaptioned(conn, all_rel_paths: list[str]) -> list[str]:
    """Return rel_paths not yet in DB or in DB with NULL caption."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rel_path FROM photos
            WHERE rel_path = ANY(%s) AND caption IS NOT NULL
        """, (all_rel_paths,))
        done = {row[0] for row in cur.fetchall()}
    return [p for p in all_rel_paths if p not in done]

def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    return img

def upsert_photo(conn, rel_path: str, caption: str, embedding: list[float]):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO photos (rel_path, caption, embedding, captioned_at)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (rel_path) DO UPDATE
                SET caption      = EXCLUDED.caption,
                    embedding    = EXCLUDED.embedding,
                    captioned_at = EXCLUDED.captioned_at
        """, (rel_path, caption, embedding, datetime.now(timezone.utc)))
    conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Photo archive root directory")
    parser.add_argument("--conn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N photos (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan only, do not load models or write to DB")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        console.print(f"[red]Root not found: {root}")
        return

    # ── Scan all photos ──────────────────────────────────────────────────────
    console.print(f"[bold]Scanning {root}...")
    all_photos = list(walk_photos(root))
    console.print(f"Found [bold]{len(all_photos)}[/bold] photos.")

    if args.dry_run:
        console.print("[yellow]Dry run — exiting without processing.")
        return

    # ── Connect and find uncaptioned ─────────────────────────────────────────
    conn = psycopg2.connect(args.conn)
    all_rel = [r for r, _ in all_photos]
    todo_rel = set(get_uncaptioned(conn, all_rel))
    todo = [(r, p) for r, p in all_photos if r in todo_rel]

    if args.limit:
        todo = todo[:args.limit]

    console.print(f"[bold]{len(todo)}[/bold] photos need captioning "
                  f"({len(all_photos) - len(todo)} already done).")

    if not todo:
        console.print("[green]Nothing to do — all photos already captioned.")
        conn.close()
        return

    # ── Load models ───────────────────────────────────────────────────────────
    console.print("[bold cyan]Loading Qwen2-VL-7B...")
    qwen_path = find_local_model(QWEN_REPO_ID)
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_path,
        quantization_config=bnb_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    console.print("[bold cyan]Loading sentence embedder...")
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    # ── Process ───────────────────────────────────────────────────────────────
    errors = []
    for rel_path, abs_path in tqdm(todo, desc="Captioning"):
        try:
            img = load_image(abs_path)

            # Caption
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": CAPTION_PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[img], return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0]
            caption = caption.split("assistant")[-1].strip()

            # Embed
            embedding = embedder.encode(caption, normalize_embeddings=True).tolist()

            # Upsert
            upsert_photo(conn, rel_path, caption, embedding)

        except Exception as e:
            console.print(f"[red]Error on {rel_path}: {e}")
            errors.append((rel_path, str(e)))
            continue

    conn.close()
    gc.collect()
    torch.cuda.empty_cache()

    console.print(f"\n[green]Done. {len(todo) - len(errors)} succeeded, "
                  f"{len(errors)} errors.")
    if errors:
        console.print("[yellow]Failed paths:")
        for p, e in errors:
            console.print(f"  {p}: {e}")

if __name__ == "__main__":
    main()
