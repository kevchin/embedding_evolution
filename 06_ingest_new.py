#!/usr/bin/env python3
"""
Incremental photo ingest — captions and embeds only NEW photos.
Idempotent: safe to re-run at any time, will never re-process already captioned photos.

Strategies (pick one per run):
  --since YYYY-MM-DD     Only scan subdirectories modified on or after this date
  --since-days N         Only scan subdirectories modified within the last N days
  --paths a/b c/d        Only scan specific subdirectories under root
  --full                 Scan entire archive (same as original 02_ingest.py)

Examples:
  # Ingest everything added today
  python 06_ingest_new.py --root /mnt/ssd/photos --conn "$DB" --since 2026-03-29

  # Ingest everything added in the last 7 days
  python 06_ingest_new.py --root /mnt/ssd/photos --conn "$DB" --since-days 7

  # Ingest a specific folder you just added
  python 06_ingest_new.py --root /mnt/ssd/photos --conn "$DB" --paths 2026/03/29

  # Full scan (finds anything missing, regardless of date)
  python 06_ingest_new.py --root /mnt/ssd/photos --conn "$DB" --full

  # Dry run — see what would be ingested without doing anything
  python 06_ingest_new.py --root /mnt/ssd/photos --conn "$DB" --since-days 7 --dry-run
"""

import os
import gc
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
from rich.table import Table as RichTable
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration

pillow_heif.register_heif_opener()
console = Console()

HF_CACHE       = Path.home() / ".cache" / "huggingface" / "hub"
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


# ── Model loading ─────────────────────────────────────────────────────────────

def find_local_model(repo_id: str) -> str:
    folder = "models--" + repo_id.replace("/", "--")
    snap_dir = HF_CACHE / folder / "snapshots"
    if not snap_dir.exists():
        raise FileNotFoundError(f"Model not in cache: {snap_dir}")
    return str(sorted(snap_dir.iterdir(), key=lambda p: p.stat().st_mtime)[-1])


# ── Photo discovery ───────────────────────────────────────────────────────────

def walk_subtree(root: Path, subdir: Path) -> list[tuple[str, Path]]:
    """Yield (rel_path_str, abs_path) for all supported images under subdir."""
    results = []
    for abs_path in sorted(subdir.rglob("*")):
        if abs_path.suffix.lower() in SUPPORTED_EXTS and abs_path.is_file():
            rel = abs_path.relative_to(root)
            results.append((str(rel), abs_path))
    return results


def find_modified_subdirs(root: Path, since: datetime) -> list[Path]:
    """
    Return top-level subdirectories (e.g. 2026/03/29) whose mtime is >= since.
    Walks up to 4 levels deep to find date-structured folders.
    Only returns leaf directories that actually contain photos.
    """
    since_ts = since.timestamp()
    candidates = []

    def _walk(path: Path, depth: int):
        if depth > 4:
            return
        try:
            if path.stat().st_mtime >= since_ts:
                # Check if it contains any photos directly
                has_photos = any(
                    f.suffix.lower() in SUPPORTED_EXTS
                    for f in path.iterdir()
                    if f.is_file()
                )
                if has_photos:
                    candidates.append(path)
                # Always recurse to find nested new dirs
                for child in sorted(path.iterdir()):
                    if child.is_dir():
                        _walk(child, depth + 1)
        except PermissionError:
            pass

    for child in sorted(root.iterdir()):
        if child.is_dir():
            _walk(child, 1)

    return candidates


def get_already_captioned(conn, rel_paths: list[str]) -> set[str]:
    """Return rel_paths that already have a caption in the DB."""
    if not rel_paths:
        return set()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rel_path FROM photos
            WHERE rel_path = ANY(%s) AND caption IS NOT NULL
        """, (rel_paths,))
        return {row[0] for row in cur.fetchall()}


# ── Processing ────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incrementally ingest new photos into the caption/embedding database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root",  required=True, help="Photo archive root directory")
    parser.add_argument("--conn",  required=True, help="PostgreSQL connection string")

    # Scan strategy — mutually exclusive
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--since", metavar="YYYY-MM-DD",
        help="Scan subdirectories modified on or after this date"
    )
    scope.add_argument(
        "--since-days", type=int, metavar="N",
        help="Scan subdirectories modified within the last N days"
    )
    scope.add_argument(
        "--paths", nargs="+", metavar="SUBDIR",
        help="Scan specific subdirectories under root (e.g. 2026/03/29)"
    )
    scope.add_argument(
        "--full", action="store_true",
        help="Scan entire archive (finds anything not yet captioned)"
    )

    parser.add_argument("--limit",   type=int, default=None,
                        help="Stop after N photos (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and report what would be ingested, without processing")
    parser.add_argument("--embed-model", default=EMBED_MODEL_ID,
                        help=f"Sentence transformer model ID (default: {EMBED_MODEL_ID})")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        console.print(f"[red]Root not found: {root}")
        return

    # ── Resolve scan scope ────────────────────────────────────────────────────
    if args.full:
        console.print(f"[bold]Full scan of {root}...")
        all_photos = walk_subtree(root, root)

    elif args.paths:
        all_photos = []
        for rel_sub in args.paths:
            subdir = root / rel_sub
            if not subdir.exists():
                console.print(f"[yellow]Skipping missing path: {subdir}")
                continue
            found = walk_subtree(root, subdir)
            console.print(f"  {rel_sub}: {len(found)} photos found")
            all_photos.extend(found)

    else:
        if args.since:
            since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)

        console.print(f"[bold]Scanning for directories modified since {since_dt.date()}...")
        modified_dirs = find_modified_subdirs(root, since_dt)

        if not modified_dirs:
            console.print("[yellow]No recently modified directories found. "
                          "Try --since with an earlier date or use --full.")
            return

        console.print(f"Found [bold]{len(modified_dirs)}[/bold] recently modified directories:")
        all_photos = []
        for d in modified_dirs:
            rel = d.relative_to(root)
            found = walk_subtree(root, d)
            console.print(f"  {rel}: {len(found)} photos")
            all_photos.extend(found)

    console.print(f"\n[bold]{len(all_photos)}[/bold] total photos in scope.")

    if not all_photos:
        console.print("[green]Nothing to scan.")
        return

    # ── Connect and filter already-captioned ──────────────────────────────────
    console.print("[bold]Connecting to database...")
    try:
        conn = psycopg2.connect(args.conn, connect_timeout=5)
    except psycopg2.OperationalError as e:
        console.print(f"[red]DB connection failed: {e}")
        return

    all_rel = [r for r, _ in all_photos]
    already_done = get_already_captioned(conn, all_rel)
    todo = [(r, p) for r, p in all_photos if r not in already_done]

    if args.limit:
        todo = todo[:args.limit]

    # ── Summary table ─────────────────────────────────────────────────────────
    tbl = RichTable(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Count", justify="right")
    tbl.add_row("Photos in scope",       str(len(all_photos)))
    tbl.add_row("Already captioned",     str(len(already_done)))
    tbl.add_row("Need captioning",       str(len(todo)),)
    if args.limit and len(todo) > args.limit:
        tbl.add_row("Capped by --limit", str(args.limit))
    console.print(tbl)

    if args.dry_run:
        if todo:
            console.print("\n[yellow]Dry run — these would be ingested:")
            for rel, _ in todo[:20]:
                console.print(f"  {rel}")
            if len(todo) > 20:
                console.print(f"  ... and {len(todo) - 20} more")
        else:
            console.print("[green]Dry run — nothing to ingest.")
        conn.close()
        return

    if not todo:
        console.print("[green]Nothing to do — all scanned photos already captioned.")
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

    console.print(f"[bold cyan]Loading sentence embedder ({args.embed_model})...")
    embedder = SentenceTransformer(args.embed_model)

    # ── Process ───────────────────────────────────────────────────────────────
    errors = []
    for rel_path, abs_path in tqdm(todo, desc="Captioning"):
        try:
            img = load_image(abs_path)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": CAPTION_PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], return_tensors="pt").to("cuda")

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0]
            caption = caption.split("assistant")[-1].strip()

            embedding = embedder.encode(
                caption, normalize_embeddings=True
            ).tolist()

            upsert_photo(conn, rel_path, caption, embedding)

        except Exception as e:
            console.print(f"[red]Error on {rel_path}: {e}")
            errors.append((rel_path, str(e)))
            continue

    conn.close()
    gc.collect()
    torch.cuda.empty_cache()

    succeeded = len(todo) - len(errors)
    console.print(f"\n[green]Done. {succeeded} succeeded, {len(errors)} errors.")
    if errors:
        console.print("[yellow]Failed paths:")
        for p, e in errors:
            console.print(f"  {p}: {e}")


if __name__ == "__main__":
    main()
