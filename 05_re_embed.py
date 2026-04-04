#!/usr/bin/env python3
"""
Re-embed all photos using a new sentence embedding model.
Run once after changing EMBED_MODEL_ID in the Streamlit app.

Usage:
    python re_embed.py \
        --conn "postgresql://user:pass@localhost/photos" \
        [--model "BAAI/bge-large-en-v1.5"] \
        [--batch-size 64] \
        [--dry-run]
"""

import gc
import argparse
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console

console = Console()

DEFAULT_MODEL      = "BAAI/bge-large-en-v1.5"
DEFAULT_BATCH_SIZE = 64


def get_all_captioned(conn) -> list[tuple[int, str]]:
    """Return (id, caption) for all rows with a non-null caption."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, caption FROM photos WHERE caption IS NOT NULL;")
        return cur.fetchall()


def resize_embedding_column(conn, dims: int):
    """Drop and re-add the embedding column at the required dimension."""
    with conn.cursor() as cur:
        console.print(f"[bold cyan]Resizing embedding column to {dims} dimensions...")
        cur.execute("ALTER TABLE photos DROP COLUMN IF EXISTS embedding;")
        cur.execute(f"ALTER TABLE photos ADD COLUMN embedding vector({dims});")
    conn.commit()


def rebuild_index(conn):
    """Drop and recreate the IVFFlat cosine index."""
    with conn.cursor() as cur:
        console.print("[bold cyan]Rebuilding vector index...")
        cur.execute("DROP INDEX IF EXISTS photos_embedding_idx;")
        cur.execute("""
            CREATE INDEX photos_embedding_idx
            ON photos USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed photo captions with a new sentence transformer model."
    )
    parser.add_argument(
        "--conn", required=True,
        help='PostgreSQL connection string, e.g. "postgresql://user:pass@localhost/photos"'
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Sentence transformer model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Connect and count rows, but do not re-embed or modify the DB"
    )
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    console.print(f"[bold]Connecting to database...")
    try:
        conn = psycopg2.connect(args.conn, connect_timeout=5)
    except psycopg2.OperationalError as e:
        console.print(f"[red]DB connection failed: {e}")
        return

    rows = get_all_captioned(conn)
    console.print(f"Found [bold]{len(rows):,}[/bold] captioned photos to re-embed.")

    if args.dry_run:
        console.print("[yellow]Dry run — exiting without modifying the database.")
        conn.close()
        return

    if not rows:
        console.print("[green]Nothing to do — no captioned photos found.")
        conn.close()
        return

    # ── Load model ────────────────────────────────────────────────────────────
    console.print(f"[bold cyan]Loading embedding model: {args.model}...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}")
        conn.close()
        return

    # Probe output dimension with a test encode
    test_vec = model.encode("test", normalize_embeddings=True)
    dims = len(test_vec)
    console.print(f"Model output dimension: [bold]{dims}[/bold]")

    # ── Resize column ─────────────────────────────────────────────────────────
    resize_embedding_column(conn, dims)

    # ── Embed in batches ──────────────────────────────────────────────────────
    ids      = [r[0] for r in rows]
    captions = [r[1] for r in rows]
    errors   = []

    console.print(f"[bold]Re-embedding {len(rows):,} photos in batches of {args.batch_size}...")

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Embedding"):
        batch_ids      = ids[i : i + args.batch_size]
        batch_captions = captions[i : i + args.batch_size]

        try:
            embeddings = model.encode(
                batch_captions,
                normalize_embeddings=True,
                batch_size=args.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            console.print(f"[red]Embedding failed on batch {i}–{i + args.batch_size}: {e}")
            errors.extend(batch_ids)
            continue

        try:
            with conn.cursor() as cur:
                for row_id, emb in zip(batch_ids, embeddings):
                    vec_str = "[" + ",".join(f"{v:.6f}" for v in emb.tolist()) + "]"
                    cur.execute(
                        "UPDATE photos SET embedding = %s::vector WHERE id = %s",
                        (vec_str, row_id),
                    )
            conn.commit()
        except Exception as e:
            console.print(f"[red]DB write failed on batch {i}–{i + args.batch_size}: {e}")
            conn.rollback()
            errors.extend(batch_ids)

    # ── Rebuild index ─────────────────────────────────────────────────────────
    rebuild_index(conn)

    conn.close()
    gc.collect()

    succeeded = len(rows) - len(errors)
    console.print(f"\n[green]Done. {succeeded:,} succeeded, {len(errors):,} errors.")
    if errors:
        console.print(f"[yellow]Failed IDs: {errors}")


if __name__ == "__main__":
    main()
