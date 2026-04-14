#!/usr/bin/env python3
"""
Look up an image file in the Postgres photo archive by SHA-256 hash.

Usage:
    python lookup_by_hash.py /path/to/image.jpg
    python lookup_by_hash.py /path/to/image.jpg --dsn "postgresql://claude1:claude1@localhost/photo_archive"
"""

import argparse
import hashlib
import os
import sys

import psycopg2
from psycopg2.extras import RealDictCursor


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def lookup_hash(conn, content_hash: str) -> list[dict]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, filepath, filename, photo_date, file_size_bytes, width, height,
                   ingested_at, caption
            FROM photos
            WHERE content_hash = %s
            ORDER BY ingested_at
            """,
            (content_hash,),
        )
        return cur.fetchall()


def main():
    parser = argparse.ArgumentParser(description="Find a file in the photo DB by SHA-256.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PHOTO_DB_DSN", "postgresql://claude1:claude1@localhost/photo_archive"),
        help="Postgres DSN (or set PHOTO_DB_DSN env var).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: File not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Computing SHA-256 for: {args.image}")
    digest = sha256_file(args.image)
    print(f"SHA-256: {digest}\n")

    conn = psycopg2.connect(args.dsn)
    try:
        rows = lookup_hash(conn, digest)
    finally:
        conn.close()

    if not rows:
        print("No matching entries found in the database.")
        sys.exit(0)

    print(f"Found {len(rows)} matching row(s):\n")
    for row in rows:
        print(f"  id          : {row['id']}")
        print(f"  filepath    : {row['filepath']}")
        print(f"  filename    : {row['filename']}")
        print(f"  photo_date  : {row['photo_date']}")
        print(f"  dimensions  : {row['width']}x{row['height']}")
        print(f"  size_bytes  : {row['file_size_bytes']}")
        print(f"  ingested_at : {row['ingested_at']}")
        if row["caption"]:
            print(f"  caption     : {row['caption'][:120]}{'...' if len(row['caption'] or '') > 120 else ''}")
        print()


if __name__ == "__main__":
    main()
