#!/usr/bin/env python3
"""
Find duplicate photos (same SHA-256, different filepath/extension case),
keep the best copy (lowercase ext, earliest ingested), delete the rest
from both the filesystem and Postgres.

Usage:
    python dedup_cleanup.py --archive-root /path/to/archive
    python dedup_cleanup.py --archive-root /path/to/archive --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


def find_duplicates(conn) -> list[list[dict]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT id, filepath, filename, content_hash, ingested_at
            FROM photos
            WHERE content_hash IN (
                SELECT content_hash FROM photos GROUP BY content_hash HAVING COUNT(*) > 1
            )
            ORDER BY content_hash, ingested_at
        """)
        rows = cur.fetchall()

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["content_hash"], []).append(row)
    return list(groups.values())


def pick_keeper(group: list[dict]) -> dict:
    def score(row):
        ext = Path(row["filepath"]).suffix
        return (0 if ext == ext.lower() else 1, row["ingested_at"])
    return min(group, key=score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PHOTO_DB_DSN", "postgresql://claude1:claude1@localhost/photo_archive"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no deletes.")
    args = parser.parse_args()

    archive_root = Path(args.archive_root).resolve()
    conn = psycopg2.connect(args.dsn)

    groups = find_duplicates(conn)
    print(f"Found {len(groups)} duplicate group(s).\n")

    deleted_files = 0
    deleted_rows  = 0
    errors        = 0

    for group in groups:
        keeper   = pick_keeper(group)
        to_delete = [r for r in group if r["id"] != keeper["id"]]

        print(f"Hash: {keeper['content_hash'][:16]}…")
        print(f"  KEEP  [{keeper['id']}] {keeper['filepath']}")

        for row in to_delete:
            abs_path = archive_root / row["filepath"]
            print(f"  DEL   [{row['id']}] {row['filepath']}")

            if args.dry_run:
                continue

            if abs_path.exists():
                try:
                    abs_path.unlink()
                    deleted_files += 1
                except Exception as e:
                    print(f"    ERROR deleting file: {e}", file=sys.stderr)
                    errors += 1
            else:
                print(f"    WARNING: file not found on disk: {abs_path}")

            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM photos WHERE id = %s", (row["id"],))
                conn.commit()
                deleted_rows += 1
            except Exception as e:
                conn.rollback()
                print(f"    ERROR deleting DB row: {e}", file=sys.stderr)
                errors += 1

        print()

    conn.close()

    if args.dry_run:
        print("Dry-run complete — no changes made.")
    else:
        print(f"Done — deleted {deleted_files} file(s), {deleted_rows} DB row(s), {errors} error(s).")


if __name__ == "__main__":
    main()
