#!/usr/bin/env python3
"""
Phase 3: Reset captions so they will be re-processed on next ingest run.
Does NOT delete rows — just NULLs the caption and embedding.

Usage:
    # Reset everything
    python 03_reset.py --conn "..." --all

    # Reset only a path substring (e.g. one year)
    python 03_reset.py --conn "..." --path-contains "2018"

    # Completely drop and recreate the table (nuclear option)
    python 03_reset.py --conn "..." --drop
"""

import argparse
import psycopg2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Reset all captions")
    group.add_argument("--path-contains", type=str,
                       help="Reset rows where rel_path contains this string")
    group.add_argument("--drop", action="store_true",
                       help="DROP the photos table entirely (nuclear reset)")
    args = parser.parse_args()

    with psycopg2.connect(args.conn) as conn:
        with conn.cursor() as cur:
            if args.drop:
                confirm = input("Type 'yes' to DROP the photos table: ")
                if confirm.strip().lower() == "yes":
                    cur.execute("DROP TABLE IF EXISTS photos")
                    conn.commit()
                    print("Table dropped. Re-run 01_setup_db.py to recreate.")
                else:
                    print("Aborted.")

            elif args.all:
                cur.execute("""
                    UPDATE photos
                    SET caption = NULL, embedding = NULL, captioned_at = NULL
                """)
                conn.commit()
                print(f"Reset {cur.rowcount} rows.")

            elif args.path_contains:
                cur.execute("""
                    UPDATE photos
                    SET caption = NULL, embedding = NULL, captioned_at = NULL
                    WHERE rel_path ILIKE %s
                """, (f"%{args.path_contains}%",))
                conn.commit()
                print(f"Reset {cur.rowcount} rows matching '{args.path_contains}'.")

if __name__ == "__main__":
    main()
