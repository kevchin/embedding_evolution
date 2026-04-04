#!/usr/bin/env python3
"""
Check ingestion progress.

Usage:
    python 04_status.py --conn "postgresql://user:pass@localhost/photos"
"""

import argparse
import psycopg2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", required=True)
    args = parser.parse_args()

    with psycopg2.connect(args.conn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM photos")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM photos WHERE caption IS NOT NULL")
            done = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM photos WHERE caption IS NULL")
            pending = cur.fetchone()[0]

            cur.execute("""
                SELECT DATE(captioned_at), COUNT(*)
                FROM photos
                WHERE captioned_at IS NOT NULL
                GROUP BY DATE(captioned_at)
                ORDER BY DATE(captioned_at) DESC
                LIMIT 7
            """)
            daily = cur.fetchall()

    print(f"\nTotal rows:     {total}")
    print(f"Captioned:      {done}  ({done/max(total,1)*100:.1f}%)")
    print(f"Pending:        {pending}")
    if daily:
        print("\nRecent activity:")
        for date, count in daily:
            print(f"  {date}: {count} photos captioned")

if __name__ == "__main__":
    main()
