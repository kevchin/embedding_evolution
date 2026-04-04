#!/usr/bin/env python3
"""
Phase 1: Set up the database schema.
Safe to re-run — uses CREATE IF NOT EXISTS everywhere (idempotent).

Usage:
    python 01_setup_db.py --conn "postgresql://user:pass@localhost/photos"
"""

import argparse
import psycopg2

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS photos (
    id            SERIAL PRIMARY KEY,
    rel_path      TEXT NOT NULL UNIQUE,   -- e.g. 2018/06/12/IMG_7274.jpg
    caption       TEXT,                   -- NULL = not yet captioned
    embedding     vector(1024),           -- BAAI/bge-large-en-v1.5 dim=1024
    captioned_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS photos_embedding_idx
    ON photos USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS photos_rel_path_idx
    ON photos (rel_path text_pattern_ops);
"""

TEST_QUERY = """
SELECT rel_path, caption,
       embedding <=> '[0.1, 0.2, 0.3]'::vector(3) AS dist
FROM photos LIMIT 1;
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", required=True,
                        help="PostgreSQL connection string")
    args = parser.parse_args()

    print(f"Connecting to database...")
    with psycopg2.connect(args.conn) as conn:
        with conn.cursor() as cur:
            print("Creating schema...")
            cur.execute(SCHEMA)
            conn.commit()
            print("Schema created (idempotent — safe to re-run).")

            # Verify pgvector is working
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            if row:
                print(f"pgvector version: {row[0]} — OK")
            else:
                print("WARNING: pgvector extension not found!")

            cur.execute("SELECT COUNT(*) FROM photos")
            count = cur.fetchone()[0]
            print(f"Photos table: {count} rows currently.")

    print("Setup complete.")

if __name__ == "__main__":
    main()
