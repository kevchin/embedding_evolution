#!/usr/bin/env python3
"""
Streamlit photo search app.
Searches by caption similarity (pgvector cosine) + optional path substring.

Usage:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message="Accessing `__path__` from",
    category=UserWarning,
)
import logging
logging.getLogger("transformers.utils.import_utils").setLevel(logging.ERROR)


import streamlit as st
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageOps

# Register HEIC/HEIF support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

st.set_page_config(page_title="Photo Search", layout="wide")

# ── CLI argument parsing ────────────────────────────────────────────────────
# Streamlit intercepts argparse, so we parse sys.argv manually.
# Usage: streamlit run stream5.py -- --db "postgresql://..." --root "/path/to/photos"
import sys

def _parse_cli_arg(flag: str) -> str | None:
    """Return the value after `flag` in sys.argv, or None if not present."""
    args = sys.argv[1:]
    try:
        idx = args.index(flag)
        return args[idx + 1]
    except (ValueError, IndexError):
        return None

_cli_db   = _parse_cli_arg("--db")
_cli_root = _parse_cli_arg("--root")

# Priority: CLI arg > env var > empty string (user fills in via UI)
_default_db   = _cli_db   or os.environ.get("PHOTO_DB_CONN", "")
_default_root = _cli_root or os.environ.get("PHOTO_ROOT", "")

#EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
DEFAULT_RESULTS = 24

# ── Sidebar config ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    pg_conn_str = st.text_input(
        "PostgreSQL connection string",
        value=_default_db,
        placeholder="postgresql://user:pass@localhost/photos",
        type="password",
    )
    photo_root = st.text_input(
        "Photo root directory",
        value=_default_root,
        placeholder="/mnt/ssd/photos",
    )
    result_count = st.slider("Max results", 6, 96, DEFAULT_RESULTS, step=6)
    min_score = st.slider(
        "Min similarity score", 0.0, 1.0, 0.0, step=0.05,
        help="Lower this if semantic search returns 0 results. 0.0 returns all results ranked by similarity."
    )
    st.divider()

    # ── Connect & Verify button ────────────────────────────────────────────
    connect_clicked = st.button("Connect & Verify", type="primary", width="content")

    if connect_clicked:
        all_ok = True

        # 1. Check photo root directory
        root_path = Path(photo_root)
        if root_path.exists() and root_path.is_dir():
            try:
                entries = list(root_path.iterdir())
                st.success(f"✅ Photo root accessible  \n`{photo_root}`  \n{len(entries)} top-level entries")
            except PermissionError:
                st.error(f"❌ Permission denied: `{photo_root}`")
                all_ok = False
        else:
            st.error(f"❌ Directory not found: `{photo_root}`")
            all_ok = False

        # 2. Check database connection
        try:
            conn_test = psycopg2.connect(pg_conn_str, connect_timeout=5)
            with conn_test.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM photos WHERE caption IS NOT NULL AND embedding IS NOT NULL;")
                (row_count,) = cur.fetchone()
            conn_test.close()
            st.success(f"✅ Database connected  \n{row_count:,} indexed photos with embeddings")
        except psycopg2.OperationalError as e:
            st.error(f"❌ DB connection failed:  \n`{e}`")
            all_ok = False
        except psycopg2.errors.UndefinedTable:
            st.error("❌ Table `photos` not found in the database.")
            all_ok = False
        except Exception as e:
            st.error(f"❌ Unexpected DB error:  \n`{e}`")
            all_ok = False

        # 3. HEIC warning
        if not HEIC_SUPPORT:
            st.warning("⚠️ HEIC support unavailable.  \nInstall with: `pip install pillow-heif`")

        if all_ok:
            st.info("✅ Ready to search!")

    st.caption(
        "Pre-fill via CLI: `streamlit run stream5.py -- --db <conn> --root <path>`  \n"
        "Or set env vars: `PHOTO_DB_CONN` and `PHOTO_ROOT`"
    )

root = Path(photo_root)


# ── Cache DB connection and embedder ───────────────────────────────────────
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_ID)

@st.cache_resource
def get_conn(conn_str: str):
    return psycopg2.connect(conn_str)

def open_image_safe(path: Path):
    """Open an image, applying EXIF orientation correction automatically."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # ← fixes rotation/flip from EXIF tag
        img.load()
        if img.width <= 0 or img.height <= 0:
            return None, f"Invalid dimensions: {img.width}x{img.height}"
        return img, None
    except Exception as e:
        return None, str(e)

def open_image_safe_old(path: Path):
    """Open an image, returning (img, error_str). Forces full decode to catch
    corrupt files early. Handles HEIC if pillow-heif is installed."""
    try:
        img = Image.open(path)
        img.load()  # Force full decode — catches corrupt/truncated files before display
        if img.width <= 0 or img.height <= 0:
            return None, f"Invalid dimensions: {img.width}x{img.height}"
        return img, None
    except Exception as e:
        return None, str(e)


# ── Main UI ────────────────────────────────────────────────────────────────
st.title("Photo Archive Search")

col1, col2 = st.columns([3, 1])
with col1:
    caption_query = st.text_input(
        "Caption search",
        placeholder="kids playing at the beach",
    )
with col2:
    path_filter = st.text_input(
        "Path contains",
        placeholder="2018",
    )

search_mode = st.radio(
    "Caption match mode",
    ["Semantic similarity", "Exact string match"],
    horizontal=True,
)

show_debug = st.checkbox("Show debug scores", value=False,
                         help="Shows the top 5 raw similarity scores — useful if you're getting 0 results")

search_clicked = st.button("Search", type="primary")

if search_clicked and caption_query:
    try:
        conn = get_conn(pg_conn_str)
        embedder = get_embedder()

        path_like = f"%{path_filter}%" if path_filter else "%"

        if search_mode == "Exact string match":
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT rel_path, caption,
                           1.0 AS score
                    FROM photos
                    WHERE caption ILIKE %s
                      AND rel_path ILIKE %s
                      AND caption IS NOT NULL
                    ORDER BY rel_path
                    LIMIT %s
                """, (f"%{caption_query}%", path_like, result_count))
                rows = cur.fetchall()

        else:
            # Semantic similarity search
            # Change this at the top
	    	# Change the encode call in the search section
            query_vec = embedder.encode(
	    	f"Represent this sentence for searching relevant passages: {caption_query}",
			normalize_embeddings=True).tolist()

            vec_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

            # Optional debug: show top 5 raw scores before any filter
            if show_debug:
                with st.expander("🔍 Debug: top 5 raw similarity scores (no filters applied)"):
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT rel_path, caption,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM photos
                            WHERE caption IS NOT NULL AND embedding IS NOT NULL
                            ORDER BY embedding <=> %s::vector
                            LIMIT 5
                        """, (vec_str, vec_str))
                        debug_rows = cur.fetchall()
                    for r in debug_rows:
                        st.write(f"`{r['score']:.4f}` — {r['rel_path']}")
                        st.caption((r["caption"] or "")[:300])

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT rel_path, caption,
                           1 - (embedding <=> %s::vector) AS score
                    FROM photos
                    WHERE rel_path ILIKE %s
                      AND caption IS NOT NULL
                      AND embedding IS NOT NULL
                      AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vec_str, path_like, vec_str, min_score, vec_str, result_count))
                rows = cur.fetchall()

        # ── Result summary ─────────────────────────────────────────────────
        st.caption(
            f"{len(rows)} results for **{caption_query}**"
            + (f" in paths matching **{path_filter}**" if path_filter else "")
            + f" · mode: {search_mode}"
            + (f" · min score: {min_score}" if search_mode == "Semantic similarity" else "")
        )

        if not rows:
            st.info(
                "No results found. Try:\n"
                "- Lowering the **Min similarity score** slider to 0.0\n"
                "- Enabling **Show debug scores** to see what scores your top matches actually get\n"
                "- Switching to **Exact string match** to confirm captions exist"
            )

        # ── Display grid ───────────────────────────────────────────────────
        cols = st.columns(3)
        load_errors = []

        for i, row in enumerate(rows):
            abs_path = root / row["rel_path"]
            score = row["score"]
            caption_text = row["caption"] or ""

            with cols[i % 3]:
                if abs_path.exists():
                    img, err = open_image_safe(abs_path)
                    if img:
                        if img.width > 0 and img.height > 0:
                            st.image(img, width="stretch", caption=None)
                        else:
                            st.warning(f"⚠️ Invalid image dimensions ({img.width}x{img.height})")
                            load_errors.append((row["rel_path"], f"Invalid dimensions: {img.width}x{img.height}"))
                    else:
                        st.warning(f"⚠️ Can't display image  \n`{abs_path.suffix.upper()}` format unsupported or file corrupt")
                        load_errors.append((row["rel_path"], err))

                    st.caption(
                        f"`{row['rel_path']}`  \n"
                        f"similarity: **{score:.3f}**"
                    )
                    with st.expander("Caption"):
                        st.write(caption_text)

                    with open(abs_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download",
                            data=f,
                            file_name=abs_path.name,
                            mime="image/jpeg",
                            key=f"dl_{i}",
                        )
                else:
                    st.warning(f"File not found: `{row['rel_path']}`")
                    st.caption(f"`{row['rel_path']}`")

        # ── Load error summary ─────────────────────────────────────────────
        if load_errors:
            with st.expander(f"⚠️ {len(load_errors)} image(s) could not be displayed"):
                for path, err in load_errors:
                    st.text(f"{path}: {err}")
                if not HEIC_SUPPORT:
                    st.info("HEIC files require `pillow-heif`. Run: `pip install pillow-heif`")

    except Exception as e:
        st.error(f"Search error: {e}")
        st.exception(e)

elif search_clicked:
    st.warning("Enter a caption query to search.")
