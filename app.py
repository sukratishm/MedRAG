import os
import shutil
import random
from pathlib import Path

import streamlit as st
from PIL import Image

from visual_search import VisualSearchEngine


APP_TITLE = "MedRAG Visual Search"


def _get_paths() -> tuple[Path, Path]:
    repo_index = Path("index").resolve()
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/medrag_data")).resolve()
    index_dir = Path(os.getenv("INDEX_DIR", data_dir / "index")).resolve()
    return repo_index, index_dir


def _ensure_index_available() -> Path:
    repo_index, index_dir = _get_paths()
    if index_dir.exists():
        return index_dir
    if repo_index.exists():
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(repo_index, index_dir)
        return index_dir
    raise FileNotFoundError(
        "FAISS index not found. Expected at /var/data/index or ./index"
    )


@st.cache_resource(show_spinner=True)
def _load_engine() -> VisualSearchEngine:
    index_dir = _ensure_index_available()
    return VisualSearchEngine(index_dir=index_dir, device="auto", top_k=5)

def _pick_sample_image(data_dir: Path) -> Path | None:
    images_dir = data_dir / "images"
    if not images_dir.exists():
        return None
    candidates = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_results(results):
    for r in results:
        cols = st.columns([1, 3])
        with cols[0]:
            if r.filepath and Path(r.filepath).exists():
                try:
                    img = Image.open(r.filepath).convert("RGB")
                    st.image(img, use_column_width=True)
                except Exception:
                    st.caption("Image preview unavailable")
            else:
                st.caption("Image file not present")
        with cols[1]:
            st.markdown(f"**Rank {r.rank}**")
            st.write(f"Similarity: {r.similarity:.3f}")
            st.write(f"Filename: {r.filename}")
            st.write(f"Labels: {r.labels}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload a chest X-ray to retrieve similar cases.")

    with st.sidebar:
        st.markdown("**Index Status**")
        try:
            index_dir = _ensure_index_available()
            st.write(f"Index dir: `{index_dir}`")
            data_dir = index_dir.parent
        except FileNotFoundError as e:
            st.error(str(e))
            return

        st.markdown("**Settings**")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5, step=1)
        sample_clicked = st.button("Use Sample Image")
        sample_path = _pick_sample_image(data_dir) if sample_clicked else None

    uploaded = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg"])
    if not uploaded and not sample_path:
        st.info("Upload an image or click “Use Sample Image”.")
        return

    try:
        if sample_path:
            query_img = Image.open(sample_path).convert("RGB")
        else:
            query_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    st.subheader("Query Image")
    st.image(query_img, use_column_width=True)

    with st.spinner("Loading model and searching..."):
        engine = _load_engine()
        results = engine.search(query_img, top_k=top_k, load_images=False)

    st.subheader("Similar Cases")
    if not results:
        st.warning("No results found.")
        return
    _render_results(results)


if __name__ == "__main__":
    main()
