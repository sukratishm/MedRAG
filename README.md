---
title: MedRAG Diagnostic Assistant
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# MedRAG

MedRAG is a multimodal chest X-ray retrieval and diagnostic-assistance app built on:
- BiomedCLIP for image embeddings and zero-shot disease scoring
- FAISS for similar-case retrieval
- a crosscheck layer that combines classifier output with retrieved case evidence
- Streamlit for the application UI

The current app supports:
- chest X-ray upload
- sample-image testing
- similar-case retrieval from the indexed gallery
- zero-shot disease probability ranking
- retrieval-supported clinical assessment text
- Hugging Face Spaces deployment through Docker

## Current App Flow

1. The user uploads a chest X-ray or selects a sample image.
2. The app encodes the image with BiomedCLIP.
3. FAISS retrieves the most visually similar historical cases.
4. BiomedCLIP scores 14 CheXpert disease prompts.
5. A crosscheck step combines retrieval agreement with classifier confidence.
6. The app renders:
   - generated clinical assessment
   - ranked diagnoses
   - top disease probabilities
   - similar historical cases

## Project Files

Core app:
- `app.py` - Streamlit UI and diagnosis pipeline
- `visual_search.py` - FAISS-backed visual search engine
- `download_assets.py` - downloads demo index/images and prefetches BiomedCLIP

Index/data tooling:
- `gallery_builder.py` - build FAISS index from chest X-ray images
- `data_downloader.py` - download source datasets
- `rewrite_metadata.py` - rewrite metadata filepaths for deployment

Research/demo:
- `MedRAG.ipynb` - notebook containing the retrieval, zero-shot classification, and crosscheck logic that the app was ported from

Deployment:
- `Dockerfile` - Hugging Face Spaces container build
- `start.sh` - startup entrypoint for Spaces
- `requirements-space.txt` - CPU-friendly dependencies for Spaces
- `render.yaml` - older Render deployment config

## Hugging Face Spaces

This repo is configured for a Docker Space.

### Deploy steps

1. Create a new Hugging Face Space.
2. Choose `Docker`.
3. Push this repo to the Space remote.
4. Let the Space build and start.

The Space startup does the following:
- installs CPU-only PyTorch
- downloads the public `index.zip` and `images.zip`
- prefetches the BiomedCLIP model
- starts Streamlit on port `7860`

## Local Run

Install dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-space.txt
```

Run the app:

```bash
python download_assets.py
streamlit run app.py
```

## Data Notes

The deployed demo uses a reduced subset of CheXpert so it can run on free CPU infrastructure.

Assets are pulled from public Google Drive links by default:
- FAISS index archive
- subset image archive

If needed, override them with:
- `GDRIVE_INDEX_URL`
- `GDRIVE_IMAGES_URL`

Optional environment variables:
- `DATA_DIR`
- `HF_HOME`
- `PREFETCH_MODEL`

## Limitations

- The app is a diagnostic aid, not a clinical decision system.
- Free-tier hosting will have slow cold starts.
- The generated assessment is rule-based synthesis from model scores and retrieval support, not a physician-grade interpretation.
- The original project plan referenced a larger multi-agent/LLM flow; the current deployed app implements the retrieval + classifier + crosscheck path from the notebook.
