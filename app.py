import os
import random
import shutil
from collections import Counter
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from visual_search import VisualSearchEngine


APP_TITLE = "Multimodal Medical RAG Diagnostic Assistant"
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DISEASE_PROMPTS = {
    "No Finding": "Chest X-ray with no abnormality, normal findings",
    "Enlarged Cardiomediastinum": "Chest X-ray showing enlarged cardiomediastinum",
    "Cardiomegaly": "Chest X-ray showing cardiomegaly, enlarged heart",
    "Lung Opacity": "Chest X-ray showing lung opacity",
    "Lung Lesion": "Chest X-ray showing lung lesion or mass",
    "Edema": "Chest X-ray showing pulmonary edema, fluid in lungs",
    "Consolidation": "Chest X-ray showing consolidation in lung",
    "Pneumonia": "Chest X-ray showing pneumonia, lung infection",
    "Atelectasis": "Chest X-ray showing atelectasis, collapsed lung",
    "Pneumothorax": "Chest X-ray showing pneumothorax, air in pleural space",
    "Pleural Effusion": "Chest X-ray showing pleural effusion, fluid around lung",
    "Pleural Other": "Chest X-ray showing pleural abnormality",
    "Fracture": "Chest X-ray showing rib fracture or bone fracture",
    "Support Devices": "Chest X-ray showing support devices, tubes or lines",
}
INPUT_GUARDRAIL_PROMPTS = {
    "Chest X-ray": "A diagnostic chest X-ray radiograph showing the thorax and lungs",
    "Portrait Photo": "A portrait photograph of a person or celebrity",
    "Animal Photo": "A natural photograph of an animal or pet",
    "Document Screenshot": "A screenshot of a document, website, or computer interface",
    "Natural Image": "A normal everyday color photograph of a scene or object",
}
SYNONYMS = {
    "Pleural Effusion": ["pleural fluid", "fluid around lung", "effusion"],
    "Cardiomegaly": ["enlarged heart", "cardiac enlargement"],
    "Pneumonia": ["lung infection", "consolidation"],
    "Edema": ["fluid in lungs", "pulmonary edema"],
    "Atelectasis": ["collapsed lung", "lung collapse"],
    "Lung Opacity": ["opacity", "haziness", "infiltrate"],
    "No Finding": ["normal", "no abnormality", "clear"],
}


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
    raise FileNotFoundError("FAISS index not found. Expected at DATA_DIR/index or ./index")


@st.cache_resource(show_spinner=True)
def _load_engine() -> VisualSearchEngine:
    index_dir = _ensure_index_available()
    return VisualSearchEngine(index_dir=index_dir, device="auto", top_k=5)


@st.cache_resource(show_spinner=False)
def _load_text_features() -> tuple[list[str], torch.Tensor]:
    engine = _load_engine()
    tokenizer = __import__("open_clip").get_tokenizer(MODEL_ID)
    with torch.no_grad():
        tokens = tokenizer(list(DISEASE_PROMPTS.values())).to(engine.device)
        text_features = engine._model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return list(DISEASE_PROMPTS.keys()), text_features


@st.cache_resource(show_spinner=False)
def _load_guardrail_features() -> tuple[list[str], torch.Tensor]:
    engine = _load_engine()
    tokenizer = __import__("open_clip").get_tokenizer(MODEL_ID)
    with torch.no_grad():
        tokens = tokenizer(list(INPUT_GUARDRAIL_PROMPTS.values())).to(engine.device)
        text_features = engine._model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return list(INPUT_GUARDRAIL_PROMPTS.keys()), text_features


def _pick_sample_image(data_dir: Path) -> Path | None:
    images_dir = data_dir / "images"
    if not images_dir.exists():
        return None
    candidates = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg"))
    if not candidates:
        return None
    return random.choice(candidates)


@torch.no_grad()
def _predict_diseases(image: Image.Image) -> dict[str, float]:
    engine = _load_engine()
    disease_names, text_features = _load_text_features()
    tensor = engine._transform(image.convert("RGB")).unsqueeze(0).to(engine.device)
    image_features = engine._model.encode_image(tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(similarities * 100, dim=0).detach().cpu().tolist()
    results = {
        disease_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(disease_names))
    }
    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))


@torch.no_grad()
def _validate_input_image(image: Image.Image) -> tuple[bool, dict[str, float]]:
    engine = _load_engine()
    labels, text_features = _load_guardrail_features()
    tensor = engine._transform(image.convert("RGB")).unsqueeze(0).to(engine.device)
    image_features = engine._model.encode_image(tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(similarities * 100, dim=0).detach().cpu().tolist()
    scores = {labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(labels))}
    chest_score = scores["Chest X-ray"]
    next_best = max(score for label, score in scores.items() if label != "Chest X-ray")
    is_valid = chest_score >= 55 and chest_score > next_best
    return is_valid, dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _labels_match(disease: str, label_str: str) -> bool:
    label_lower = label_str.lower()
    if disease.lower() in label_lower:
        return True
    return any(syn.lower() in label_lower for syn in SYNONYMS.get(disease, []))


def _crosscheck(similar_cases, disease_probs: dict[str, float]) -> list[dict]:
    top_diseases = list(disease_probs.keys())[:5]
    diagnosis = []
    total_cases = max(len(similar_cases), 1)

    for disease in top_diseases:
        llm_prob = disease_probs[disease]
        matching_cases = sum(1 for case in similar_cases if _labels_match(disease, case.labels))
        gallery_support = matching_cases / total_cases
        confidence = (llm_prob / 100 * 0.5) + (gallery_support * 0.5)
        if gallery_support >= 0.6 and llm_prob >= 20:
            status = "HIGH"
        elif gallery_support >= 0.3 or llm_prob >= 15:
            status = "MEDIUM"
        else:
            status = "LOW"
        diagnosis.append({
            "disease": disease,
            "llm_probability": llm_prob,
            "matching_cases": matching_cases,
            "total_cases": total_cases,
            "gallery_support": f"{matching_cases}/{total_cases} cases",
            "confidence": round(confidence * 100, 1),
            "status": status,
        })
    return sorted(diagnosis, key=lambda item: item["confidence"], reverse=True)


def _positive_labels(label_str: str) -> list[str]:
    positives = []
    for part in label_str.split(" | "):
        if ": Positive" in part:
            positives.append(part.split(":")[0])
    return positives


def _generate_assessment(diagnosis: list[dict], similar_cases) -> str:
    primary = diagnosis[0]
    top_positive_labels = Counter()
    for case in similar_cases:
        top_positive_labels.update(_positive_labels(case.labels))

    supporting_findings = ", ".join(label for label, _ in top_positive_labels.most_common(3)) or "no repeated positive findings"
    differential = ", ".join(item["disease"] for item in diagnosis[1:4])

    return f"""
## Primary Clinical Impression

Based on visual similarity retrieval and zero-shot disease classification, the leading impression is **{primary["disease"]}** with a combined confidence of **{primary["confidence"]}%**.

## Evidence Summary

- The classifier estimated **{primary["llm_probability"]}%** probability for {primary["disease"]}.
- The retrieval engine found **{primary["gallery_support"]}** similar cases supporting this diagnosis.
- The most repeated positive findings among retrieved cases were: **{supporting_findings}**.

## Differential Diagnosis

Alternative conditions to consider are **{differential}**. These remain relevant because visually similar cases include overlapping thoracic findings common across chest X-ray pathology.

## Clinical Note

This is a retrieval-supported decision aid, not a definitive medical diagnosis. Final interpretation should be confirmed by a radiologist or clinician.
""".strip()


def _run_analysis(image: Image.Image, top_k: int):
    engine = _load_engine()
    similar_cases = engine.search(image, top_k=top_k, load_images=False)
    disease_probs = _predict_diseases(image)
    diagnosis = _crosscheck(similar_cases, disease_probs)
    assessment = _generate_assessment(diagnosis, similar_cases)
    return similar_cases, disease_probs, diagnosis, assessment


def _render_similar_cases(similar_cases):
    st.markdown("### Similar Historical Cases")
    for idx, case in enumerate(similar_cases, start=1):
        cols = st.columns([1, 3])
        with cols[0]:
            if case.filepath and Path(case.filepath).exists():
                try:
                    st.image(Image.open(case.filepath).convert("RGB"), use_container_width=True)
                except Exception:
                    st.caption("Preview unavailable")
        with cols[1]:
            st.markdown(f"**#{idx} {case.filename}**")
            st.write(f"Similarity: {case.similarity:.3f}")
            positives = _positive_labels(case.labels)
            st.write(f"Confirmed findings: {', '.join(positives) if positives else 'None'}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Upload a chest X-ray. The system retrieves similar historical cases and generates a retrieval-supported differential diagnosis."
    )

    with st.sidebar:
        st.markdown("**Index Status**")
        try:
            index_dir = _ensure_index_available()
            st.write(f"Index dir: `{index_dir}`")
            data_dir = index_dir.parent
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

        top_k = st.slider("Retrieved Cases", min_value=3, max_value=20, value=5, step=1)
        if st.button("Use Sample Image"):
            st.session_state["sample_path"] = str(_pick_sample_image(data_dir) or "")
        if st.button("Clear"):
            st.session_state.pop("sample_path", None)
            st.session_state.pop("analysis_ready", None)
            st.rerun()
        st.caption("First analysis can still be slow on Render free tier.")

    uploaded = st.file_uploader("Upload Patient Chest X-Ray", type=["png", "jpg", "jpeg"])
    sample_path = st.session_state.get("sample_path")

    query_image = None
    if uploaded is not None:
        query_image = Image.open(uploaded).convert("RGB")
        st.session_state["analysis_ready"] = True
    elif sample_path:
        query_image = Image.open(sample_path).convert("RGB")
        st.session_state["analysis_ready"] = True

    left, right = st.columns([1.05, 1.25])

    with left:
        st.markdown("### Input X-Ray")
        if query_image is not None:
            st.image(query_image, use_container_width=True)
        else:
            st.info("Upload an image or use the sample button.")

    with right:
        st.markdown("### Generated Clinical Assessment")
        if query_image is None:
            st.info("Run an analysis to generate the assessment.")
            return

        if st.button("Submit", type="primary") or st.session_state.get("analysis_ready"):
            with st.spinner("Running retrieval, classification, and crosscheck..."):
                is_valid_xray, input_scores = _validate_input_image(query_image)
                if not is_valid_xray:
                    st.error("This tool only supports chest X-ray images. Please upload a chest radiograph.")
                    st.markdown("### Input Validation")
                    for label, score in list(input_scores.items())[:3]:
                        st.write(f"{label}: {score}%")
                    st.session_state["analysis_ready"] = False
                    return
                similar_cases, disease_probs, diagnosis, assessment = _run_analysis(query_image, top_k)

            st.markdown(assessment)
            st.markdown("### Ranked Diagnoses")
            for item in diagnosis:
                st.write(
                    f"**{item['disease']}** | classifier {item['llm_probability']}% | "
                    f"gallery {item['gallery_support']} | confidence {item['confidence']}% [{item['status']}]"
                )
            st.markdown("### Top Disease Probabilities")
            for disease, prob in list(disease_probs.items())[:5]:
                st.write(f"{disease}: {prob}%")
            _render_similar_cases(similar_cases)
            st.session_state["analysis_ready"] = False


if __name__ == "__main__":
    main()
