"""Streamlit app for satellite + social-media disaster situation reports."""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
for path in [SRC_DIR, SRC_DIR / "week11", SRC_DIR / "week12", SRC_DIR / "week15", SRC_DIR / "week16", SRC_DIR / "week17"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from week11_dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from week12_model_backbones import ArcMarginProduct, ObjectDamageRepresentationModel
from week15_fuse_event import build_event
from week16_build_event_documents import build_document
from week17_generate_situation_report import template_report


PREFERRED_CHECKPOINT = (
    ROOT
    / "results"
    / "week12"
    / "week12_convnext_tiny_gated_effective_no_sampler"
    / "checkpoints"
    / "week12_convnext_tiny_gated_ce_best.pt"
)

HUMANITARIAN_KEYWORDS = {
    "infrastructure_damage": ["building", "bridge", "road", "power", "utility", "collapse", "collapsed", "damage", "damaged", "crumble"],
    "affected_people": ["people", "families", "residents", "evacuated", "shelter", "homeless", "suffering"],
    "rescue_efforts": ["rescue", "search", "volunteer", "donation", "aid", "relief", "support", "help"],
    "injured_dead": ["dead", "death", "killed", "injured", "casualties", "fatalities"],
    "missing_or_found_people": ["missing", "found", "trapped"],
    "vehicle_damage": ["vehicle", "car", "truck", "bus"],
}

EMOTION_KEYWORDS = {
    "fear": ["terrifying", "scared", "fear", "panic", "danger", "threat"],
    "sadness": ["sad", "dead", "killed", "praying", "suffering", "tragedy", "heartbreaking"],
    "anger": ["angry", "outrage", "furious", "blame", "failed"],
    "hope": ["hope", "rescue", "help", "support", "relief", "safe", "praying"],
}


def decode_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {uploaded_file.name}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray, size: int = 128) -> torch.Tensor:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@st.cache_resource(show_spinner=False)
def load_week12_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("backbone", "convnext_tiny")
    fusion = checkpoint.get("fusion", "gated")
    embedding_dim = int(checkpoint.get("embedding_dim", 256))
    model = ObjectDamageRepresentationModel(backbone=backbone, fusion=fusion, embedding_dim=embedding_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arc_head = None
    if checkpoint.get("arc_head_state_dict") is not None:
        arc_head = ArcMarginProduct(
            embedding_dim,
            len(CLASS_NAMES),
            scale=float(checkpoint.get("arcface_scale", 30.0)),
            margin=float(checkpoint.get("arcface_margin", 0.3)),
        )
        arc_head.load_state_dict(checkpoint["arc_head_state_dict"])
        arc_head.eval()
    return model, arc_head


@torch.no_grad()
def predict_crop_pairs(pre_files, post_files, checkpoint_path: Path) -> dict[str, Any]:
    if len(pre_files) != len(post_files):
        raise ValueError("Upload the same number of pre-disaster and post-disaster building crops.")
    if not pre_files:
        raise ValueError("Upload at least one pre/post building crop pair.")

    model, arc_head = load_week12_model(str(checkpoint_path))
    counts: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    total_confidence = 0.0
    rows = []

    for pre_file, post_file in zip(pre_files, post_files):
        pre_image = decode_image(pre_file)
        post_image = decode_image(post_file)
        diff_image = np.abs(pre_image.astype(np.float32) - post_image.astype(np.float32)).astype(np.uint8)

        pre = preprocess_image(pre_image).unsqueeze(0)
        post = preprocess_image(post_image).unsqueeze(0)
        diff = preprocess_image(diff_image).unsqueeze(0)
        logits, embeddings = model(pre, post, diff, return_embedding=True)
        if arc_head is not None:
            logits = arc_head.cosine_logits(embeddings)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        prediction = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[prediction].item())
        class_name = CLASS_NAMES[prediction]
        counts[class_name] += 1
        confidence_sum[class_name] += confidence
        total_confidence += confidence
        rows.append({"pre": pre_file.name, "post": post_file.name, "prediction": class_name, "confidence": confidence})

    total = len(rows)
    return {
        "source": "streamlit_week12_preferred_convnext_tiny_gated",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "total_buildings": total,
        "no_damage": counts["no_damage"],
        "minor": counts["minor_damage"],
        "major": counts["major_damage"],
        "destroyed": counts["destroyed"],
        "total_damaged": counts["minor_damage"] + counts["major_damage"] + counts["destroyed"],
        "confidence": total_confidence / max(total, 1),
        "class_confidence": {
            class_name: confidence_sum[class_name] / max(counts[class_name], 1)
            for class_name in CLASS_NAMES
            if counts[class_name] > 0
        },
        "predictions": rows,
    }


def parse_tweets(text: str, uploaded_file) -> list[str]:
    tweets = [line.strip() for line in text.splitlines() if line.strip()]
    if uploaded_file is not None:
        raw = uploaded_file.getvalue().decode("utf-8", errors="replace")
        if uploaded_file.name.lower().endswith(".csv"):
            reader = csv.DictReader(io.StringIO(raw))
            for row in reader:
                value = row.get("tweet_text") or row.get("text") or row.get("tweet") or ""
                if value.strip():
                    tweets.append(value.strip())
        else:
            tweets.extend(line.strip() for line in raw.splitlines() if line.strip())
    return list(dict.fromkeys(tweets))


def count_keyword_matches(tweet: str, keyword_map: dict[str, list[str]]) -> str | None:
    lowered = tweet.lower()
    scores = {
        label: sum(1 for keyword in keywords if keyword in lowered)
        for label, keywords in keyword_map.items()
    }
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score > 0 else None


def aggregate_tweets(tweets: list[str]) -> dict[str, Any]:
    humanitarian = Counter()
    emotion = Counter()
    informative = 0
    for tweet in tweets:
        human_label = count_keyword_matches(tweet, HUMANITARIAN_KEYWORDS)
        emotion_label = count_keyword_matches(tweet, EMOTION_KEYWORDS)
        if human_label is not None:
            humanitarian[human_label] += 1
            informative += 1
        else:
            humanitarian["other_relevant_information"] += 1
            informative += 1
        emotion[emotion_label or "neutral"] += 1
    return {
        "source": "streamlit_keyword_social_aggregation",
        "informative_posts": informative,
        "humanitarian": dict(humanitarian),
        "emotion": dict(emotion),
        "representative_posts": tweets[:5],
    }


def save_outputs(event: dict[str, Any], document: dict[str, Any], report: str) -> tuple[Path, Path, Path]:
    event_dir = ROOT / "results" / "streamlit_app"
    event_dir.mkdir(parents=True, exist_ok=True)
    event_json = event_dir / f"{event['event']}.json"
    doc_json = event_dir / f"{event['event']}_document.json"
    report_md = event_dir / f"{event['event']}_report.md"
    event_json.write_text(json.dumps(event, indent=2), encoding="utf-8")
    doc_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
    report_md.write_text(report + "\n", encoding="utf-8")
    return event_json, doc_json, report_md


st.set_page_config(page_title="Disaster Situation Report Generator", layout="wide")
st.title("Disaster Situation Report Generator")

with st.sidebar:
    st.subheader("Model Status")
    checkpoint_path = st.text_input("Week 12 vision checkpoint", value=str(PREFERRED_CHECKPOINT.relative_to(ROOT)))
    checkpoint = ROOT / checkpoint_path
    if checkpoint.exists():
        st.success("Preferred Week 12 gated ConvNeXt checkpoint found.")
    else:
        st.warning("Preferred vision checkpoint not found. Use manual satellite counts.")
    st.caption("Final fusion uses satellite damage assessment + social-media context.")

event_id = st.text_input("Event ID", value="validation_demo")

left, right = st.columns(2)

with left:
    st.header("Satellite Damage")
    mode = st.radio("Input mode", ["Manual counts", "Upload building crop pairs", "Upload satellite JSON"], horizontal=False)
    satellite_payload: dict[str, Any] | None = None

    if mode == "Manual counts":
        destroyed = st.number_input("Destroyed buildings", min_value=0, value=1496, step=1)
        major = st.number_input("Major damage buildings", min_value=0, value=77, step=1)
        minor = st.number_input("Minor damage buildings", min_value=0, value=60, step=1)
        confidence = st.slider("Vision confidence", 0.0, 1.0, 0.91)
        satellite_payload = {"destroyed": destroyed, "major": major, "minor": minor, "confidence": confidence}
    elif mode == "Upload satellite JSON":
        uploaded_json = st.file_uploader("Satellite JSON", type=["json"], key="satellite_json")
        if uploaded_json is not None:
            satellite_payload = json.loads(uploaded_json.getvalue().decode("utf-8"))
    else:
        st.caption("Upload aligned object-level building crops. Full-scene xBD images require a building extraction step before this classifier.")
        pre_files = st.file_uploader("Pre-disaster building crops", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="pre")
        post_files = st.file_uploader("Post-disaster building crops", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="post")
        if st.button("Run Week 12 Vision Model", disabled=not checkpoint.exists()):
            try:
                satellite_payload = predict_crop_pairs(pre_files, post_files, checkpoint)
                st.session_state["satellite_payload"] = satellite_payload
            except Exception as exc:
                st.error(str(exc))
        satellite_payload = st.session_state.get("satellite_payload")

    if satellite_payload:
        st.json(satellite_payload)

with right:
    st.header("Social Media")
    social_mode = st.radio("Social input mode", ["Paste/upload tweets", "Upload social JSON"], horizontal=False)
    social_payload: dict[str, Any] | None = None

    if social_mode == "Upload social JSON":
        uploaded_social = st.file_uploader("Social JSON", type=["json"], key="social_json")
        if uploaded_social is not None:
            social_payload = json.loads(uploaded_social.getvalue().decode("utf-8"))
    else:
        tweet_text = st.text_area("Tweets, one per line", height=220)
        tweet_file = st.file_uploader("Optional TXT or CSV with tweet_text column", type=["txt", "csv"], key="tweets")
        tweets = parse_tweets(tweet_text, tweet_file)
        st.caption(f"{len(tweets)} tweets loaded.")
        social_payload = aggregate_tweets(tweets) if tweets else None

    if social_payload:
        st.json(social_payload)

st.header("Situation Report")
generate = st.button("Generate Report", type="primary", disabled=not (event_id and satellite_payload and social_payload))
if generate and satellite_payload and social_payload:
    event = build_event(
        event_id,
        satellite_payload,
        social_payload,
        {"satellite_source": mode, "social_source": social_mode},
    )
    document = build_document(event)
    report = template_report(event)
    event_json, doc_json, report_md = save_outputs(event, document, report)

    tab_report, tab_event, tab_document = st.tabs(["Report", "Fusion JSON", "RAG Document"])
    with tab_report:
        st.markdown(report)
        st.download_button("Download report", report, file_name=f"{event_id}_report.md")
    with tab_event:
        st.json(event)
        st.caption(f"Saved to {event_json.relative_to(ROOT)}")
    with tab_document:
        st.json(document)
        st.caption(f"Saved to {doc_json.relative_to(ROOT)}")
    st.success(f"Saved report to {report_md.relative_to(ROOT)}")
