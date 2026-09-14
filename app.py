# app.py
import os
import tempfile
from pathlib import Path

import av
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
MAX_VIDEO_FRAMES = 8

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading deepfake detection model...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return processor, model, device


def normalize_label(label: str) -> str:
    text = label.lower().replace("_", " ").replace("-", " ")

    if any(word in text for word in ("fake", "deepfake", "manipulated", "synthetic")):
        return "FAKE"

    if any(word in text for word in ("real", "authentic", "original", "human")):
        return "REAL"

    return label.upper()


@torch.inference_mode()
def analyze_image(image: Image.Image) -> dict:
    processor, model, device = load_model()

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    logits = model(**inputs).logits[0]
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

    scores = {}
    for index, probability in enumerate(probabilities):
        raw_label = model.config.id2label.get(index, f"LABEL_{index}")
        label = normalize_label(raw_label)
        scores[label] = scores.get(label, 0.0) + float(probability)

    predicted_label = max(scores, key=scores.get)
    confidence = scores[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "scores": scores,
    }


def sample_video_frames(video_path: str, frame_count: int) -> list[Image.Image]:
    container = av.open(video_path)
    stream = container.streams.video[0]

    total_frames = stream.frames
    duration = float(stream.duration * stream.time_base) if stream.duration else None

    frames = []

    if total_frames and total_frames > 0:
        target_indices = set(
            np.linspace(
                0,
                max(total_frames - 1, 0),
                min(frame_count, total_frames),
                dtype=int,
            ).tolist()
        )

        for index, frame in enumerate(container.decode(stream)):
            if index in target_indices:
                frames.append(frame.to_image().convert("RGB"))

            if len(frames) >= len(target_indices):
                break
    else:
        decoded = list(container.decode(stream))

        if decoded:
            indices = np.linspace(
                0,
                len(decoded) - 1,
                min(frame_count, len(decoded)),
                dtype=int,
            )

            frames = [
                decoded[index].to_image().convert("RGB")
                for index in indices
            ]

    container.close()

    if not frames:
        raise ValueError("No readable video frames were found.")

    return frames


def analyze_video(video_path: str) -> dict:
    frames = sample_video_frames(video_path, MAX_VIDEO_FRAMES)
    frame_results = [analyze_image(frame) for frame in frames]

    fake_scores = [
        result["scores"].get("FAKE", 0.0)
        for result in frame_results
    ]
    real_scores = [
        result["scores"].get("REAL", 0.0)
        for result in frame_results
    ]

    mean_fake = float(np.mean(fake_scores))
    mean_real = float(np.mean(real_scores))

    if mean_fake >= mean_real:
        label = "FAKE"
        confidence = mean_fake
    else:
        label = "REAL"
        confidence = mean_real

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": mean_fake,
        "real_probability": mean_real,
        "frames": frames,
        "frame_results": frame_results,
    }


def show_result(label: str, confidence: float):
    if label == "FAKE":
        st.error(f"Likely manipulated/deepfake — {confidence:.2%} confidence")
    elif label == "REAL":
        st.success(f"Likely authentic — {confidence:.2%} confidence")
    else:
        st.info(f"{label} — {confidence:.2%} confidence")


st.title("🔍 Deepfake Detection")
st.caption(
    "Upload an image or video. Results are model estimates and are not forensic proof."
)

uploaded_file = st.file_uploader(
    "Choose an image or video",
    type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded_file:
    extension = Path(uploaded_file.name).suffix.lower()
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if extension in image_extensions:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        if st.button("Analyze image", type="primary"):
            try:
                with st.spinner("Analyzing image..."):
                    result = analyze_image(image)

                show_result(result["label"], result["confidence"])

                st.subheader("Model probabilities")
                st.bar_chart(
                    {
                        label: probability
                        for label, probability in result["scores"].items()
                    }
                )
            except Exception as error:
                st.exception(error)

    else:
        video_bytes = uploaded_file.getvalue()
        st.video(video_bytes)

        if st.button("Analyze video", type="primary"):
            temporary_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension,
                ) as temporary_file:
                    temporary_file.write(video_bytes)
                    temporary_path = temporary_file.name

                with st.spinner("Sampling and analyzing video frames..."):
                    result = analyze_video(temporary_path)

                show_result(result["label"], result["confidence"])

                st.subheader("Aggregate probabilities")
                st.bar_chart(
                    {
                        "FAKE": result["fake_probability"],
                        "REAL": result["real_probability"],
                    }
                )

                st.subheader("Sampled frames")

                columns = st.columns(4)
                for index, (frame, frame_result) in enumerate(
                    zip(result["frames"], result["frame_results"])
                ):
                    with columns[index % 4]:
                        st.image(
                            frame,
                            caption=(
                                f"Frame {index + 1}: "
                                f"{frame_result['label']} "
                                f"({frame_result['confidence']:.1%})"
                            ),
                            use_container_width=True,
                        )
            except Exception as error:
                st.exception(error)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)