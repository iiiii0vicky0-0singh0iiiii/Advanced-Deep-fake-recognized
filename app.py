# app.py
import os
import random
import tempfile
from pathlib import Path

import av
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MAX_VIDEO_FRAMES = 8

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

st.set_page_config(
    page_title="Advanced Deepfake Recognition",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }

        .subtitle {
            color: #9ca3af;
            font-size: 1.08rem;
            margin-bottom: 1.5rem;
        }

        .feature-card {
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 18px;
            height: 100%;
            background: #111827;
        }

        .feature-title {
            color: #60a5fa;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .model-badge {
            display: inline-block;
            background: #1e3a8a;
            color: #dbeafe;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .warning-box {
            border-left: 5px solid #f59e0b;
            background: #422006;
            padding: 14px;
            border-radius: 8px;
        }

        .result-fake {
            border: 1px solid #ef4444;
            background: rgba(127, 29, 29, 0.35);
            padding: 20px;
            border-radius: 12px;
        }

        .result-real {
            border: 1px solid #22c55e;
            background: rgba(20, 83, 45, 0.35);
            padding: 20px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the Vision Transformer model...")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def normalize_label(label: str) -> str:
    text = label.lower().replace("_", " ").replace("-", " ")

    if any(
        word in text
        for word in ("deepfake", "fake", "manipulated", "synthetic")
    ):
        return "FAKE"

    if any(
        word in text
        for word in ("realism", "real", "authentic", "original", "human")
    ):
        return "REAL"

    return label.upper()


@torch.inference_mode()
def analyze_image(image: Image.Image) -> dict:
    processor, model, device = load_model()

    rgb_image = image.convert("RGB")

    inputs = processor(
        images=rgb_image,
        return_tensors="pt",
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    output = model(**inputs)
    probabilities = torch.softmax(output.logits[0], dim=-1)
    probabilities = probabilities.detach().cpu().numpy()

    scores = {}

    for index, probability in enumerate(probabilities):
        raw_label = model.config.id2label.get(
            index,
            f"LABEL_{index}",
        )

        normalized_label = normalize_label(raw_label)

        scores[normalized_label] = (
            scores.get(normalized_label, 0.0)
            + float(probability)
        )

    predicted_label = max(scores, key=scores.get)

    return {
        "label": predicted_label,
        "confidence": scores[predicted_label],
        "scores": scores,
    }


def sample_video_frames(
    video_path: str,
    frame_count: int = MAX_VIDEO_FRAMES,
) -> list[Image.Image]:
    container = av.open(video_path)

    try:
        if not container.streams.video:
            raise ValueError("The uploaded file does not contain a video stream.")

        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        sampled_frames = []

        if total_frames > 0:
            target_indices = set(
                np.linspace(
                    0,
                    total_frames - 1,
                    min(frame_count, total_frames),
                    dtype=int,
                ).tolist()
            )

            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in target_indices:
                    sampled_frames.append(
                        frame.to_image().convert("RGB")
                    )

                if len(sampled_frames) == len(target_indices):
                    break
        else:
            # Reservoir sampling avoids keeping the complete video in memory.
            reservoir = []
            generator = random.Random(42)

            for frame_index, frame in enumerate(container.decode(stream)):
                image = frame.to_image().convert("RGB")

                if len(reservoir) < frame_count:
                    reservoir.append((frame_index, image))
                else:
                    replacement_index = generator.randint(0, frame_index)

                    if replacement_index < frame_count:
                        reservoir[replacement_index] = (
                            frame_index,
                            image,
                        )

            reservoir.sort(key=lambda item: item[0])
            sampled_frames = [item[1] for item in reservoir]

        if not sampled_frames:
            raise ValueError("No readable frames were found in the video.")

        return sampled_frames

    finally:
        container.close()


def analyze_video(video_path: str) -> dict:
    frames = sample_video_frames(video_path)
    frame_results = [analyze_image(frame) for frame in frames]

    fake_probabilities = [
        result["scores"].get("FAKE", 0.0)
        for result in frame_results
    ]

    real_probabilities = [
        result["scores"].get("REAL", 0.0)
        for result in frame_results
    ]

    mean_fake = float(np.mean(fake_probabilities))
    mean_real = float(np.mean(real_probabilities))
    peak_fake = float(np.max(fake_probabilities))

    fake_frame_count = sum(
        result["label"] == "FAKE"
        for result in frame_results
    )

    if mean_fake >= mean_real:
        final_label = "FAKE"
        confidence = mean_fake
    else:
        final_label = "REAL"
        confidence = mean_real

    return {
        "label": final_label,
        "confidence": confidence,
        "fake_probability": mean_fake,
        "real_probability": mean_real,
        "peak_fake_probability": peak_fake,
        "fake_frame_count": fake_frame_count,
        "frames": frames,
        "frame_results": frame_results,
    }


def probability_bar(label: str, probability: float):
    probability = min(max(float(probability), 0.0), 1.0)
    st.progress(
        probability,
        text=f"{label}: {probability:.2%}",
    )


def show_image_result(result: dict):
    label = result["label"]
    confidence = result["confidence"]

    if label == "FAKE":
        st.markdown(
            f"""
            <div class="result-fake">
                <h3>⚠️ Likely Deepfake or Manipulated</h3>
                <p>Model confidence: <strong>{confidence:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif label == "REAL":
        st.markdown(
            f"""
            <div class="result-real">
                <h3>✅ Likely Authentic</h3>
                <p>Model confidence: <strong>{confidence:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(f"Prediction: {label} — {confidence:.2%}")

    st.subheader("Prediction probabilities")

    fake_probability = result["scores"].get("FAKE", 0.0)
    real_probability = result["scores"].get("REAL", 0.0)

    probability_bar("Deepfake", fake_probability)
    probability_bar("Authentic", real_probability)

    with st.expander("How was this result calculated?"):
        st.markdown(
            """
            1. The image was converted to RGB.
            2. It was resized and normalized using the model's processor.
            3. The image was divided into **16 × 16 pixel patches**.
            4. A Vision Transformer analyzed relationships between those patches.
            5. The model produced two logits: **Deepfake** and **Realism**.
            6. Softmax converted those logits into probabilities.
            """
        )


def show_model_information():
    st.markdown("## Why this model is different")

    column1, column2, column3 = st.columns(3)

    with column1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Vision Transformer</div>
                Instead of relying only on small local filters, the model treats
                an image as a sequence of patches and uses self-attention to
                compare spatial relationships across the image.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with column2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Task-specific training</div>
                This is a binary image classifier fine-tuned specifically on
                authentic and deepfake images. It is not a general chatbot or a
                generic object-recognition model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with column3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-title">Open and reproducible</div>
                The model architecture, weights, preprocessing configuration,
                class labels and published evaluation results are available on
                Hugging Face for inspection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Model used in this project")

    st.markdown(
        f"""
        <span class="model-badge">{MODEL_ID}</span>

        - **Architecture:** ViT Base, patch size 16, input size 224 × 224
        - **Base checkpoint:** `google/vit-base-patch16-224-in21k`
        - **Task:** Binary image classification
        - **Classes:** Deepfake and Realism
        - **Reported test examples:** 56,001
        - **Reported accuracy:** 92.12%
        - **Reported deepfake recall:** 97.15%
        - **Reported realism recall:** 87.08%
        - **License:** Apache 2.0
        - **Model card:** [{MODEL_ID}]({MODEL_URL})

        The metrics above are reported by the model author on the model's test
        dataset. They are not an independent evaluation and do not guarantee
        equal performance on every camera, compression level or generator.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Comparison with other approaches")

    comparison_data = [
        {
            "Approach": "This project: ViT deepfake classifier",
            "Main evidence": "Relationships between 16 × 16 image patches",
            "Best use": "Screening images for deepfake patterns",
            "Main limitation": "Image-only binary prediction",
        },
        {
            "Approach": "Traditional CNN detector",
            "Main evidence": "Local textures and convolutional features",
            "Best use": "Fast detection of known local artifacts",
            "Main limitation": "May overfit to compression or texture clues",
        },
        {
            "Approach": "Generic AI-image detector",
            "Main evidence": "Broad synthetic-image characteristics",
            "Best use": "Detecting many types of generated artwork",
            "Main limitation": "Not necessarily specialized for deepfakes",
        },
        {
            "Approach": "Metadata or C2PA verification",
            "Main evidence": "File history and cryptographic provenance",
            "Best use": "Verifying trusted content origin",
            "Main limitation": "Missing metadata does not prove manipulation",
        },
        {
            "Approach": "Cloud detection API",
            "Main evidence": "Provider-specific private models",
            "Best use": "Large production workflows",
            "Main limitation": "Cost, privacy and limited model transparency",
        },
    ]

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("## Why use it for this project?")

    st.markdown(
        """
        1. **It matches the project task.** The required output is a direct
           authentic-versus-deepfake classification.

        2. **It uses global image context.** Self-attention can compare distant
           image patches instead of examining only small local regions.

        3. **It is explainable at the system level.** The application shows both
           class probabilities, sampled video frames and per-frame results.

        4. **It is reproducible.** The same public model and preprocessing
           configuration can be tested by other researchers.

        5. **It can be extended.** The classifier can become one component in a
           larger system containing metadata checks, face analysis, frequency
           analysis and temporal video models.
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Important:</strong> This model should be used as a screening
            tool, not as the only evidence for legal, journalistic, academic or
            disciplinary decisions. No deepfake detector is universally reliable.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_method_and_limitations():
    st.markdown("## Detection methodology")

    st.markdown(
        """
        ### Image analysis

        The uploaded image is resized to the model's expected 224 × 224 input.
        It is divided into 16 × 16 patches and passed through a Vision
        Transformer. The classifier returns probabilities for Deepfake and
        Realism.

        ### Video analysis

        This model is an image classifier, not a native temporal video model.
        The application therefore samples up to eight frames, analyzes every
        frame independently and averages their probabilities.

        The application also displays the highest fake probability and the
        number of sampled frames classified as fake.
        """
    )

    st.markdown("## Known limitations")

    limitations = [
        "It does not analyze audio.",
        "It does not directly model motion between video frames.",
        "It can perform differently on unseen deepfake generators.",
        "Heavy compression, resizing and screenshots can alter useful evidence.",
        "A high confidence score is not proof of manipulation.",
        "An authentic classification does not prove that media is original.",
        "Performance can vary across demographics and capture conditions.",
        "The reported model-card metrics are not an independent audit.",
    ]

    for limitation in limitations:
        st.markdown(f"- {limitation}")

    st.markdown("## Recommended complete forensic workflow")

    st.markdown(
        """
        1. Run this model as the first visual screening stage.
        2. Inspect several frames instead of trusting one frame.
        3. Check file metadata and content provenance.
        4. Compare the media with its original source.
        5. Analyze audio separately.
        6. Use a temporal video model when motion evidence matters.
        7. Require human review for important decisions.
        """
    )


st.markdown(
    '<div class="main-title">Advanced Deepfake Recognition</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        A transparent Vision Transformer screening system for images and videos.
    </div>
    """,
    unsafe_allow_html=True,
)

analyze_tab, model_tab, method_tab = st.tabs(
    [
        "🔍 Analyze Media",
        "🧠 Why This Model?",
        "📚 Method & Limitations",
    ]
)

with analyze_tab:
    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload media to begin. The model loads during the first analysis."
        )

    else:
        extension = Path(uploaded_file.name).suffix.lower()
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

        if extension in image_extensions:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            if st.button(
                "Analyze image",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Running Vision Transformer inference..."
                    ):
                        image_result = analyze_image(image)

                    show_image_result(image_result)

                except Exception as error:
                    st.error("The image could not be analyzed.")
                    st.exception(error)

        else:
            video_bytes = uploaded_file.getvalue()
            st.video(video_bytes)

            if st.button(
                "Analyze video frames",
                type="primary",
                use_container_width=True,
            ):
                temporary_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=extension,
                    ) as temporary_file:
                        temporary_file.write(video_bytes)
                        temporary_path = temporary_file.name

                    with st.spinner(
                        "Sampling frames and running model inference..."
                    ):
                        video_result = analyze_video(temporary_path)

                    show_image_result(
                        {
                            "label": video_result["label"],
                            "confidence": video_result["confidence"],
                            "scores": {
                                "FAKE": video_result["fake_probability"],
                                "REAL": video_result["real_probability"],
                            },
                        }
                    )

                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Frames analyzed",
                        len(video_result["frames"]),
                    )

                    metric2.metric(
                        "Frames classified fake",
                        video_result["fake_frame_count"],
                    )

                    metric3.metric(
                        "Highest fake probability",
                        f"{video_result['peak_fake_probability']:.2%}",
                    )

                    st.subheader("Frame-level evidence")

                    frame_columns = st.columns(4)

                    for index, (frame, frame_result) in enumerate(
                        zip(
                            video_result["frames"],
                            video_result["frame_results"],
                        )
                    ):
                        with frame_columns[index % 4]:
                            st.image(
                                frame,
                                use_container_width=True,
                            )

                            if frame_result["label"] == "FAKE":
                                st.error(
                                    f"Frame {index + 1}: FAKE\n\n"
                                    f"{frame_result['confidence']:.2%}"
                                )
                            else:
                                st.success(
                                    f"Frame {index + 1}: REAL\n\n"
                                    f"{frame_result['confidence']:.2%}"
                                )

                except Exception as error:
                    st.error("The video could not be analyzed.")
                    st.exception(error)

                finally:
                    if temporary_path and os.path.exists(temporary_path):
                        os.remove(temporary_path)

with model_tab:
    show_model_information()

with method_tab:
    show_method_and_limitations()
