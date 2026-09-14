# app.py
import hashlib
import io
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import av
import numpy as np
import streamlit as st
import torch
import transformers
from PIL import Image, ImageEnhance, ImageOps
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
)


MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
APP_VERSION = "2.0.0"
MAX_UPLOAD_MB = 100
MAX_IMAGE_PIXELS = 20_000_000
MAX_UNKNOWN_DURATION_FRAMES = 2000

st.set_page_config(
    page_title="Verity | Deepfake Evidence Studio",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1018;
        color: #e8eef7;
    }
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.035em;
        color: #f0f5ff !important;
    }
    p, li {
        line-height: 1.65;
    }
    .eyebrow {
        color: #5eead4;
        letter-spacing: 0.22em;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .hero {
        padding: 25px 0 22px;
    }
    .hero h1 {
        font-size: clamp(2.3rem, 5vw, 4.4rem);
        font-weight: 800;
        line-height: 1.04;
        margin: 10px 0 18px;
    }
    .hero h1 span {
        color: #5eead4;
    }
    .hero p {
        color: #a6b5cb;
        max-width: 730px;
        font-size: 1.08rem;
    }
    .card {
        background: #111b2b;
        border: 1px solid #243247;
        border-radius: 16px;
        padding: 22px;
        min-height: 170px;
    }
    .card .number {
        font-size: 0.78rem;
        color: #5eead4;
        letter-spacing: 0.12em;
    }
    .card h3 {
        margin: 12px 0 8px;
        font-size: 1.2rem;
    }
    .card p {
        color: #acbad0;
        font-size: 0.93rem;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background: #111b2b;
        border: 1px solid #243247;
        padding: 18px;
        border-radius: 14px;
    }
    section[data-testid="stSidebar"] {
        background: #0f1724;
    }
    button[kind="primary"] {
        background: #5eead4;
        color: #06231e;
        border: none;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
    )

    # Use the same resolved model revision for weights and preprocessing.
    revision = getattr(config, "_commit_hash", None) or "main"

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=False,
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        revision=revision,
        config=config,
        use_safetensors=True,
        trust_remote_code=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mapped_labels = {}
    aliases = {
        "deepfake": "FAKE",
        "fake": "FAKE",
        "realism": "REAL",
        "real": "REAL",
        "authentic": "REAL",
    }

    for index in range(config.num_labels):
        raw = config.id2label.get(
            index,
            config.id2label.get(str(index), ""),
        )
        normalized = str(raw).lower().replace("_", "").replace("-", "")
        mapped_labels[index] = aliases.get(normalized)

    if (
        config.num_labels != 2
        or set(mapped_labels.values()) != {"FAKE", "REAL"}
    ):
        raise ValueError(
            "The checkpoint has unexpected class labels. "
            "Review its config instead of guessing which label means fake."
        )

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "labels": mapped_labels,
        "revision": revision,
        "lock": threading.Lock(),
    }


def fake_score(image, bundle):
    with bundle["lock"], torch.inference_mode():
        inputs = bundle["processor"](
            images=image.convert("RGB"),
            return_tensors="pt",
        )
        inputs = {
            name: value.to(bundle["device"])
            for name, value in inputs.items()
        }

        logits = bundle["model"](**inputs).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()

        fake_index = next(
            index
            for index, label in bundle["labels"].items()
            if label == "FAKE"
        )

        score = float(probabilities[fake_index])

    if not math.isfinite(score):
        raise ValueError("The model returned an invalid score.")

    return score


# ----------------------------------------------------------------------
# Media preparation
# ----------------------------------------------------------------------

def load_image(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "Image exceeds 20 megapixels. Resize it before uploading."
            )

        return ImageOps.exif_transpose(source).convert("RGB")


def thumbnail(image):
    preview = image.copy()
    preview.thumbnail((640, 640))
    output = io.BytesIO()
    preview.save(output, format="JPEG", quality=85)
    return output.getvalue()


def image_views(image, audit):
    yield "Original", image

    if not audit:
        return

    compressed = io.BytesIO()
    image.save(compressed, format="JPEG", quality=75)

    with Image.open(io.BytesIO(compressed.getvalue())) as jpeg:
        yield "JPEG quality 75", jpeg.convert("RGB")

    reduced_size = (
        max(1, round(image.width * 0.75)),
        max(1, round(image.height * 0.75)),
    )
    reduced = image.resize(
        reduced_size,
        Image.Resampling.LANCZOS,
    ).resize(
        image.size,
        Image.Resampling.LANCZOS,
    )

    yield "Resize 75% and restore", reduced
    yield "Brightness −10%", ImageEnhance.Brightness(image).enhance(0.9)


def extract_video_samples(path, count):
    samples = []
    warnings = []

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("No video stream was found.")

        stream = container.streams.video[0]
        origin = int(stream.start_time or 0)

        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = None

        if duration is not None and (
            not math.isfinite(duration) or duration <= 0
        ):
            duration = None

        seen = set()

        if duration is not None and stream.time_base is not None:
            targets = np.linspace(
                duration * 0.05,
                duration * 0.95,
                count,
            )

            for target in targets:
                chosen = None
                try:
                    offset = origin + int(float(target) / stream.time_base)

                    container.seek(
                        offset,
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    )

                    # Bound decoding after each seek for unusual files.
                    for index, frame in enumerate(container.decode(stream)):
                        if index >= 1000:
                            break

                        if frame.pts is None:
                            continue

                        seconds = float(
                            (frame.pts - origin) * stream.time_base
                        )

                        if seconds >= float(target):
                            chosen = (frame, seconds)
                            break

                except Exception:
                    chosen = None

                if chosen is None:
                    continue

                frame, seconds = chosen
                identity = int(frame.pts)

                if identity in seen:
                    continue

                seen.add(identity)
                samples.append({
                    "timestamp_seconds": round(max(0.0, seconds), 3),
                    "frame_index": None,
                    "image": frame.to_image().convert("RGB"),
                })

            sampling = "Frames near evenly spaced timestamps from 5% to 95%"

            if len(samples) < count:
                warnings.append(
                    f"Requested {count} distinct frames; decoded "
                    f"{len(samples)}. Sampling coverage is incomplete."
                )

        else:
            # Bounded reservoir sampling for files with missing duration.
            generator = np.random.default_rng(42)
            reservoir = []
            truncated = False

            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_UNKNOWN_DURATION_FRAMES:
                    truncated = True
                    break

                if len(reservoir) < count:
                    slot = len(reservoir)
                else:
                    slot = int(generator.integers(0, index + 1))
                    if slot >= count:
                        continue

                seconds = None
                if frame.pts is not None and stream.time_base is not None:
                    seconds = round(
                        max(
                            0.0,
                            float((frame.pts - origin) * stream.time_base),
                        ),
                        3,
                    )

                sample = {
                    "timestamp_seconds": seconds,
                    "frame_index": index,
                    "image": frame.to_image().convert("RGB"),
                }

                if slot == len(reservoir):
                    reservoir.append(sample)
                else:
                    reservoir[slot] = sample

            samples = sorted(reservoir, key=lambda item: item["frame_index"])
            sampling = "Deterministic reservoir sampling, seed 42"

            if truncated:
                warnings.append(
                    "Duration unavailable: sampled only the first "
                    f"{MAX_UNKNOWN_DURATION_FRAMES} decoded frames."
                )

    if not samples:
        raise ValueError("No usable video frames could be decoded.")

    return samples, duration, sampling, warnings


# ----------------------------------------------------------------------
# Review rules — application heuristics, not calibrated probabilities
# ----------------------------------------------------------------------

def make_decision(rows, kind, settings, warnings):
    scores = [row["fake_score"] for row in rows]
    main_score = scores[0] if kind == "image" else float(np.mean(scores))
    spread = max(scores) - min(scores)
    reasons = []

    if settings["context"] != "A human face is visible":
        reasons.append("The selected input context is outside the intended scope.")

    if warnings:
        reasons.extend(warnings)

    if kind == "image" and len(scores) > 1:
        if spread > settings["spread_limit"]:
            reasons.append("Small image changes caused a large score variation.")

        if min(scores) < 0.5 <= max(scores):
            reasons.append("The class prediction changed between image variants.")

    if kind == "video":
        if min(scores) <= settings["low"] and max(scores) >= settings["high"]:
            reasons.append("Sampled frames contain strongly conflicting scores.")

    if settings["low"] < main_score < settings["high"]:
        reasons.append("The score falls inside the selected review band.")

    if reasons:
        verdict = "INCONCLUSIVE — REVIEW NEEDED"
    elif main_score >= settings["high"]:
        verdict = "FAKE-LEANING"
    else:
        verdict = "REAL-LEANING"

    if not reasons:
        reasons.append(
            "The image score passed the selected review rules."
            if kind == "image"
            else "The mean sampled-frame score passed the selected review rules."
        )

    return {
        "verdict": verdict,
        "fake_score": main_score,
        "score_spread": spread,
        "reasons": reasons,
    }


def analyze(data, filename, kind, settings, status):
    started = time.perf_counter()

    status.info("Loading model… The first run downloads the weights.")
    bundle = load_model()
    rows, previews, warnings = [], [], []
    media_details = {}

    if kind == "image":
        image = load_image(data)
        media_details = {
            "width": image.width,
            "height": image.height,
            "sampling": "Original plus selected image transformations",
        }

        if min(image.size) < 128:
            warnings.append(
                "The image has a side smaller than 128 pixels; "
                "fine facial detail may be insufficient."
            )

        for label, view in image_views(image, settings["audit"]):
            status.info(f"Analyzing: {label}")
            rows.append({
                "sample": label,
                "fake_score": fake_score(view, bundle),
            })
            previews.append(thumbnail(view))

    else:
        suffix = Path(filename).suffix.lower()

        with tempfile.TemporaryDirectory() as folder:
            video_path = Path(folder) / f"input{suffix}"
            video_path.write_bytes(data)

            status.info("Extracting video samples…")
            samples, duration, sampling, warnings = extract_video_samples(
                str(video_path),
                settings["frames"],
            )

            media_details = {
                "duration_seconds": duration,
                "sampling": sampling,
                "requested_frames": settings["frames"],
                "analyzed_frames": len(samples),
            }

            for index, sample in enumerate(samples):
                status.info(f"Analyzing frame {index + 1}/{len(samples)}")
                rows.append({
                    "sample": f"Frame {index + 1}",
                    "timestamp_seconds": sample["timestamp_seconds"],
                    "frame_index": sample["frame_index"],
                    "fake_score": fake_score(sample["image"], bundle),
                })
                previews.append(thumbnail(sample["image"]))

    decision = make_decision(rows, kind, settings, warnings)

    report = {
        "application": "Verity Evidence Studio",
        "application_version": APP_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": {
            "name": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "kind": kind,
            **media_details,
        },
        "model": {
            "id": MODEL_ID,
            "revision": bundle["revision"],
            "device": str(bundle["device"]),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
        },
        "settings": settings,
        "result": decision,
        "samples": rows,
        "elapsed_seconds_including_model_loading": round(
            time.perf_counter() - started,
            2,
        ),
        "interpretation": [
            "Scores are softmax outputs, not calibrated authenticity probabilities.",
            "Review thresholds are application heuristics, not validated guarantees.",
            "Image variants use the same model and are not independent detectors.",
            "Video analysis covers sampled images, not audio or temporal motion.",
            "File hashes identify bytes; they do not authenticate media origin.",
        ],
    }

    return report, previews


# ----------------------------------------------------------------------
# Result presentation
# ----------------------------------------------------------------------

def show_results(report, previews):
    result = report["result"]
    rows = report["samples"]
    settings = report["settings"]
    kind = report["file"]["kind"]
    score = result["fake_score"]

    st.divider()
    st.subheader("Your evidence brief")

    if result["verdict"].startswith("INCONCLUSIVE"):
        st.warning(result["verdict"])
    elif result["verdict"] == "FAKE-LEANING":
        st.error("FAKE-LEANING — inspect the evidence before acting.")
    else:
        st.success("REAL-LEANING — origin and authenticity remain unverified.")

    a, b, c = st.columns(3)
    a.metric(
        "Original fake score" if kind == "image" else "Mean frame fake score",
        f"{score:.1%}",
    )
    b.metric(
        "Variant score range" if kind == "image" else "Frame score range",
        f"{result['score_spread'] * 100:.1f} pp",
    )
    c.metric(
        "Views checked" if kind == "image" else "Frames checked",
        len(rows),
    )

    st.progress(score, text=f"Fake-class score: {score:.1%}")
    st.caption(
        "The score is a model output. It is not the probability that "
        "the conclusion is correct."
    )

    with st.expander("Why did the app give this result?", expanded=True):
        for reason in result["reasons"]:
            st.write(f"• {reason}")

        st.caption(
            f"Review band: {settings['low']:.0%}–{settings['high']:.0%}. "
            "These are configurable review rules, not benchmarked thresholds."
        )

    st.subheader(
        "Does the prediction survive small changes?"
        if kind == "image"
        else "Where does the evidence change?"
    )

    chart_rows = []
    for row in rows:
        chart_rows.append({
            "Sample": row["sample"],
            "Fake score": row["fake_score"],
        })

    chart = (
        alt.Chart(alt.Data(values=chart_rows))
        .mark_bar(color="#5eead4", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Sample:N", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Fake score:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("Sample:N"),
                alt.Tooltip("Fake score:Q", format=".1%"),
            ],
        )
        .properties(height=240)
    )

    st.altair_chart(chart, use_container_width=True)

    if kind == "image":
        st.caption(
            "The original score remains the primary result. The extra views "
            "test sensitivity to JPEG compression, resizing and brightness. "
            "Stable output can still be wrong."
        )
    else:
        st.caption(
            "Each frame is classified independently. Short manipulations "
            "between samples may be missed."
        )

    for start in range(0, len(rows), 4):
        columns = st.columns(4)
        for offset, row in enumerate(rows[start:start + 4]):
            with columns[offset]:
                st.image(previews[start + offset], use_container_width=True)
                st.write(row["sample"])

                if kind == "video":
                    timestamp = row.get("timestamp_seconds")
                    if timestamp is not None:
                        st.caption(f"Timestamp: {timestamp:.3f} seconds")
                    else:
                        st.caption(f"Decoded frame index: {row['frame_index']}")

                st.caption(f"Fake score: {row['fake_score']:.1%}")

    st.subheader("What should you do next?")

    if result["verdict"].startswith("INCONCLUSIVE"):
        st.write(
            "Obtain a higher-quality original, inspect the conflicting samples, "
            "and compare the source before drawing a conclusion."
        )
    elif result["verdict"] == "FAKE-LEANING":
        st.write(
            "Check the original publication and provenance. Treat the score as "
            "a reason to investigate, not sufficient evidence to accuse someone."
        )
    else:
        st.write(
            "Use the result as one screening signal. Check provenance if "
            "establishing who created the media or when it was recorded matters."
        )

    st.download_button(
        "↓ Download evidence report",
        data=json.dumps(report, indent=2, allow_nan=False),
        file_name=f"verity-report-{report['file']['sha256'][:12]}.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Inspect report and model identity"):
        st.json(report)


# ----------------------------------------------------------------------
# Main interface
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">VERITY / DEEPFAKE EVIDENCE STUDIO</div>
        <h1>Before you trust it,<br><span>put it to the test.</span></h1>
        <p>
            Inspect a prediction, test its stability, and keep the evidence.
            Built for people who need to explain their decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

analyze_tab, why_tab, method_tab = st.tabs(
    ["◈ Analyze", "Why choose Verity?", "Model & method"]
)

with why_tab:
    st.header("Choose it when you need to defend a result.")
    st.write(
        "Verity connects a detector's output to a practical review process: "
        "what was tested, whether the score changed, which frames were "
        "sampled, and what another reviewer needs to reproduce the check."
    )

    features = [
        (
            "01 / CHALLENGE",
            "Stress-test the prediction",
            "Check whether modest JPEG compression, resizing or brightness "
            "changes alter the image result.",
        ),
        (
            "02 / LOCATE",
            "Inspect video samples",
            "Review individual frames with timestamps and separate scores, "
            "instead of relying solely on a whole-video average.",
        ),
        (
            "03 / DOCUMENT",
            "Keep a reviewable record",
            "Export scores, settings, model revision and the exact file's "
            "SHA-256 fingerprint in one report.",
        ),
    ]

    for column, (number, title, text) in zip(st.columns(3), features):
        with column:
            st.markdown(
                f"""
                <div class="card">
                    <div class="number">{number}</div>
                    <h3>{title}</h3>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("What you gain beyond a single prediction")

    st.table([
        {
            "Your question": "Does a small edit change the answer?",
            "A single prediction provides": "One score",
            "Verity adds": "Four image views and sensitivity checks",
        },
        {
            "Your question": "Which video frames support the result?",
            "A single prediction provides": "An overall label",
            "Verity adds": "Sample thumbnails, timestamps and scores",
        },
        {
            "Your question": "What if the evidence conflicts?",
            "A single prediction provides": "The highest-scoring class",
            "Verity adds": "Explicit review rules and an inconclusive outcome",
        },
        {
            "Your question": "Can someone review my analysis?",
            "A single prediction provides": "A displayed result",
            "Verity adds": "A downloadable report with file and model identity",
        },
    ])

    st.caption(
        "This compares workflows. It does not claim that every competing "
        "product lacks these features."
    )

    st.subheader("Where the project adds value")
    st.write(
        "The contribution is the evidence and review workflow built around "
        "a public pretrained classifier. These checks do not create a newly "
        "trained model or establish higher detection accuracy."
    )

    st.info(
        "A strong first choice for an inspectable screening demo, a research "
        "prototype, or a documented manual review. Universal superiority "
        "would require independent tests against other detectors on the "
        "same held-out datasets."
    )

with method_tab:
    st.header("Know exactly what is doing the detecting.")

    st.markdown(
        f"""
        **Checkpoint:** [{MODEL_ID}]({MODEL_URL})

        **Architecture:** Vision Transformer, ViT Base with 16 × 16 patches.

        **Model input:** RGB images processed to 224 × 224.

        **Classes:** Realism and Deepfake.

        **Author-reported accuracy:** 92.12% on 56,001 test examples.
        This is the author's evaluation, not a benchmark of this application.
        """
    )

    st.subheader("How review rules work")
    st.markdown(
        """
        - **Images:** the original image supplies the main score. Optional
          transformed views test sensitivity; their scores do not replace it.
        - **Videos:** the displayed aggregate is the mean of sampled-frame scores.
        - **Review band:** scores inside the selected middle band are inconclusive.
        - **Conflicting evidence:** substantial image sensitivity or strongly
          conflicting video samples can also trigger review.
        """
    )

    st.write(
        "The thresholds are application heuristics. Transformations are "
        "correlated checks using the same model, and no accuracy improvement "
        "is assumed."
    )

    st.subheader("Scope")
    st.write(
        "The checkpoint was trained on real and deepfake face images. The app "
        "does not automatically detect faces, inspect audio, localize altered "
        "pixels, or analyze motion. Unseen generators, compression and "
        "out-of-scope content can produce incorrect results."
    )

    st.subheader("Data handling")
    st.write(
        "On Streamlit Cloud, uploaded media is processed on the server. "
        "This code does not send media to an external inference API. "
        "Temporary video files are removed after analysis; current results "
        "and thumbnails are held in your Streamlit session. Hosting-platform "
        "retention and logging are outside this application's control."
    )

with analyze_tab:
    st.caption(
        "For images and videos containing visible human faces. "
        "First use downloads the model."
    )

    with st.expander("Review settings", expanded=False):
        context = st.selectbox(
            "Input context",
            [
                "A human face is visible",
                "Other content / unsure",
            ],
        )

        audit = st.checkbox(
            "Run image stability checks",
            value=True,
            help="Analyze the original plus three modest transformations.",
        )

        frames = st.slider(
            "Video samples",
            min_value=4,
            max_value=12,
            value=8,
            step=2,
        )

        high = st.slider(
            "Outer decision threshold",
            min_value=0.60,
            max_value=0.90,
            value=0.70,
            step=0.05,
            help="At 0.70, scores between 0.30 and 0.70 require review.",
        )

        spread_limit = st.slider(
            "Image score variation allowed",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
            help="0.20 means a range of 20 percentage points.",
        )

    settings = {
        "context": context,
        "audit": audit,
        "frames": frames,
        "low": round(1.0 - high, 2),
        "high": high,
        "spread_limit": spread_limit,
    }

    upload = st.file_uploader(
        "Drop media here",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "mkv", "avi", "webm"],
        help=f"Application limit: {MAX_UPLOAD_MB} MB.",
    )

    if upload is None:
        st.info("Choose a file to create an evidence brief.")

    elif upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload a file smaller than {MAX_UPLOAD_MB} MB.")

    else:
        data = upload.getvalue()
        suffix = Path(upload.name).suffix.lower()
        kind = (
            "image"
            if suffix in {".jpg", ".jpeg", ".png", ".webp"}
            else "video"
        )

        file_hash = hashlib.sha256(data).hexdigest()
        analysis_key = hashlib.sha256(
            (
                file_hash
                + upload.name
                + json.dumps(settings, sort_keys=True)
                + APP_VERSION
            ).encode("utf-8")
        ).hexdigest()

        try:
            if kind == "image":
                st.image(
                    thumbnail(load_image(data)),
                    caption=upload.name,
                )
            else:
                st.video(data)
        except Exception as error:
            st.error(f"Cannot preview this file: {error}")
            st.stop()

        if st.button(
            "Create evidence brief →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.pop("verity_analysis", None)
            status = st.empty()

            try:
                report, previews = analyze(
                    data,
                    upload.name,
                    kind,
                    settings,
                    status,
                )

                st.session_state["verity_analysis"] = {
                    "key": analysis_key,
                    "report": report,
                    "previews": previews,
                }

                status.empty()

            except Exception as error:
                status.empty()
                st.error("Analysis could not be completed.")
                with st.expander("Error details"):
                    st.code(f"{type(error).__name__}: {error}")

        saved = st.session_state.get("verity_analysis")

        if saved and saved["key"] == analysis_key:
            show_results(saved["report"], saved["previews"])
        elif saved:
            st.caption(
                "The file or review settings changed. Run analysis again "
                "to produce a matching report."
            )

st.divider()
st.caption(
    "VERITY · Inspectable deepfake screening · "
    "Built on a public model, with evidence you can review."
)
