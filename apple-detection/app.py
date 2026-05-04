import importlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from config import DEFAULT_CHECKPOINT

st: Any = importlib.import_module("streamlit")

st.set_page_config(page_title="Apple Detection", page_icon="🍎", layout="centered")


def predict_demo(image: Image.Image) -> tuple[bool, float]:
    """Heuristic fallback when no checkpoint is available.

    Conservative on purpose: avoids calling common banana/yellow images "apple".
    Returns (is_apple, confidence_of_that_boolean).
    """

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False, 0.55

    # Center-crop to reduce background influence.
    h, w, _ = arr.shape
    y0, y1 = int(h * 0.15), int(h * 0.85)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    if y1 > y0 and x1 > x0:
        arr = arr[y0:y1, x0:x1]

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    # reasonably vivid pixels
    valid = (delta > 0.12) & (maxc > 0.25)
    if not np.any(valid):
        return False, 0.55

    # Hue in [0,1)
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    r_is_max = mask & (maxc == r)
    g_is_max = mask & (maxc == g)
    b_is_max = mask & (maxc == b)

    hue[r_is_max] = ((g[r_is_max] - b[r_is_max]) / delta[r_is_max]) % 6.0
    hue[g_is_max] = ((b[g_is_max] - r[g_is_max]) / delta[g_is_max]) + 2.0
    hue[b_is_max] = ((r[b_is_max] - g[b_is_max]) / delta[b_is_max]) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(maxc)
    saturation[mask] = delta[mask] / np.maximum(maxc[mask], 1e-6)

    vivid = valid & (saturation > 0.20)

    # Banana/yellow rejection: hue ~ 40-70 degrees.
    yellow = vivid & (hue >= 0.11) & (hue <= 0.19)
    yellow_ratio = float(np.mean(yellow))
    if yellow_ratio >= 0.03:
        confidence = min(0.92, 0.60 + yellow_ratio * 5.0)
        return False, float(confidence)

    # Apple-ish hues: red or green.
    red = vivid & ((hue <= 0.05) | (hue >= 0.95))
    green = vivid & (hue >= 0.24) & (hue <= 0.45)
    score = float(max(np.mean(red), np.mean(green)))

    # Conservative threshold.
    threshold = 0.08
    sharpness = 85.0
    p_apple = 1.0 / (1.0 + np.exp(-(score - threshold) * sharpness))
    p_apple = float(np.clip(p_apple, 0.10, 0.90))

    is_apple = p_apple >= 0.5
    confidence = p_apple if is_apple else (1.0 - p_apple)
    return bool(is_apple), float(confidence)


st.markdown(
    """
    <style>
        :root {
            --ink: #111827;
            --muted: #5f6c7b;
            --panel: #ffffff;
            --surface: #f7f8fb;
            --accent: #2d6cdf;
            --accent-soft: rgba(45, 108, 223, 0.12);
            --border: rgba(17, 24, 39, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(45, 108, 223, 0.12), transparent 38%),
                radial-gradient(circle at top right, rgba(15, 23, 42, 0.08), transparent 34%),
                linear-gradient(180deg, #fbfcfe 0%, #f2f5fb 100%);
            color: var(--ink);
        }
        html, body, [class*="css"]  {
            font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
        }
        .block-container {
            max-width: 980px;
            padding-top: 3rem;
            padding-bottom: 3.5rem;
        }
        .hero-card {
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.6rem 1.9rem;
            background: var(--panel);
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.6rem;
        }
        .hero-title {
            font-size: 2.6rem;
            line-height: 1.05;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }
        .hero-copy {
            font-size: 1.02rem;
            color: var(--muted);
        }
        .result-card {
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        }
        .verdict {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--ink);
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.85rem;
            font-weight: 600;
        }
        .muted {
            color: var(--muted);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Apple Detection</div>
        <div class="hero-copy">Upload an image to get <b>True/False</b> (Apple) and confidence.</div>
        <div class="pill" style="margin-top: 0.8rem;">Vision classifier</div>
    </div>
    """,
    unsafe_allow_html=True,
)

checkpoint_path = Path(DEFAULT_CHECKPOINT)

with st.sidebar:
    st.markdown("### Settings")
    model_source = st.radio(
        "Model source",
        options=["Default checkpoint", "Upload checkpoint", "Demo mode (no model)"],
        index=0 if checkpoint_path.exists() else 2,
    )
    uploaded_checkpoint: Optional[Any]
    if model_source == "Upload checkpoint":
        uploaded_checkpoint = st.file_uploader("Checkpoint file", type=["pt", "pth"], key="checkpoint")
    else:
        uploaded_checkpoint = None

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if model_source == "Default checkpoint" and not checkpoint_path.exists():
    st.markdown(
        f"""
        <div class="result-card">
            <div class="pill">Setup required</div>
            <div style="margin-top: 0.6rem; font-size: 1.1rem; font-weight: 600;">Default checkpoint not found</div>
            <div class="muted" style="margin-top: 0.35rem;">Expected at:</div>
            <div style="font-size: 0.95rem; margin-top: 0.15rem;">{checkpoint_path}</div>
            <div class="muted" style="margin-top: 0.6rem;">Use “Upload checkpoint” or “Demo mode (no model)” in the sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded_file is None:
    st.info("Choose an image, then click Upload to run a prediction.")
    st.stop()

image = Image.open(uploaded_file)
preview_col, result_col = st.columns([1.05, 0.95], gap="large")

with preview_col:
    st.subheader("Preview")
    st.image(image, width="stretch")

with result_col:
    st.subheader("Prediction")
    if model_source == "Upload checkpoint" and uploaded_checkpoint is None:
        st.info("Upload a checkpoint file in the sidebar to enable model predictions.")
    else:
        try:
            if model_source == "Demo mode (no model)":
                verdict, confidence = predict_demo(image)
                st.caption("Demo mode is a simple heuristic (not a trained model).")
            elif model_source == "Upload checkpoint":
                assert uploaded_checkpoint is not None
                from src.inference import predict_is_apple_from_bytes

                verdict, confidence = predict_is_apple_from_bytes(image, uploaded_checkpoint.getvalue())
            else:
                from src.inference import predict_is_apple

                verdict, confidence = predict_is_apple(image, checkpoint_path)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            verdict_text = "True" if verdict else "False"
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="muted" style="font-weight: 600;">Apple</div>
                    <div class="verdict">{verdict_text}</div>
                    <div class="muted" style="margin-top: 0.7rem;">Confidence</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: var(--accent);">{confidence:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
