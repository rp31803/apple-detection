import json
import base64
import io
from typing import Tuple
import numpy as np
from PIL import Image

def predict_demo(image: Image.Image) -> Tuple[bool, float]:
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

def handler(event, context):
    if event['httpMethod'] != 'POST':
        return {
            'statusCode': 405,
            'body': json.dumps({'error': 'Method not allowed'})
        }

    try:
        body = json.loads(event['body'])
        if 'image' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }

        # Decode base64 image
        image_data = base64.b64decode(body['image'])
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Use demo mode
        verdict, confidence = predict_demo(image)
        model_type = "demo"

        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'verdict': verdict,
                'confidence': confidence,
                'image_data': img_str,
                'model_type': model_type
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Prediction failed: {str(e)}'})
        }