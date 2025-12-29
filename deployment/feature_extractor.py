"""
deployment/feature_extractor.py

Shared utilities:
- decode base64/data-url images
- convert to 128-dim feature vector (16x8 grayscale normalized)
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def decode_image_to_pil(image_str: str) -> Image.Image:
    """
    Accepts:
      - raw base64 string
      - data URL: data:image/jpeg;base64,...
    Returns PIL Image (RGB).
    """
    if not isinstance(image_str, str) or not image_str.strip():
        raise ValueError("image must be a non-empty string")

    s = image_str.strip()
    if s.startswith("data:"):
        # data:image/...;base64,XXXX
        try:
            s = s.split(",", 1)[1]
        except Exception as e:
            raise ValueError("Invalid data URL format") from e

    try:
        raw = base64.b64decode(s, validate=False)
    except Exception as e:
        raise ValueError("Invalid base64 image") from e

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError("Decoded bytes are not a valid image") from e

    return img


def image_to_features_128(image_str: str) -> list[float]:
    """
    Convert an image to a 128-dim feature vector:
      - grayscale
      - resize to (16, 8) => 128 pixels
      - flatten and normalize to [0,1]
    """
    img = decode_image_to_pil(image_str).convert("L")
    img = img.resize((16, 8))
    arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
    if arr.size != 128:
        raise RuntimeError(f"Unexpected feature size {arr.size}, expected 128")
    return arr.tolist()


def safe_features_from_payload(payload: dict) -> Tuple[Optional[list[float]], Optional[str]]:
    """
    Accept either:
      - payload["features"] (list of floats length 128)
      - payload["image"] (base64/data-url)
    Returns (features, error_message).
    """
    if not isinstance(payload, dict):
        return None, "payload must be a JSON object"

    if "features" in payload and payload["features"] is not None:
        feats = payload["features"]
        if not isinstance(feats, list) or len(feats) != 128:
            return None, "features must be a list of length 128"
        for i, v in enumerate(feats):
            if not isinstance(v, (int, float)):
                return None, f"features[{i}] must be numeric"
        return [float(v) for v in feats], None

    if "image" in payload and payload["image"] is not None:
        try:
            feats = image_to_features_128(payload["image"])
            return feats, None
        except Exception as e:
            return None, str(e)

    return None, "Provide either 'image' (base64/data-url) or 'features' (len=128)."
