"""
deployment/feature_extractor.py

Shared utilities:
- decode base64/data-url images
- convert to feature vector (default 128) as grayscale normalized
- accept legacy payload key "state" as alias for "features"
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple, List

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


def image_to_features(image_str: str, feature_len: int = 128) -> List[float]:
    """
    Convert an image to a feature vector of length feature_len:
      - grayscale
      - resize to (W,H) such that W*H == feature_len (best-effort)
      - flatten and normalize to [0,1]
    """
    if feature_len <= 0:
        raise ValueError("feature_len must be > 0")

    # Choose a stable shape: prefer 16x( feature_len/16 ) if divisible, else near-square
    if feature_len % 16 == 0:
        w, h = 16, feature_len // 16
    else:
        w = int(np.floor(np.sqrt(feature_len)))
        h = int(np.ceil(feature_len / max(w, 1)))
        # fix if mismatch due rounding
        while w * h < feature_len:
            w += 1
        # we'll crop/pad after resize if needed (rare)

    img = decode_image_to_pil(image_str).convert("L")
    img = img.resize((w, h))
    arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    if arr.size < feature_len:
        arr = np.pad(arr, (0, feature_len - arr.size), mode="constant", constant_values=0.0)
    elif arr.size > feature_len:
        arr = arr[:feature_len]

    return arr.tolist()


def safe_features_from_payload(payload: dict, expected_len: int = 128) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Accept either:
      - payload["features"] (list[float] length expected_len)
      - payload["state"]    (LEGACY alias of features)
      - payload["image"]    (base64/data-url -> features)
    Returns (features, error_message).
    """
    if not isinstance(payload, dict):
        return None, "payload must be a JSON object"

    # Legacy alias
    if "features" not in payload and "state" in payload:
        payload = dict(payload)
        payload["features"] = payload.get("state")

    if "features" in payload and payload["features"] is not None:
        feats = payload["features"]
        if not isinstance(feats, list) or len(feats) != expected_len:
            return None, f"features must be a list of length {expected_len}"
        for i, v in enumerate(feats):
            if not isinstance(v, (int, float)):
                return None, f"features[{i}] must be numeric"
        return [float(v) for v in feats], None

    if "image" in payload and payload["image"] is not None:
        try:
            feats = image_to_features(payload["image"], feature_len=expected_len)
            return feats, None
        except Exception as e:
            return None, str(e)

    return None, f"Provide either 'image' (base64/data-url) or 'features'/'state' (len={expected_len})."
