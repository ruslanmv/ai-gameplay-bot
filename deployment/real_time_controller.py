"""
real_time_controller.py (production-ready)

- Timeouts + retries (prevents hanging)
- Clear, structured errors (no noisy stack traces by default)
- Validates state shape (expects 128 floats)
- Lets you override base URLs via env vars:
    NN_API_URL=http://localhost:5000
    TRANSFORMER_API_URL=http://localhost:5001
- Uses /health to detect if service is up (optional helper)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------
# Configuration
# ----------------------------
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "128"))

NN_BASE_URL = os.getenv("NN_API_URL", "http://localhost:5000").rstrip("/")
TRANSFORMER_BASE_URL = os.getenv("TRANSFORMER_API_URL", "http://localhost:5001").rstrip("/")

NN_PREDICT_URL = f"{NN_BASE_URL}/predict"
TRANSFORMER_PREDICT_URL = f"{TRANSFORMER_BASE_URL}/predict"

DEFAULT_TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT", "5"))
DEFAULT_RETRIES = int(os.getenv("PREDICT_RETRIES", "2"))  # additional attempts
DEFAULT_BACKOFF = float(os.getenv("PREDICT_BACKOFF", "0.3"))


# ----------------------------
# HTTP session with retries
# ----------------------------
def _build_session(retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF) -> requests.Session:
    session = requests.Session()

    retry_cfg = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


# ----------------------------
# Validation helpers
# ----------------------------
def _validate_state(state: Sequence[float]) -> List[float]:
    if not isinstance(state, (list, tuple, np.ndarray)):
        raise ValueError(f"'state' must be a list/tuple/ndarray of length {INPUT_SIZE}")

    if len(state) != INPUT_SIZE:
        raise ValueError(f"'state' must have length {INPUT_SIZE}, got {len(state)}")

    # Convert to plain python floats (JSON-serializable)
    try:
        return [float(x) for x in state]
    except Exception as e:
        raise ValueError(f"'state' must contain numeric values: {e}") from e


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {"_raw": data}
    except Exception:
        # Not JSON
        return {"_raw": resp.text}


# ----------------------------
# Core request logic
# ----------------------------
def _predict(url: str, state: Sequence[float], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    payload = {"state": _validate_state(state)}

    try:
        resp = _SESSION.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        # Network or connection error
        raise RuntimeError(f"Failed to reach prediction service at {url}: {e}") from e

    if resp.status_code != 200:
        body = _safe_json(resp)
        msg = body.get("error") or body.get("message") or str(body)
        raise RuntimeError(f"Prediction service error ({resp.status_code}) at {url}: {msg}")

    body = _safe_json(resp)
    action = body.get("action")
    if not action:
        raise RuntimeError(f"Malformed response from {url}: missing 'action' field. Body: {body}")

    return str(action)


def get_action_from_nn(state: Sequence[float], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Get the predicted action from the Neural Network model.
    """
    return _predict(NN_PREDICT_URL, state, timeout=timeout)


def get_action_from_transformer(state: Sequence[float], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Get the predicted action from the Transformer model.
    """
    return _predict(TRANSFORMER_PREDICT_URL, state, timeout=timeout)


def unified_predictor(
    state: Sequence[float],
    use_transformer: bool = True,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """
    Unified predictor that uses either the Neural Network or Transformer model.
    """
    if use_transformer:
        return get_action_from_transformer(state, timeout=timeout)
    return get_action_from_nn(state, timeout=timeout)


# Optional helpers (useful in prod to check readiness)
def is_service_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = _SESSION.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ----------------------------
# CLI usage
# ----------------------------
def _main() -> int:
    example_state = np.random.rand(INPUT_SIZE).tolist()

    print(f"NN service healthy? {is_service_healthy(NN_BASE_URL)} ({NN_BASE_URL})")
    print(f"Transformer service healthy? {is_service_healthy(TRANSFORMER_BASE_URL)} ({TRANSFORMER_BASE_URL})")
    print()

    try:
        print("Using Neural Network:")
        action_nn = unified_predictor(example_state, use_transformer=False)
        print(f"Predicted Action (NN): {action_nn}")
    except Exception as e:
        print(f"[NN ERROR] {e}", file=sys.stderr)

    print()

    try:
        print("Using Transformer:")
        action_tr = unified_predictor(example_state, use_transformer=True)
        print(f"Predicted Action (Transformer): {action_tr}")
    except Exception as e:
        print(f"[TRANSFORMER ERROR] {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
