"""
deployment/stream_sessions.py

Stream session manager:
- Resolve YouTube/Twitch URLs to direct media URLs via yt-dlp
- Capture frames via OpenCV
- Convert frames -> 128-dim state (16x8 grayscale flattened)
- Inference by calling existing /predict services (NN/Transformer)
- Realtime events via Server-Sent Events (SSE)

Train mode (lightweight online finetune):
- Uses predicted actions as pseudo-labels
- Performs small online updates on the NN weights
- Periodically saves a .pth to models/neural_network/uploads/stream_<session>.pth
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, List

import cv2
import numpy as np
import requests


ACTION_TO_INDEX = {
    "move_forward": 0,
    "move_backward": 1,
    "turn_left": 2,
    "turn_right": 3,
    "attack": 4,
    "jump": 5,
    "interact": 6,
    "use_item": 7,
    "open_inventory": 8,
    "cast_spell": 9,
}


def _now() -> float:
    return time.time()


def _safe_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def resolve_stream_url(url: str) -> str:
    """
    Resolve a YouTube/Twitch page URL to a direct media URL using yt-dlp.
    Requires `yt-dlp` installed (pip install yt-dlp) and typically ffmpeg available.
    """
    cmd = ["yt-dlp", "-g", "--no-warnings", "--no-playlist", url]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=25).strip()
        # yt-dlp may output multiple lines; first is usually best
        direct = out.splitlines()[0].strip()
        if not direct:
            raise RuntimeError("yt-dlp returned empty direct URL")
        return direct
    except FileNotFoundError as e:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed: {e.output[:4000]}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("yt-dlp timed out resolving the stream URL") from e


def frame_to_state(frame_bgr: np.ndarray) -> List[float]:
    """
    Convert a BGR frame -> 128-dim feature vector:
      - grayscale
      - resize to (16, 8)
      - flatten to 128 floats in [0,1]
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)  # 16*8=128
    vec = (small.astype(np.float32) / 255.0).reshape(-1)
    return vec.tolist()


def jpeg_b64(frame_bgr: np.ndarray, max_w: int = 320) -> str:
    """Encode a small thumbnail as base64 JPEG for UI previews."""
    h, w = frame_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class StreamSession:
    session_id: str
    mode: str  # "train" | "infer"
    url: str
    model_type: str  # "nn" | "transformer"
    direct_url: str
    nn_port: int
    tr_port: int
    repo_root: Path
    nn_active_weights: Path

    # runtime
    started_at: float = field(default_factory=_now)
    stopped_at: Optional[float] = None
    last_event_at: Optional[float] = None
    frames: int = 0
    fps_est: float = 0.0
    last_action: Optional[str] = None
    last_conf: Optional[float] = None
    last_error: Optional[str] = None
    training_steps: int = 0
    saved_model_path: Optional[str] = None

    _stop_flag: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _events: "queue.Queue[dict]" = field(default_factory=lambda: queue.Queue(maxsize=500), init=False, repr=False)

    # online training (NN only)
    _trainer: Optional["OnlineNNTrainer"] = field(default=None, init=False, repr=False)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop_flag.set()

    def _predict_action(self, state: List[float]) -> Tuple[str, Optional[float]]:
        port = self.nn_port if self.model_type == "nn" else self.tr_port
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/predict", json={"state": state}, timeout=6)
            if resp.status_code != 200:
                raise RuntimeError(f"predict HTTP {resp.status_code}")
            data = resp.json()
            action = data.get("action") or "unknown_action"
            # predictors currently return only action string; confidence is unknown
            return action, None
        except Exception as e:
            raise RuntimeError(f"predict failed: {e}") from e

    def _push_event(self, ev: dict) -> None:
        ev["session_id"] = self.session_id
        ev["t"] = _now()
        self.last_event_at = ev["t"]
        try:
            self._events.put_nowait(ev)
        except queue.Full:
            # drop oldest to make room
            try:
                _ = self._events.get_nowait()
            except Exception:
                pass
            try:
                self._events.put_nowait(ev)
            except Exception:
                pass

    def iter_sse(self) -> Iterator[str]:
        """
        SSE generator. Yields:
          data: { ... }\n\n
        """
        # initial hello
        yield f"data: {_safe_json({'type': 'hello', 'session_id': self.session_id})}\n\n"
        while True:
            if self.stopped_at is not None and self._events.empty():
                yield f"data: {_safe_json({'type': 'done', 'session_id': self.session_id})}\n\n"
                break
            try:
                ev = self._events.get(timeout=0.75)
                yield f"data: {_safe_json(ev)}\n\n"
            except queue.Empty:
                # keep-alive comment
                yield ": ping\n\n"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "url": self.url,
            "model_type": self.model_type,
            "direct_url": self.direct_url,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_event_at": self.last_event_at,
            "frames": self.frames,
            "fps_est": self.fps_est,
            "last_action": self.last_action,
            "last_conf": self.last_conf,
            "last_error": self.last_error,
            "training_steps": self.training_steps,
            "saved_model_path": self.saved_model_path,
            "running": self.is_running(),
        }

    def start(self, include_frames: bool = True, max_fps: float = 8.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={"include_frames": include_frames, "max_fps": max_fps},
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self, include_frames: bool = True, max_fps: float = 8.0) -> None:
        t0 = _now()
        last_tick = t0
        frames_for_fps = 0

        # NN online trainer (only when mode=train and model_type=nn)
        if self.mode == "train" and self.model_type == "nn":
            try:
                self._trainer = OnlineNNTrainer(repo_root=self.repo_root, active_weights=self.nn_active_weights)
            except Exception as e:
                self.last_error = f"trainer init failed: {e}"
                self._push_event({"type": "error", "message": self.last_error})
                self.stopped_at = _now()
                return

        # open stream
        cap = cv2.VideoCapture(self.direct_url)
        if not cap.isOpened():
            self.last_error = "OpenCV could not open stream URL"
            self._push_event({"type": "error", "message": self.last_error})
            self.stopped_at = _now()
            return

        min_dt = 1.0 / max(0.5, float(max_fps))

        self._push_event(
            {
                "type": "started",
                "mode": self.mode,
                "model_type": self.model_type,
                "include_frames": include_frames,
                "max_fps": max_fps,
            }
        )

        try:
            while not self._stop_flag.is_set():
                now = _now()
                if now - last_tick < min_dt:
                    time.sleep(0.005)
                    continue
                last_tick = now

                ok, frame = cap.read()
                if not ok or frame is None:
                    self.last_error = "stream ended or frame read failed"
                    self._push_event({"type": "ended", "message": self.last_error})
                    break

                self.frames += 1
                frames_for_fps += 1

                # update fps estimate every ~2 seconds
                if now - t0 >= 2.0:
                    self.fps_est = frames_for_fps / (now - t0)
                    t0 = now
                    frames_for_fps = 0

                state = frame_to_state(frame)
                try:
                    action, conf = self._predict_action(state)
                    self.last_action, self.last_conf = action, conf
                except Exception as e:
                    self.last_error = str(e)
                    self._push_event({"type": "error", "message": self.last_error})
                    continue

                thumb = jpeg_b64(frame) if include_frames else ""

                ev = {
                    "type": "inference",
                    "frame": self.frames,
                    "action": action,
                    "confidence": conf,
                    "fps": self.fps_est,
                    "thumb_jpeg_b64": thumb,
                }
                self._push_event(ev)

                # online training step
                if self.mode == "train" and self.model_type == "nn" and self._trainer is not None:
                    idx = ACTION_TO_INDEX.get(action)
                    if idx is not None:
                        loss = self._trainer.step(state, idx)
                        self.training_steps += 1
                        if self.training_steps % 20 == 0:
                            self._push_event({"type": "train", "step": self.training_steps, "loss": loss})
                        if self.training_steps % 200 == 0:
                            saved = self._trainer.save_stream_checkpoint(self.session_id)
                            self.saved_model_path = saved
                            self._push_event({"type": "checkpoint", "path": saved, "step": self.training_steps})

        finally:
            cap.release()
            self.stopped_at = _now()
            self._push_event({"type": "stopped"})


class OnlineNNTrainer:
    """
    Very lightweight online finetune loop for GameplayNN.
    Uses NLLLoss over log(probabilities) because the model outputs softmax probs.
    """

    def __init__(self, repo_root: Path, active_weights: Path, lr: float = 1e-4) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        from models.neural_network.nn_model import GameplayNN

        self.repo_root = repo_root
        self.active_weights = active_weights

        self.device = "cpu"
        self.model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
        if active_weights.exists():
            state = torch.load(str(active_weights), map_location="cpu")
            self.model.load_state_dict(state)
        self.model.train()

        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.crit = nn.NLLLoss()

        self._torch = torch

    def step(self, state: List[float], label_idx: int) -> float:
        t = self._torch.tensor(state, dtype=self._torch.float32).unsqueeze(0)
        y = self._torch.tensor([label_idx], dtype=self._torch.long)

        self.opt.zero_grad()
        probs = self.model(t)  # softmax probs
        logp = self._torch.log(probs + 1e-8)
        loss = self.crit(logp, y)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def save_stream_checkpoint(self, session_id: str) -> str:
        import torch

        out_dir = self.repo_root / "models" / "neural_network" / "uploads"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stream_{session_id}.pth"
        torch.save(self.model.state_dict(), str(out_path))
        return str(out_path.relative_to(self.repo_root))


class StreamManager:
    def __init__(self, repo_root: Path, nn_port: int, tr_port: int, nn_active_weights: Path) -> None:
        self.repo_root = repo_root
        self.nn_port = nn_port
        self.tr_port = tr_port
        self.nn_active_weights = nn_active_weights
        self._lock = threading.Lock()
        self._sessions: Dict[str, StreamSession] = {}

    def create_session(
        self,
        session_id: str,
        mode: str,
        url: str,
        model_type: str,
        include_frames: bool = True,
        max_fps: float = 8.0,
    ) -> StreamSession:
        mode = (mode or "").strip().lower()
        model_type = (model_type or "").strip().lower()

        if mode not in ("train", "infer"):
            raise ValueError("mode must be 'train' or 'infer'")
        if model_type not in ("nn", "transformer"):
            raise ValueError("model_type must be 'nn' or 'transformer'")

        direct = resolve_stream_url(url)

        sess = StreamSession(
            session_id=session_id,
            mode=mode,
            url=url,
            model_type=model_type,
            direct_url=direct,
            nn_port=self.nn_port,
            tr_port=self.tr_port,
            repo_root=self.repo_root,
            nn_active_weights=self.nn_active_weights,
        )

        with self._lock:
            # stop and replace if exists
            old = self._sessions.get(session_id)
            if old:
                try:
                    old.stop()
                except Exception:
                    pass
            self._sessions[session_id] = sess

        sess.start(include_frames=include_frames, max_fps=max_fps)
        return sess

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            sess = self._sessions.get(session_id)
        if not sess:
            return False
        sess.stop()
        return True

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: s.to_dict() for sid, s in self._sessions.items()}


def make_default_manager(repo_root: Path, nn_port: int, tr_port: int, nn_active_weights: Path) -> StreamManager:
    return StreamManager(repo_root=repo_root, nn_port=nn_port, tr_port=tr_port, nn_active_weights=nn_active_weights)
