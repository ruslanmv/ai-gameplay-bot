"""
End-to-end smoke test (production-safe):

1) Quick-train NN (~2s) -> models/neural_network/nn_model_finetuned.pth
2) Quick-train Transformer (~2s) -> models/transformer/transformer_model_finetuned.pth
3) Start deploy_nn.py + deploy_transformer.py
4) Start control_backend.py
5) Verify /health and real /api/test_predict calls
6) ALWAYS stop everything cleanly (success or failure)

Run:
  uv run python scripts/smoke_e2e.py
"""

from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

import requests
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

NN_WEIGHTS = PROJECT / "models/neural_network/nn_model_finetuned.pth"
TR_WEIGHTS = PROJECT / "models/transformer/transformer_model_finetuned.pth"

NN_URL = "http://127.0.0.1:5000"
TR_URL = "http://127.0.0.1:5001"
CB_URL = "http://127.0.0.1:8000"

TIMEOUT_START = 12.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _wait_health(url: str, timeout: float = TIMEOUT_START) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _terminate(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    finally:
        print(f"[CLEANUP] {name} stopped")


# -----------------------------------------------------------------------------
# Quick training
# -----------------------------------------------------------------------------
def quick_train_nn(seconds: float = 2.0) -> None:
    from models.neural_network.nn_model import GameplayNN

    model = GameplayNN(input_size=128, hidden_size=64, output_size=10)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    steps = 0
    while time.time() - start < seconds:
        x = torch.randn(64, 128)
        y = torch.randint(0, 10, (64,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        steps += 1

    NN_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), NN_WEIGHTS)
    print(f"[OK] NN trained {steps} steps, saved -> {NN_WEIGHTS}")


def quick_train_transformer(seconds: float = 2.0) -> None:
    from models.transformer.transformer_model import GameplayTransformer

    model = GameplayTransformer(
        input_size=128,
        num_heads=4,
        hidden_size=64,
        num_layers=2,
        output_size=10,
    )
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()
    steps = 0
    while time.time() - start < seconds:
        x = torch.randn(32, 128)
        y = torch.randint(0, 10, (32,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        steps += 1

    TR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TR_WEIGHTS)
    print(f"[OK] Transformer trained {steps} steps, saved -> {TR_WEIGHTS}")


# -----------------------------------------------------------------------------
# Main test
# -----------------------------------------------------------------------------
def main() -> int:
    print("== Smoke E2E: quick train + start services + API checks ==")

    nn_proc = tr_proc = cb_proc = None

    try:
        # 1) Train models
        quick_train_nn(2.0)
        quick_train_transformer(2.0)

        # 2) Spawn services
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT)

        def spawn(script: Path) -> subprocess.Popen:
            preexec = os.setsid if os.name != "nt" else None
            return subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT),
                env=env,
                preexec_fn=preexec,
            )

        nn_proc = spawn(PROJECT / "deployment/deploy_nn.py")
        tr_proc = spawn(PROJECT / "deployment/deploy_transformer.py")
        cb_proc = spawn(PROJECT / "deployment/control_backend.py")

        # 3) Health checks
        if not _wait_health(NN_URL):
            raise RuntimeError("NN service did not become healthy")
        if not _wait_health(TR_URL):
            raise RuntimeError("Transformer service did not become healthy")
        if not _wait_health(CB_URL):
            raise RuntimeError("Control backend did not become healthy")

        # 4) API checks
        status = requests.get(f"{CB_URL}/api/status", timeout=3).json()
        print("[OK] /api/status:", status)

        for model in ("nn", "transformer"):
            r = requests.post(
                f"{CB_URL}/api/test_predict",
                json={"model": model},
                timeout=5,
            )
            print(f"[OK] /api/test_predict ({model}) ->", r.status_code, r.json())

        print("\n[PASS] End-to-end smoke test succeeded.")
        print(f"UI available at: {CB_URL}/")
        return 0

    except Exception as e:
        print(f"\n[FAIL] Smoke test failed: {e}")
        return 1

    finally:
        # 5) ALWAYS cleanup
        _terminate(cb_proc, "control_backend")
        _terminate(tr_proc, "transformer_service")
        _terminate(nn_proc, "nn_service")


if __name__ == "__main__":
    raise SystemExit(main())
