from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import modal


app = modal.App("cute-dsl-matmul")
REPO_ROOT = (
    Path(__file__).resolve().parents[4]
    if len(Path(__file__).resolve().parents) > 4
    else Path.cwd()
)
LOG_DIR = Path(__file__).resolve().parent / "logs" / "cute_dsl_matmul"
IMAGE = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .pip_install("nvidia-cutlass-dsl")
)


@app.function(image=IMAGE, gpu="H100", timeout=60 * 20)
def run_example(source_b64: str) -> dict:
    import subprocess
    import sys
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cute_dsl_matmul_"))
    source_path = root / "matmul.py"
    source_path.write_bytes(base64.b64decode(source_b64))
    process = subprocess.run(
        [sys.executable, str(source_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60 * 10,
    )
    return {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


@app.local_entrypoint()
def main():
    source = REPO_ROOT / "examples/cute_dsl/matmul.py"
    result = run_example.remote(base64.b64encode(source.read_bytes()).decode("ascii"))
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = LOG_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved results to {out_dir}")
