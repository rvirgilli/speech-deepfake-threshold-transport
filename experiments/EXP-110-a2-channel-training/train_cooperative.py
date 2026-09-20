"""Run the EXP-110 trainer with gpu-queue v2 epoch safe points.

The underlying recipe already writes ``epoch_N.pth`` and then atomically
replaces ``resume.pth``.  This launcher publishes progress only after that
replace succeeds, so a v2 pause acknowledgement always refers to a durable,
resumable checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

sys.path.insert(0, os.getcwd())
from train import SCRIPT, install_xlsr_compat


def _completed_epochs(model_dir: Path) -> int:
    completed = []
    for checkpoint in model_dir.glob("epoch_*.pth"):
        try:
            completed.append(int(checkpoint.stem.removeprefix("epoch_")) + 1)
        except ValueError:
            continue
    return max(completed, default=0)


def _total_epochs() -> int:
    configured = os.environ.get("GPUQ_WORK_TOTAL")
    if configured:
        return int(float(configured))
    try:
        index = sys.argv.index("--num_epochs")
        return int(sys.argv[index + 1])
    except (ValueError, IndexError):
        return 30


def _publish_safe_point(model_dir: Path) -> None:
    if not os.environ.get("GPUQ_PAUSE_REQUEST_PATH"):
        return
    current = _completed_epochs(model_dir)
    checkpoint = model_dir / "resume.pth"
    command = [
        shutil.which("gpuq2") or str(Path.home() / ".local/bin/gpuq2"),
        "safe-point",
        "--current",
        str(current),
        "--total",
        str(_total_epochs()),
        "--unit",
        "epoch",
        "--phase",
        "training",
    ]
    if checkpoint.is_file() and current:
        command.extend(
            [
                "--checkpoint-id",
                f"epoch-{current - 1:03d}",
                "--checkpoint-durable",
            ]
        )
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    model_dir = Path(os.environ["EXP110_MODELS_DIR"])
    install_xlsr_compat()
    _publish_safe_point(model_dir)

    original_replace = os.replace

    def replace_with_safe_point(
        source: str | bytes,
        destination: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        if Path(os.fsdecode(destination)).name == "resume.pth":
            _publish_safe_point(model_dir)

    os.replace = replace_with_safe_point
    sys.argv = [SCRIPT] + sys.argv[1:]
    runpy.run_path(SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
