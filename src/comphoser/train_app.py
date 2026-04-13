"""Package-owned public training entrypoint for ComPhoser."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .train_args import parse_args


def _ensure_repo_root_on_sys_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)


def _load_retained_trainer_main() -> Callable[[Any], None]:
    _ensure_repo_root_on_sys_path()
    from examples.dreambooth.train_dreambooth_lora_flux2_klein_img2img import main as retained_main

    return retained_main


def run_with_args(args: Any) -> int:
    retained_main = _load_retained_trainer_main()
    retained_main(args)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    return run_with_args(args)


__all__ = ["main", "run_with_args"]
