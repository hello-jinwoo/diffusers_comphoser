"""Package-owned public training entrypoint for ComPhoser."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Sequence


def _load_trainer_module():
    return import_module("comphoser.trainer")


def run_with_args(args: Any) -> int:
    return _load_trainer_module().run_with_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    return _load_trainer_module().cli_main(argv)


__all__ = ["main", "run_with_args"]
