#!/usr/bin/env python
# coding=utf-8

"""Compatibility shim for the package-owned ComPhoser trainer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    _repo_src_str = str(_REPO_SRC)
    if _repo_src_str in sys.path:
        sys.path.remove(_repo_src_str)
    sys.path.insert(0, _repo_src_str)


def main(argv: Sequence[str] | None = None) -> int:
    from comphoser.trainer import cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
