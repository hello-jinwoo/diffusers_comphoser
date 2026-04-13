"""Installable training entrypoint for ComPhoser."""

from __future__ import annotations

from typing import Sequence

from comphoser.train_app import main as train_main


def main(argv: Sequence[str] | None = None) -> int:
    return train_main(argv)


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
