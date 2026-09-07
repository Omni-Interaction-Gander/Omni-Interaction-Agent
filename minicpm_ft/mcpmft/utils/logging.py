from __future__ import annotations

import logging
import os
import sys


def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def rank_zero_info(logger: logging.Logger, message: str, *args) -> None:
    if is_rank_zero():
        logger.info(message, *args)

