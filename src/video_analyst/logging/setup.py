"""Logging helpers — never log secrets."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_SECRET_PATTERNS = (
    re.compile(r"(api[_-]?key\s*[:=]\s*)(\S+)", re.I),
    re.compile(r"(authorization\s*[:=]\s*bearer\s+)(\S+)", re.I),
    re.compile(r"(gsk_[A-Za-z0-9]+)"),
)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = msg
        for pat in _SECRET_PATTERNS:
            if pat.groups == 1:
                redacted = pat.sub("***REDACTED***", redacted)
            else:
                redacted = pat.sub(r"\1***REDACTED***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("video_analyst")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = RedactingFilter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redactor)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(redactor)
        logger.addHandler(fh)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"video_analyst.{name}")
    return logging.getLogger("video_analyst")
