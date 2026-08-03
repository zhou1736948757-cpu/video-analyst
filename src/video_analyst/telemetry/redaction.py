"""Redaction helpers for telemetry (Epic 6).

确保 API Key、Authorization Header、完整 base64 图片不进入遥测记录。
"""

from __future__ import annotations

import re

_SECRET_PATTERNS = (
    re.compile(r"(sk-[A-Za-z0-9]{10,})"),
    re.compile(r"(gsk_[A-Za-z0-9]{10,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9\-\._~+/]+=*)", re.I),
    re.compile(r"(api[_-]?key\s*[:=]\s*)(\S+)", re.I),
    re.compile(r"(authorization\s*[:=]\s*)(\S+)", re.I),
)


def redact_secrets(text: str) -> str:
    """从文本中移除常见密钥模式。返回脱敏后文本。"""
    if not text:
        return text
    redacted = text
    for pat in _SECRET_PATTERNS:
        if pat.groups >= 2:
            redacted = pat.sub(r"\1***REDACTED***", redacted)
        else:
            redacted = pat.sub("***REDACTED***", redacted)
    return redacted


def is_safe_for_telemetry(value: str) -> bool:
    """检查字符串是否可安全写入遥测（不含密钥模式）。"""
    if not value:
        return True
    for pat in _SECRET_PATTERNS:
        if pat.search(value):
            return False
    return True


__all__ = ["is_safe_for_telemetry", "redact_secrets"]