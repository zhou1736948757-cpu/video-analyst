"""Content hashing helpers for cache reproducibility (no secrets)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, max_bytes: int | None = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if max_bytes is None:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            remaining = max_bytes
            while remaining > 0:
                block = fh.read(min(1024 * 1024, remaining))
                if not block:
                    break
                h.update(block)
                remaining -= len(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- Image fingerprint (path + size + sha256; mtime is a fast pre-filter) ----


class ImageFingerprint:
    """Stable fingerprint for an on-disk image.

    The fingerprint includes:
      - relative_path  (so different layouts don't collide)
      - size_bytes     (cheap pre-check)
      - sha256         (content hash; full correctness)
      - mtime_ns       (optional fast pre-filter; not authoritative)

    Use `ImageFingerprint.from_path()` then `.fast_key()` to gate expensive
    SHA scans. Compare `full_fingerprint()` for authoritative equality.
    """

    __slots__ = ("relative_path", "size_bytes", "sha256", "mtime_ns")

    def __init__(
        self,
        relative_path: str,
        size_bytes: int,
        sha256: str,
        mtime_ns: int | None = None,
    ) -> None:
        self.relative_path = relative_path
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.mtime_ns = mtime_ns

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        root: Path | None = None,
        max_bytes: int | None = 16 * 1024 * 1024,
    ) -> "ImageFingerprint":
        rel = str(path.resolve())
        if root is not None:
            try:
                rel = str(path.resolve().relative_to(root.resolve()))
            except ValueError:
                rel = str(path.resolve())
        st = path.stat()
        size = int(st.st_size)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        sha = sha256_file(path, max_bytes=max_bytes)
        return cls(relative_path=rel, size_bytes=size, sha256=sha, mtime_ns=mtime_ns)

    def fast_key(self) -> tuple[str, int, int | None]:
        """Cheap key: path + size + mtime_ns. Identical images on disk with
        unchanged mtime yield the same fast_key."""
        return (self.relative_path, self.size_bytes, self.mtime_ns)

    def full_fingerprint(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "mtime_ns": self.mtime_ns,
        }

    def model_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "mtime_ns": self.mtime_ns,
        }


def image_fingerprint_cache_key(path: Path, root: Path | None = None) -> str:
    """Stable cache key for an image content fingerprint.

    Note: this actually reads the file. Use `image_fast_key()` first to
    short-circuit when only mtime/size change detection is needed.
    """
    return sha256_json(ImageFingerprint.from_path(path, root=root).full_fingerprint())


def image_fast_key(path: Path, root: Path | None = None) -> str:
    """Cheap key (path + size + mtime). NOT content-aware."""
    st = path.stat()
    rel = str(path.resolve())
    if root is not None:
        try:
            rel = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    return sha256_json({"rel": rel, "size": st.st_size, "mtime_ns": mtime_ns})


__all__ = [
    "ImageFingerprint",
    "image_fast_key",
    "image_fingerprint_cache_key",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
]