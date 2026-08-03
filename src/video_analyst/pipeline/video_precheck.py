"""输入视频预检（修复 Bug 3）。

在任何 Dry Run 或 Pipeline 规划前验证：

* 输入路径存在
* 输入路径是普通文件（不是目录、不是设备）
* ffprobe 可读取
* 至少存在一个视频流
* 视频时长 > 0

异常：

* VideoInputNotFoundError：路径不存在
* VideoInputIsDirectoryError：路径是目录
* VideoInputEmptyError：文件大小为 0
* VideoProbeError：ffprobe 失败
* VideoStreamMissingError：无视频流
* InvalidVideoDurationError：duration ≤ 0

不在这里重新实现 ffprobe 解析；复用 ``media.probe`` 已有的探测能力并增强错误。
"""

from __future__ import annotations

from pathlib import Path

from video_analyst.env.detect import FFmpegEnvironment, detect_ffmpeg_environment
from video_analyst.exceptions import MediaProbeError
from video_analyst.media.probe import probe_media


class VideoInputNotFoundError(FileNotFoundError):
    """输入视频文件不存在。"""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Video file not found: {path}")
        self.path = Path(path)


class VideoInputIsDirectoryError(IsADirectoryError):
    """输入路径是目录而不是普通文件。"""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Video path is a directory, not a file: {path}")
        self.path = Path(path)


class VideoInputEmptyError(ValueError):
    """输入文件大小为 0。"""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Video file is empty (0 bytes): {path}")
        self.path = Path(path)


class VideoStreamMissingError(ValueError):
    """媒体文件不包含视频流。"""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Video file has no video stream: {path}")
        self.path = Path(path)


class InvalidVideoDurationError(ValueError):
    """视频时长 ≤ 0（可能是空或损坏）。"""

    def __init__(self, path: Path, duration: float | None) -> None:
        super().__init__(
            f"Video has non-positive duration ({duration}): {path}"
        )
        self.path = Path(path)
        self.duration = duration


def validate_video_input(
    video_path: Path | str,
    env: FFmpegEnvironment | None = None,
) -> Path:
    """预检输入视频。返回 resolved absolute Path。

    抛出：

    * VideoInputNotFoundError
    * VideoInputIsDirectoryError
    * VideoInputEmptyError
    * VideoProbeError
    * VideoStreamMissingError
    * InvalidVideoDurationError
    """
    path = Path(video_path).expanduser()
    if not path.exists():
        raise VideoInputNotFoundError(path)
    if path.is_dir():
        raise VideoInputIsDirectoryError(path)
    if not path.is_file():
        # 设备 / socket / 命名管道等
        raise VideoInputNotFoundError(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise VideoInputNotFoundError(path) from exc
    if size == 0:
        raise VideoInputEmptyError(path)

    env = env or detect_ffmpeg_environment()
    try:
        info = probe_media(path, env)
    except MediaProbeError as exc:
        # 重新抛出，保持类型一致
        raise VideoProbeError(str(exc)) from exc

    if not info.video_streams:
        raise VideoStreamMissingError(path)
    duration = info.duration
    if duration is None or duration <= 0:
        raise InvalidVideoDurationError(path, duration)

    return path.resolve()


__all__ = [
    "InvalidVideoDurationError",
    "VideoInputEmptyError",
    "VideoInputIsDirectoryError",
    "VideoInputNotFoundError",
    "VideoStreamMissingError",
    "validate_video_input",
]