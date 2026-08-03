from pathlib import Path
from unittest.mock import patch

import pytest

from video_analyst.env.detect import FFmpegEnvironment
from video_analyst.exceptions import AudioExtractError, MediaProbeError
from video_analyst.media.audio import extract_audio
from video_analyst.media.probe import probe_media
from video_analyst.process import CommandResult


def _env(tmp_path: Path) -> FFmpegEnvironment:
    return FFmpegEnvironment(
        ffmpeg_path=tmp_path / "ffmpeg",
        ffprobe_path=tmp_path / "ffprobe",
        ffmpeg_version="ffmpeg version test",
        ffprobe_version="ffprobe version test",
    )


def test_probe_media_parses_json(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    payload = """
    {
      "format": {
        "format_name": "mov,mp4",
        "format_long_name": "QuickTime / MP4",
        "duration": "10.5",
        "size": "12345",
        "bit_rate": "1000"
      },
      "streams": [
        {
          "index": 0,
          "codec_type": "video",
          "codec_name": "h264",
          "width": 1920,
          "height": 1080,
          "avg_frame_rate": "30/1",
          "r_frame_rate": "30/1",
          "pix_fmt": "yuv420p"
        },
        {
          "index": 1,
          "codec_type": "audio",
          "codec_name": "aac",
          "sample_rate": "48000",
          "channels": 2,
          "channel_layout": "stereo",
          "tags": {"language": "eng"}
        }
      ]
    }
    """

    with patch(
        "video_analyst.media.probe.run_command",
        return_value=CommandResult(["ffprobe"], 0, payload, ""),
    ):
        info = probe_media(video, _env(tmp_path))
        assert info.duration == 10.5
        assert len(info.video_streams) == 1
        assert info.video_streams[0].width == 1920
        assert info.audio_streams[0].language == "eng"


def test_probe_media_failure(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    with patch(
        "video_analyst.media.probe.run_command",
        return_value=CommandResult(["ffprobe"], 1, "", "bad file"),
    ):
        with pytest.raises(MediaProbeError):
            probe_media(video, _env(tmp_path))


def test_extract_audio_success(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    out = tmp_path / "a.mp3"

    def fake_run(args, **kwargs):
        Path(args[-1]).write_bytes(b"mp3data")
        return CommandResult([str(a) for a in args], 0, "", "ok")

    with patch("video_analyst.media.audio.run_command", side_effect=fake_run):
        path = extract_audio(video, out, _env(tmp_path))
        assert path.is_file()


def test_extract_audio_empty_output(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    out = tmp_path / "a.mp3"

    with patch(
        "video_analyst.media.audio.run_command",
        return_value=CommandResult(["ffmpeg"], 0, "", ""),
    ):
        with pytest.raises(AudioExtractError):
            extract_audio(video, out, _env(tmp_path))
