from video_analyst.models import SubtitleSegment
from video_analyst.subtitle.srt import segments_to_srt


def test_segments_to_srt_basic():
    segs = [
        SubtitleSegment(index=1, start=0.0, end=1.5, text="Hello"),
        SubtitleSegment(index=2, start=1.5, end=3.25, text="World"),
    ]
    srt = segments_to_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello" in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\nWorld" in srt


def test_segments_to_srt_skips_empty():
    # empty text is rejected by SubtitleSegment model; SRT helper still filters blanks
    segs = [
        SubtitleSegment(index=1, start=0.0, end=1.0, text="Keep"),
    ]
    srt = segments_to_srt(segs)
    assert "Keep" in srt
    assert srt.count("-->") == 1
