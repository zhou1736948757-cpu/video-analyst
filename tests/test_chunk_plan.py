from video_analyst.media.chunk_plan import (
    AudioChunkPlan,
    _merge_short_plans,
    _split_with_overlap,
)
from video_analyst.media.silence import TimeRange


def test_split_long_range_with_overlap():
    r = TimeRange(0.0, 100.0)
    parts = _split_with_overlap(r, max_seconds=50.0, overlap=0.5)
    assert len(parts) >= 2
    assert parts[0].start == 0.0
    assert parts[-1].end == 100.0
    # overlap: next start before previous end
    assert parts[1].start < parts[0].end


def test_merge_short_plans():
    plans = [
        AudioChunkPlan("c1", 0.0, 1.0, 0.0, 0.0),
        AudioChunkPlan("c2", 1.0, 2.0, 0.0, 0.0),
        AudioChunkPlan("c3", 10.0, 20.0, 0.0, 0.0),
    ]
    out = _merge_short_plans(plans, minimum=3.0, duration=30.0)
    # short c1+c2 merge; long c3 may absorb if previous still short after merge
    assert out[0].absolute_end >= 2.0
    assert out[-1].absolute_end == 20.0
    assert len(out) >= 1
