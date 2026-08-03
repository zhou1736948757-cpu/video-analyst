from pathlib import Path

from video_analyst.cache.store import CacheStore, OutputLayout
from video_analyst.models import PipelineStage


def test_cache_state_roundtrip(tmp_path: Path):
    layout = OutputLayout.create(tmp_path / "out")
    store = CacheStore(layout)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    state = store.load_state(video)
    state.mark(PipelineStage.ENV_CHECK)
    state.mark(PipelineStage.MEDIA_PROBE)
    store.save_state(state)

    loaded = store.load_state(video)
    assert loaded.has(PipelineStage.MEDIA_PROBE)
    assert not loaded.has(PipelineStage.ASR)
