"""Pipeline orchestration for Video Analyst.

This package contains:
- phase1: env → probe → audio → ASR → LLM calibrate → active subs → Phase 2 prep
- epic2: vision coarse analysis → aggregation → refinement plan
- epic3: 5/10/15 FPS fine-frame extraction → contact sheets → fine batches
- epic4: fine temporal vision analysis → temporal events → second refinement plan
"""

from video_analyst.pipeline.epic2 import run_epic2
from video_analyst.pipeline.epic3 import run_epic3
from video_analyst.pipeline.epic4 import run_epic4
from video_analyst.pipeline.phase1 import run_phase1

__all__ = ["run_epic2", "run_epic3", "run_epic4", "run_phase1"]
