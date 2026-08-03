#!/usr/bin/env bash
# Phase 1 Whisper / subtitle self-test.
# Usage:
#   export GROQ_API_KEY='gsk_...'
#   ./scripts/run_whisper_test.sh
#
# Optional env:
#   VIDEO   path to video (default: iShot sample)
#   OUT     output dir
#   VENV    python venv path

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
VIDEO="${VIDEO:-/Users/mac/Movies/Videos/iShot_2026-07-20_19.17.45.mp4}"
OUT="${OUT:-/Users/mac/Movies/Videos/va_test_ishot_output}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "ERROR: 请先设置 GROQ_API_KEY，例如："
  echo "  export GROQ_API_KEY='gsk_你的密钥'"
  exit 2
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: 视频不存在: $VIDEO"
  exit 2
fi

echo "=== 1) 首次 analyze（应调用 Groq）==="
# 清掉 ASR 及之后阶段，保留 media/audio 缓存也可；这里用 --no-resume 做干净首次跑
rm -rf "$OUT"
video-analyst analyze "$VIDEO" -o "$OUT"
echo

echo "=== 2) 校验字幕时间戳 ==="
OUT="$OUT" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
media = json.loads((out / "media_info.json").read_text(encoding="utf-8"))
doc = json.loads((out / "subtitles" / "subtitles.json").read_text(encoding="utf-8"))
duration = float(media["duration"])
segs = doc["segments"]
print(f"engine={doc.get('engine')} language={doc.get('language')} segments={len(segs)}")
print(f"video_duration={duration}")
prev_end = -1.0
issues = []
for s in segs:
    if s["start"] > s["end"]:
        issues.append(f"inverted #{s['index']}: {s['start']} > {s['end']}")
    if s["start"] < prev_end - 1e-3:
        issues.append(
            f"overlap/non-monotonic #{s['index']}: start={s['start']} < prev_end={prev_end}"
        )
    if s["end"] > duration + 0.5:
        issues.append(f"past EOF #{s['index']}: end={s['end']} > duration={duration}")
    prev_end = max(prev_end, s["end"])
    print(f"  [{s['index']:02d}] {s['start']:7.2f} → {s['end']:7.2f}  {s['text'][:80]}")
if issues:
    print("TIMESTAMP ISSUES:")
    for i in issues:
        print(" -", i)
    raise SystemExit(1)
print("TIMESTAMP OK")
print("--- SRT head ---")
print((out / "subtitles" / "subtitles.srt").read_text(encoding="utf-8")[:800])
PY
echo

echo "=== 3) 第二次 analyze（应命中缓存，不请求 Groq）==="
ASR_RAW="$OUT/cache/asr_raw.json"
mtime1=$(stat -f %m "$ASR_RAW")
video-analyst analyze "$VIDEO" -o "$OUT" 2>&1 | tee /tmp/va_second_run.log
mtime2=$(stat -f %m "$ASR_RAW")
if grep -q "Resume: ASR raw from cache" /tmp/va_second_run.log; then
  echo "PASS: 日志显示 ASR 来自缓存"
else
  echo "FAIL: 未看到 Resume: ASR raw from cache"
  exit 1
fi
if [[ "$mtime1" == "$mtime2" ]]; then
  echo "PASS: asr_raw.json mtime 未变 ($mtime1)"
else
  echo "FAIL: asr_raw.json 被重写 ($mtime1 -> $mtime2)"
  exit 1
fi
if grep -qi "Calling Groq" /tmp/va_second_run.log; then
  echo "FAIL: 第二次仍调用了 Groq"
  exit 1
else
  echo "PASS: 第二次未调用 Groq"
fi
echo

echo "=== 4) 禁用 GROQ_API_KEY 后的提示（临时去掉 ASR 缓存阶段）==="
OUT="$OUT" python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
p = out / "cache" / "pipeline_state.json"
s = json.loads(p.read_text(encoding="utf-8"))
keep = {"env_check", "media_probe", "audio_extract"}
s["completed_stages"] = [x for x in s.get("completed_stages", []) if x in keep]
s["asr_raw_path"] = None
s["engine"] = None
s["subtitles_json_path"] = None
s["subtitles_srt_path"] = None
asr = out / "cache" / "asr_raw.json"
if asr.exists():
    asr.rename(asr.with_suffix(".json.bak"))
p.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("ASR cache temporarily disabled")
PY

set +e
env -u GROQ_API_KEY video-analyst analyze "$VIDEO" -o "$OUT" 2>&1 | tee /tmp/va_no_key.log
set -e
if grep -q "GROQ_API_KEY not configured" /tmp/va_no_key.log \
   && grep -q "openai-whisper not installed" /tmp/va_no_key.log; then
  echo "PASS: 明确提示 Groq 未配置 + 本地 Whisper 未安装"
else
  echo "FAIL: 提示不完整，见 /tmp/va_no_key.log"
fi

if [[ -f "$OUT/cache/asr_raw.json.bak" ]]; then
  mv "$OUT/cache/asr_raw.json.bak" "$OUT/cache/asr_raw.json"
fi

# Restore successful pipeline state so later resume still works
OUT="$OUT" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

out = Path(os.environ["OUT"])
asr = out / "cache" / "asr_raw.json"
state = {
    "video_path": str(Path(os.environ.get("VIDEO", "/Users/mac/Movies/Videos/iShot_2026-07-20_19.17.45.mp4"))),
    "output_dir": str(out),
    "completed_stages": [
        "env_check",
        "media_probe",
        "audio_extract",
        "asr",
        "subtitles",
        "done",
    ],
    "media_info_path": str(out / "media_info.json"),
    "audio_path": str(out / "audio" / "audio.mp3"),
    "asr_raw_path": str(asr) if asr.is_file() else None,
    "subtitles_json_path": str(out / "subtitles" / "subtitles.json"),
    "subtitles_srt_path": str(out / "subtitles" / "subtitles.srt"),
    "engine": "groq-whisper-large-v3-turbo",
    "last_error": None,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
(out / "cache" / "pipeline_state.json").write_text(
    json.dumps(state, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("pipeline_state restored to DONE")
PY

echo
echo "=== 完成 ==="
echo "字幕 JSON: $OUT/subtitles/subtitles.json"
echo "字幕 SRT:  $OUT/subtitles/subtitles.srt"
echo "可用 IINA / VLC / QuickTime 打开视频并加载 SRT 检查同步。"
echo "识别质量请人工对照语音听一遍中/日/英与专有名词。"
