"""
utils/highlight_maker.py
"""
from __future__ import annotations
import subprocess
import tempfile
import json
from pathlib import Path
import logging
import config as cfg

log = logging.getLogger(__name__)


def make_highlights(
    video_path: str,
    events: list[dict],
    out_path: str,
    pre_sec: float  = None,
    post_sec: float = None,
) -> str:
    pre_sec  = pre_sec  if pre_sec  is not None else cfg.CLIP_PRE_SEC
    post_sec = post_sec if post_sec is not None else cfg.CLIP_POST_SEC
    ffmpeg   = cfg.FFMPEG_PATH

    if not Path(ffmpeg).exists():
        raise FileNotFoundError(f"FFmpeg bulunamadi: {ffmpeg}")

    if not events:
        log.warning("  Highlight icin olay yok.")
        return ""

    duration = _duration(ffmpeg, video_path)

    intervals = []
    for ev in events:
        t     = ev["timeSeconds"]
        start = max(0.0, t - pre_sec)
        end   = min(duration, t + post_sec)
        intervals.append((start, end))
    intervals = _merge(intervals)

    log.info(f"  {len(intervals)} klip kesilecek...")

    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, (s, e) in enumerate(intervals):
            out = Path(tmp) / f"c{i:04d}.mp4"
            r = subprocess.run([
                ffmpeg, "-y",
                "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                "-i", str(video_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k",
                str(out),
            ], capture_output=True, text=True)
            if r.returncode == 0:
                clips.append(out)
                log.info(f"  ✂  {i+1}/{len(intervals)}: {s:.1f}s-{e:.1f}s")
            else:
                log.warning(f"  Klip {i} basarisiz")

        if not clips:
            return ""

        concat = Path(tmp) / "list.txt"
        concat.write_text("\n".join(f"file '{c}'" for c in clips), encoding="utf-8")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-c", "copy", str(out_path),
        ], capture_output=True, text=True)

        if r.returncode != 0:
            raise RuntimeError(f"Concat hatasi:\n{r.stderr[-300:]}")

    log.info(f"  Highlight: {out_path}")
    return out_path


def _duration(ffmpeg: str, video: str) -> float:
    ff = Path(ffmpeg)
    probe = ff.parent / "ffprobe.exe"
    if not probe.exists():
        probe = ff.parent / "ffprobe"
    try:
        r = subprocess.run([
            str(probe), "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video),
        ], capture_output=True, text=True, timeout=30)
        for s in json.loads(r.stdout).get("streams", []):
            if "duration" in s:
                return float(s["duration"])
    except Exception:
        pass
    return 9999.0


def _merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1] + 1.0:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]
