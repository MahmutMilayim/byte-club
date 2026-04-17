"""
config.py
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
FFMPEG_PATH = r"C:\xampp\htdocs\web-scraper\ffmpeg.exe"
OUTPUT_DIR  = Path("output")
VIDEO_PATH  = "mac.mp4"

# ── Modeller ───────────────────────────────────────────────────────────────
# Top icin yolo26x (kucuk nesne tespitinde daha iyi)
BALL_DETECTOR_WEIGHTS   = "yolo26x.pt"
# Oyuncu icin yolov8x
PLAYER_DETECTOR_WEIGHTS = "yolov8x.pt"

# Geriye donus uyumluluğu icin
DETECTOR_WEIGHTS = "yolov8x.pt"
DETECTOR_CONF    = 0.3
DEVICE           = "cuda"

# ── Hiz ────────────────────────────────────────────────────────────────────
IMGSZ      = 640
FRAME_SKIP = 2

# ── Saha (FIFA) ─────────────────────────────────────────────────────────────
FIELD_W = 105.0
FIELD_H = 68.0

GOAL_Y_MIN       = 29.45
GOAL_Y_MAX       = 38.55
GOAL_X_LEFT_MAX  = 2.5
GOAL_X_RIGHT_MIN = 102.5

# ── Kalibrasyon ─────────────────────────────────────────────────────────────
CALIB_EVERY_N_FRAMES = 30
MANUAL_CALIB_POINTS = {
    "pixel": [[19, 414], [1873, 414], [1917, 1058], [0, 1060]],
    "world": [[0.0, 68.0], [105.0, 68.0], [105.0, 0.0], [0.0, 0.0]],
}

# ── Takim renk egitimi ──────────────────────────────────────────────────────
TEAM_SAMPLE_FRAMES = 120

# ── Sut Tespiti ─────────────────────────────────────────────────────────────
SHOT_MIN_PIXEL_SPEED      = 25
SHOT_MAX_PIXEL_SPEED      = 400
SHOT_MIN_MPS              = 8.0
SHOT_MAX_MPS              = 55.0
SHOT_MAX_DIST_FROM_GOAL_M = 45.0
SHOT_COOLDOWN_FRAMES      = 40
GOAL_CHECK_FRAMES         = 80

# ── Highlight Video ─────────────────────────────────────────────────────────
CLIP_PRE_SEC  = 5.0
CLIP_POST_SEC = 4.0

# ── Intro atlama ────────────────────────────────────────────────────────────
SKIP_FIRST_SECONDS = 60
