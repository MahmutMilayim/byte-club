"""
Football Analytics Pipeline — Merkezi Yol ve Parametre Referansı
Tüm path sabitlerini ve temel ayarları buradan takip edebilirsin.
Bu dosya script'ler tarafından import edilmez; referans ve dokümantasyon amaçlıdır.
"""

# ─── Giriş / Çıkış ────────────────────────────────────────────────────────────
INPUT_VIDEO         = "/input/input.mp4"
INPUT_VIDEO_SAMPLE  = "/input/input_10s.mp4"   # hızlı test için
OUTPUT_DIR          = "/output"

# ─── Modeller ─────────────────────────────────────────────────────────────────
YOLO_MODEL          = "/work/models/yolo26x.pt"
PNLCALIB_DIR        = "/work/PnLCalib"
PNLCALIB_WEIGHTS_KP = "/work/PnLCalib/weights/SV_kp"
PNLCALIB_WEIGHTS_LN = "/work/PnLCalib/weights/SV_lines"

# ─── Stage 2: Person Tracking ─────────────────────────────────────────────────
STAGE2_JSON         = "/output/stage2_tracking/tracking.json"
STAGE2_VIDEO        = "/output/stage2_tracking/tracking.mp4"

# ─── Stage 3: Gameplay Filter ─────────────────────────────────────────────────
STAGE3_GAMEPLAY_JSON  = "/output/stage3_filter/gameplay.json"
STAGE3_GAMEPLAY_VIDEO = "/output/stage3_filter/gameplay.mp4"
STAGE3_REPORT_JSON    = "/output/stage3_filter/filter_report.json"

# ─── Stage 4: Team Clustering ─────────────────────────────────────────────────
STAGE4_TRACK_LABELS          = "/output/stage4_clustering/track_labels.json"
STAGE4_TRACK_LABELS_CORRECTED = "/output/stage4_clustering/track_labels_corrected.json"
STAGE4_CROP_MANIFEST         = "/output/stage4_clustering/crop_manifest.json"
STAGE4_CROPS_DIR             = "/output/stage4_clustering/crops"

# ─── Stage 5: Ball Tracking (YOLO) ────────────────────────────────────────────
STAGE5_BALL_JSON    = "/output/stage5_ball/ball_tracks.json"
STAGE5_BALL_VIDEO   = "/output/stage5_ball/ball_review.mp4"

# ─── Stage 6: Possession ──────────────────────────────────────────────────────
STAGE7_POSSESSION_JSON = "/output/stage7_possession/possession.json"

# ─── Stage 7: Field Projection ────────────────────────────────────────────────
STAGE7_KEYFRAMES_DIR     = "/output/stage6_field/keyframes"
STAGE7_KEYFRAME_JSON_DIR = "/output/stage6_field/keyframe_json"
STAGE7_BANK_V1           = "/output/stage6_field/homography_bank_raw.json"
STAGE7_BANK_V2           = "/output/stage6_field/homography_bank.json"

STAGE7_HOMOGRAPHY_BASIC   = "/output/stage6_field/homography_map_basic.json"
STAGE7_HOMOGRAPHY_MOTION  = "/output/stage6_field/homography_map.json"
STAGE7_HOMOGRAPHY_REFINED = "/output/stage6_field/homography_map_refined.json"

STAGE7_PROJECTION_JSON    = "/output/stage6_field/projection.json"            # unified Stage 6 output
STAGE7_PROJECTION_SMOOTHED = "/output/stage6_field/projection_smoothed.json"  # legacy alias kept for compatibility

# ─── Final ────────────────────────────────────────────────────────────────────
FINAL_VIDEO   = "/output/final/overlay.mp4"
FINAL_SUMMARY = "/output/final/summary.json"

# ─── Saha Boyutları ───────────────────────────────────────────────────────────
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M  = 68.0
