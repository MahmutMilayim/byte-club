"""
Stage 4 — Team Clustering (chunked)

Videoyu N_CHUNKS parçaya böler, her parçada forma rengi özelliklerine
dayalı KMeans kümeleme yaparak track'lere team_1 / team_2 / referee etiketi atar,
sonuçları overlap-based etiket hizalamasıyla birleştirir. Birleştirmeden sonra
yalnızca 'referee' kümesindeki, kale önünde yoğunlaşan izler yarı+kale tarafına
göre takıma çevrilebilir (açık renkli kaleci ↔ hakem karışıklığı).

Giriş : /output/stage3_filter/gameplay.mp4 + gameplay.json
Çıkış : /output/stage4_clustering/track_labels.json
         /output/stage4_clustering/crop_manifest.json
         /output/stage4_clustering/crops/
         /output/stage4_clustering/review.mp4  (birleştirme + kaleci düzeltmesi sonrası tek geçiş)

Oyuncu takımları yalnızca forma rengi (KMeans) ile atanır. Saha konumu + yarı bilgisi
yalnızca 'referee' kümesindeki izlere uygulanır (açık renkli kaleci ↔ hakem karışıklığı).
"""

import glob
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# ─── DOSYA YOLLARI ────────────────────────────────────────────────────────────
SRC_VIDEO   = Path("/output/stage3_filter/gameplay.mp4")
SRC_JSON    = Path("/output/stage3_filter/gameplay.json")
WORKDIR     = Path("/output/stage4_clustering/_chunks")
FINAL_OUT   = Path("/output/stage4_clustering")

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
N_CHUNKS            = 3   # bellek fix sonrası 5'ten düşürüldü; label tutarlılığı için ≥2 önerilir
VERBOSE             = True
LOG_EVERY_N_FRAMES  = 50

MIN_TRACK_SAMPLES      = 6
MAX_SAMPLES_PER_TRACK  = 20
MIN_BBOX_HEIGHT        = 45
MIN_BBOX_WIDTH         = 18

TORSO_TOP_RATIO        = 0.18
TORSO_BOTTOM_RATIO     = 0.62
TORSO_SIDE_MARGIN_RATIO = 0.18

GREEN_LOWER = np.array([25, 35, 35],  dtype=np.uint8)
GREEN_UPPER = np.array([95, 255, 255], dtype=np.uint8)

# ── Forma rengi HSV sabitleri ──────────────────────────────────────────────
# Galatasaray  : turuncu-kırmızı forma, beyaz şort
#   → HSV Hue  ≈ 0-25 (turuncu-kırmızı) ve 160-180 (kırmızı wrap)
#   → yüksek satürasyon (canlı renk)
# Trabzonspor  : gri-siyah forma
#   → düşük satürasyon, orta-düşük value
# Hakem        : açık mavi forma
#   → HSV Hue ≈ 95-135

DARK_VALUE_THRESHOLD = 60    # V < bu → "koyu/siyah"
GRAY_SAT_MAX         = 55    # S < bu → "gri" (Trabzonspor gri-siyah)
GRAY_VAL_MIN         = 55    # V aralığı: siyahtan ayır
GRAY_VAL_MAX         = 210   # V aralığı: beyazdan ayır
BLUE_LOW,  BLUE_HIGH = 95, 135     # HSV H: açık mavi (hakem)
ORANGE_LOW_1, ORANGE_HIGH_1 = 0,  25    # HSV H: turuncu-kırmızı (düşük)
ORANGE_LOW_2, ORANGE_HIGH_2 = 160, 180  # HSV H: kırmızı (yüksek wrap)
ORANGE_MIN_SAT = 80    # Turuncu için min satürasyon (ten rengi/çim karışmasını önle)

# ── Küme isimlendirme eşikleri (bilinen forma renklerine göre) ─────────────
REFEREE_BLUE_MIN  = 0.06   # Hakem: açık mavi oranı en az bu kadar
TEAM1_ORANGE_MIN  = 0.07   # Team 1 (Galatasaray): turuncu-kırmızı oranı en az bu kadar

# Chunk overlap — sınırlardaki track etiket tutarlılığı için
CHUNK_OVERLAP_RATIO = 0.08  # Her chunk %8 overlap alır

# ── Yarı bilgisi (kaleci kale tarafı ataması için) ─────────────────────────
# 1. Yarı: Team 1 (Galatasaray) kalesi SOLDA, Team 2 (Trabzonspor) kalesi SAĞDA
# 2. Yarı: Taraflar tersine döner
# gameplay.mp4 frame sayısının bu kadarı 1. yarıya aittir (yaklaşık):
FIRST_HALF_END_FRAC = 0.50

# ── Kaleci kale bölgesi (normalize frame koordinatları) ──────────────────
GK_ZONE_LEFT_X_MAX   = 0.28   # sol kale önü sağ sınırı
GK_ZONE_RIGHT_X_MIN  = 0.72   # sağ kale önü sol sınırı
GK_ZONE_Y_MIN        = 0.12
GK_ZONE_Y_MAX        = 0.95
GK_CENTER_LO_X       = 0.36   # orta saha sol
GK_CENTER_HI_X       = 0.64   # orta saha sağ
GK_CENTER_MAX_FRAC   = 0.35   # track'in bu kadarından fazlası orta sahadadaysa kaleci değil
GK_ZONE_RATIO_MIN    = 0.48   # track'in en az bu kadarı kale bölgesinde olmalı
GK_MIN_SAMPLES       = 18     # en az bu kadar pozisyon örneği olmalı

# Hakem: bbox merkezlerinin bu kadarı yatay orta bantta ise sahaya göre takıma çevirme
REFEREE_CENTER_KEEP_MIN = 0.38

# Eski GK-ref parametreleri (geriye dönük uyumluluk için korunuyor)
GK_REF_LEFT_X_MAX_FRAC      = GK_ZONE_LEFT_X_MAX
GK_REF_RIGHT_X_MIN_FRAC     = GK_ZONE_RIGHT_X_MIN
GK_REF_Y_MIN_FRAC           = GK_ZONE_Y_MIN
GK_REF_Y_MAX_FRAC           = GK_ZONE_Y_MAX
GK_REF_CENTER_LO_FRAC       = GK_CENTER_LO_X
GK_REF_CENTER_HI_FRAC       = GK_CENTER_HI_X
GK_REF_CENTER_STRIP_MAX_FRAC = GK_CENTER_MAX_FRAC
GK_REF_ZONE_RATIO_MIN       = GK_ZONE_RATIO_MIN
GK_REF_MIN_SAMPLES          = GK_MIN_SAMPLES
GK_REF_NEIGHBOR_FRAC        = 0.17
GK_REF_VOTE_MIN             = 10
GK_REF_VOTE_MARGIN          = 0.54

LABEL_COLORS = {
    "team_1":  (0, 165, 255),
    "team_2":  (40,  40,  40),
    "referee": (255, 255,   0),
    "unknown": (0, 255,   0),
}

# ─── YARDIMCI: JSON ───────────────────────────────────────────────────────────
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ─── YARDIMCI: GÖRÜNTÜ ────────────────────────────────────────────────────────
def safe_int(v):
    return int(round(float(v)))

def clip_bbox(x1, y1, x2, y2, w, h):
    return (max(0, min(w-1, x1)), max(0, min(h-1, y1)),
            max(0, min(w-1, x2)), max(0, min(h-1, y2)))

def extract_torso_patch(frame, bbox_xyxy):
    x1, y1, x2, y2 = map(safe_int, bbox_xyxy)
    x1, y1, x2, y2 = clip_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_BBOX_WIDTH or bh < MIN_BBOX_HEIGHT:
        return None, None

    tx1 = x1 + int(bw * TORSO_SIDE_MARGIN_RATIO)
    tx2 = x2 - int(bw * TORSO_SIDE_MARGIN_RATIO)
    ty1 = y1 + int(bh * TORSO_TOP_RATIO)
    ty2 = y1 + int(bh * TORSO_BOTTOM_RATIO)
    tx1, ty1, tx2, ty2 = clip_bbox(tx1, ty1, tx2, ty2, frame.shape[1], frame.shape[0])
    if tx2 <= tx1 or ty2 <= ty1:
        return None, None
    return frame[ty1:ty2, tx1:tx2].copy(), [tx1, ty1, tx2, ty2]

def compute_feature(patch_bgr):
    """HSV renk uzayında özellik çıkarımı.
    Galatasaray (turuncu-kırmızı), Trabzonspor (gri-siyah), Hakem (açık mavi)
    için özelleştirilmiş özellik vektörü.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return None, None

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    mask_non_green = cv2.bitwise_not(cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER))
    px = hsv[mask_non_green > 0]
    if px.shape[0] < 20:
        px = hsv.reshape(-1, 3)

    h, s, v = px[:, 0], px[:, 1], px[:, 2]
    mh = float(np.mean(h))
    ms = float(np.mean(s))
    mv = float(np.mean(v))

    # Karanlık piksel oranı (siyah/koyu → Trabzonspor siyah bölgesi)
    dark   = float(np.mean(v < DARK_VALUE_THRESHOLD))

    # Gri piksel oranı: düşük satürasyon + orta value → Trabzonspor gri forması
    gray   = float(np.mean(
        (s <= GRAY_SAT_MAX) & (v >= GRAY_VAL_MIN) & (v <= GRAY_VAL_MAX)
    ))

    # Açık mavi oranı → Hakem forması
    blue   = float(np.mean((h >= BLUE_LOW) & (h <= BLUE_HIGH)))

    # Turuncu-kırmızı oranı (satürasyon filtreli) → Galatasaray forması
    orange = float(np.mean(
        (((h >= ORANGE_LOW_1) & (h <= ORANGE_HIGH_1)) |
         ((h >= ORANGE_LOW_2) & (h <= ORANGE_HIGH_2))) &
        (s >= ORANGE_MIN_SAT)
    ))

    feat  = np.array([mh/179, ms/255, mv/255, dark, gray, blue, orange],
                     dtype=np.float32)
    stats = {
        "mean_h":      round(mh,    4),
        "mean_s":      round(ms,    4),
        "mean_v":      round(mv,    4),
        "dark_ratio":  round(dark,  4),
        "gray_ratio":  round(gray,  4),
        "blue_ratio":  round(blue,  4),
        "orange_ratio":round(orange,4),
    }
    return feat, stats

# ─── YARDIMCI: VİDEO ─────────────────────────────────────────────────────────
def split_video_exact(src_video, chunk_ranges, out_dir):
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {src_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writers = []
    for i, (s, e) in enumerate(chunk_ranges, 1):
        out_path = out_dir / f"chunk_{i:02d}.mp4"
        wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError(f"Writer açılamadı: {out_path}")
        writers.append((s, e, out_path, wr))

    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0
    active = set(range(len(writers)))
    with tqdm(total=total_vid, desc="Stage 4 | Split video ", unit="frame", ncols=90) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for idx in list(active):
                s, e, _, wr = writers[idx]
                if s <= frame_idx < e:
                    wr.write(frame)
                elif frame_idx >= e:
                    active.discard(idx)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    for _, _, _, wr in writers:
        wr.release()

    return [wp for (_, _, wp, _) in writers]

def concat_videos(video_paths, out_path):
    valid = [p for p in video_paths if p is not None and Path(p).exists()]
    if not valid:
        return
    list_file = Path(out_path).parent / "_concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vp in valid:
            f.write(f"file '{Path(vp).as_posix()}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(out_path)],
        check=True
    )


def render_review_video_merged_labels(video_path, frames, merged_tracks, out_path, fps_hint=None):
    """
    Tam gameplay videosu üzerinde birleştirilmiş track etiketleriyle (reclassify sonrası)
    review.mp4 üretir. merged_tracks: track_id (str) → { "label": ... }.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Review (final): video açılamadı: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or (float(fps_hint) if fps_hint else 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not rw.isOpened():
        cap.release()
        raise RuntimeError(f"Review (final): yazılamadı: {out_path}")

    def tid_label(tid):
        s = str(tid)
        rec = merged_tracks.get(s)
        if rec is None and tid in merged_tracks:
            rec = merged_tracks[tid]
        if not rec:
            return "unknown"
        return rec.get("label", "unknown")

    njson = len(frames)
    vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_hint = max(njson, vf) if vf > 0 else njson
    fidx = 0

    with tqdm(total=max(total_hint, 1), desc="Stage 4 | Review video (merged + GK fix)",
              unit="frame", ncols=90) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx < njson:
                for obj in frames[fidx].get("objects", []):
                    tid = int(obj["track_id"])
                    x1, y1, x2, y2 = map(safe_int, obj["bbox_xyxy"])
                    label = tid_label(tid)
                    color = LABEL_COLORS.get(label, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"ID {tid} | {label}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(
                        frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1
                    )
                    cv2.putText(
                        frame, text, (x1 + 4, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA,
                    )
            cv2.putText(
                frame, "Stage 4 — corrected labels", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
            )
            rw.write(frame)
            fidx += 1
            pbar.update(1)

    cap.release()
    rw.release()


def copy_crops(crop_dirs, final_crops):
    final_crops.mkdir(parents=True, exist_ok=True)
    for cd in crop_dirs:
        if cd is None or not Path(cd).exists():
            continue
        for fp in Path(cd).rglob("*"):
            if fp.is_file():
                dst = final_crops / fp.name
                if not dst.exists():
                    shutil.copy2(fp, dst)

# ─── YARDIMCI: ETİKET BİRLEŞTİRME ───────────────────────────────────────────
def normalize_track_labels(data):
    if isinstance(data, dict) and "tracks" in data:
        return data["tracks"], data
    if isinstance(data, dict):
        return data, {"tracks": data}
    raise RuntimeError("Beklenmeyen track_labels formatı")

def swap_teams(tracks):
    out = {}
    for k, v in tracks.items():
        vv = dict(v)
        if vv.get("label") == "team_1":
            vv["label"] = "team_2"
        elif vv.get("label") == "team_2":
            vv["label"] = "team_1"
        out[k] = vv
    return out

def align_chunk_to_merged(merged, chunk):
    overlap = set(merged) & set(chunk)
    same = swap = 0
    for tid in overlap:
        lm, lc = merged[tid].get("label"), chunk[tid].get("label")
        if lm not in ("team_1","team_2") or lc not in ("team_1","team_2"):
            continue
        if lm == lc:
            same += 1
        else:
            swap += 1
    if swap > same:
        return swap_teams(chunk), True
    return chunk, False

def merge_manifests(manifest_paths, out_path):
    merged = None
    for mp in manifest_paths:
        if mp is None or not Path(mp).exists():
            continue
        data = load_json(mp)
        if merged is None:
            merged = data
        elif isinstance(merged, list) and isinstance(data, list):
            merged.extend(data)
        elif isinstance(merged, dict) and isinstance(data, dict):
            for key in ("frames", "items"):
                if key in merged and key in data and isinstance(merged[key], list):
                    merged[key].extend(data[key])
    if merged is not None:
        save_json(out_path, merged)


def detect_half_boundary(video_path, gameplay_json_path):
    """
    Yarı geçiş frame'ini (input.mp4 frame indeksi) tespit eder.

    Yöntem 1 — OCR (birincil):
      gameplay.mp4'ü örnekler, sol üst köşedeki sayacı okur.
      1. yarıda dakika sürekli artar (0→45+).  Stage 3 devre arasını
      kesiyor, bu yüzden gameplay.mp4'te geçiş şöyle görünür:
        "45+X:ss" → "45:00" veya "46:00"  (küçük sıfırlama)
      İlk büyük negatif sıçrama bu geçiş noktasıdır.

    Yöntem 2 — Segment boşluğu (yedek):
      gameplay.json'daki en uzun segment aralığı devre arası ile
      örtüşür; o aralığın başındaki input.mp4 frame'i döner.

    gameplay.mp4 → input.mp4 frame dönüşümü için gameplay.json kullanılır.
    """
    # ── gameplay.json oku ─────────────────────────────────────────────────
    try:
        with open(gameplay_json_path) as f:
            gp_data = json.load(f)
        segments = gp_data.get("segments", [])
    except Exception:
        segments = []

    def _seg_start(s):
        return s.get("start_frame", s.get("start", 0))

    def _seg_end(s):
        return s.get("end_frame", s.get("end", 0))

    def _gameplay_frame_to_input(gp_frame_idx):
        """gameplay.mp4 frame sıra no → input.mp4 frame indeksi."""
        offset = 0
        for seg in segments:
            seg_len = _seg_end(seg) - _seg_start(seg)
            if offset + seg_len > gp_frame_idx:
                return _seg_start(seg) + (gp_frame_idx - offset)
            offset += seg_len
        return None

    def _gap_fallback():
        """En büyük segment aralığını devre arası say."""
        if len(segments) < 2:
            return None
        gaps = sorted(
            [(_seg_start(segments[i]) - _seg_end(segments[i - 1]),
              _seg_end(segments[i - 1]))
             for i in range(1, len(segments))],
            reverse=True,
        )
        return gaps[0][1]  # ilk yarı son frame

    # ── OCR denemesi ──────────────────────────────────────────────────────
    try:
        import pytesseract
    except ImportError:
        pytesseract = None

    if pytesseract is None:
        result = _gap_fallback()
        if result and VERBOSE:
            print(f"  [yarı sınırı] pytesseract yok → segment boşluğu: frame {result}")
        return result

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return _gap_fallback()

    # ~10 saniyede bir örnekle
    step = max(1, int(fps * 10))
    ocr_cfg = "--psm 7 -c tessedit_char_whitelist=0123456789:+"
    samples = []  # (gameplay_frame_idx, minutes, seconds)

    for gp_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, gp_idx)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # Sol üst köşe: genişliğin %13'ü, yüksekliğin %8'i
        roi = frame[:max(1, int(h * 0.08)), :max(1, int(w * 0.13))]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Koyu arka plan üzerinde beyaz/sarı metin için eşikleme
        _, thr = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        thr = cv2.resize(thr, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(thr, config=ocr_cfg).strip()
        m = re.search(r"(\d{1,3})\+?:(\d{2})", text)
        if m:
            samples.append((gp_idx, int(m.group(1)), int(m.group(2))))

    cap.release()

    if len(samples) < 6:
        result = _gap_fallback()
        if VERBOSE:
            print(f"  [yarı sınırı] OCR yetersiz ({len(samples)} örnek) "
                  f"→ segment boşluğu: frame {result}")
        return result

    # İlk büyük negatif sıçramayı bul (45+X → 45:00)
    half_gp_frame = None
    for i in range(1, len(samples)):
        cur_gp, cur_min, cur_sec  = samples[i]
        prv_gp, prv_min, prv_sec  = samples[i - 1]
        delta = (cur_min * 60 + cur_sec) - (prv_min * 60 + prv_sec)
        if delta < -30:          # 30 saniyeden fazla geri giderse → geçiş
            half_gp_frame = cur_gp
            break

    if half_gp_frame is None:
        result = _gap_fallback()
        if VERBOSE:
            print(f"  [yarı sınırı] OCR geçiş bulunamadı → segment boşluğu: frame {result}")
        return result

    # gameplay frame → input.mp4 frame
    result = _gameplay_frame_to_input(half_gp_frame)
    if result is None:
        result = _gap_fallback()

    if VERBOSE:
        prv = samples[samples.index(next(s for s in samples if s[0] >= half_gp_frame)) - 1]
        cur = next(s for s in samples if s[0] >= half_gp_frame)
        print(f"  [yarı sınırı] OCR geçiş bulundu: "
              f"{prv[1]}:{prv[2]:02d} → {cur[1]}:{cur[2]:02d}  "
              f"(gameplay frame {half_gp_frame} → input frame {result})")
    return result


def reclassify_referee_goalkeepers(merged_tracks, frames, W, H,
                                    half_boundary=None):
    """
    Yalnızca KMeans'in 'referee' (açık mavi / açık renk) kümesine verdiği izleri
    ele alır: kale önünde yoğunlaşanları yarı + kale tarafına göre team_1 / team_2
    yapar (beyaz/açık yeşil kaleci forması hakem kümesine düşmüş olabilir).

    team_1 / team_2 oyuncularına sahaya göre müdahale edilmez (kamera yarı saha
    gösterdiğinde yanlış düzeltmeleri önlemek için).

    Çoğunlukla görüntü orta bandında kalan \"referee\" izleri gerçek hakem
    sayılır ve takıma çevrilmez.
    """
    LEFT_X  = GK_ZONE_LEFT_X_MAX  * W
    RIGHT_X = GK_ZONE_RIGHT_X_MIN * W
    Y_TOP   = GK_ZONE_Y_MIN       * H
    Y_BOT   = GK_ZONE_Y_MAX       * H
    CXL     = GK_CENTER_LO_X      * W
    CXR     = GK_CENTER_HI_X      * W

    # Yarı sınırı: parametre gelmediyse toplam frame'den tahmin et
    if half_boundary is None:
        total_frames = max((fr.get("frame_idx", i) for i, fr in enumerate(frames)),
                           default=len(frames))
        half_boundary = total_frames * FIRST_HALF_END_FRAC
        if VERBOSE:
            print(f"  [yarı sınırı] OCR/segment yöntemi kullanılamadı, "
                  f"tahmin: frame {int(half_boundary)}")

    # Her track için pozisyonları yarıya göre ayır
    tid_pos_h1 = defaultdict(list)  # 1. yarı pozisyonları
    tid_pos_h2 = defaultdict(list)  # 2. yarı pozisyonları

    for fr in frames:
        fidx = fr.get("frame_idx", 0)
        half = 1 if fidx <= half_boundary else 2
        for obj in fr.get("objects", []):
            tid = str(int(obj["track_id"]))
            if tid not in merged_tracks:
                continue
            bb = obj.get("bbox_xyxy")
            if not bb or len(bb) < 4:
                continue
            x1, y1, x2, y2 = map(float, bb[:4])
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            (tid_pos_h1 if half == 1 else tid_pos_h2)[tid].append((cx, cy))

    def _gk_team_from_positions(positions_h1, positions_h2):
        """
        Yarı başına kale bölgesi oranını hesaplayıp doğru team etiketini döner.
        None → kaleci değil veya yeterli veri yok.
        """
        def _zone_stats(positions):
            n = len(positions)
            if n == 0:
                return 0.0, 0.0, 0.0
            in_ctr   = sum(1 for cx, _ in positions if CXL <= cx <= CXR) / n
            in_left  = sum(1 for cx, cy in positions
                           if cx <= LEFT_X and Y_TOP <= cy <= Y_BOT) / n
            in_right = sum(1 for cx, cy in positions
                           if cx >= RIGHT_X and Y_TOP <= cy <= Y_BOT) / n
            return in_ctr, in_left, in_right

        # Her yarının kale bölgesi istatistiğini çıkar
        results = []
        for half_idx, positions in enumerate((positions_h1, positions_h2), start=1):
            n = len(positions)
            if n < GK_MIN_SAMPLES:
                continue
            in_ctr, in_left, in_right = _zone_stats(positions)
            if in_ctr > GK_CENTER_MAX_FRAC:
                continue                 # çok fazla orta sahadaysa kaleci değil
            zone_ratio = max(in_left, in_right)
            if zone_ratio < GK_ZONE_RATIO_MIN:
                continue                 # kale bölgesinde yeterince değil
            side_left = in_left >= in_right

            # Yarı × taraf → team etiketi
            # 1. yarı: sol = team_1, sağ = team_2
            # 2. yarı: sol = team_2, sağ = team_1
            if half_idx == 1:
                team = "team_1" if side_left else "team_2"
            else:
                team = "team_2" if side_left else "team_1"
            results.append((team, zone_ratio, n))

        if not results:
            return None, 0.0
        # Birden fazla yarı uyuyorsa ağırlıklı oy (sample sayısı ile)
        votes: Counter = Counter()
        for team, zr, n in results:
            votes[team] += zr * n
        best_team = votes.most_common(1)[0][0]
        best_zone = max(zr for _, zr, _ in results)
        return best_team, best_zone

    changes = []
    for tid in list(merged_tracks.keys()):
        current_label = merged_tracks[tid].get("label")
        # Sadece hakem kümesi — oyuncu takımları yalnızca renk (KMeans)
        if current_label != "referee":
            continue

        positions_h1 = tid_pos_h1.get(tid, [])
        positions_h2 = tid_pos_h2.get(tid, [])
        total_n = len(positions_h1) + len(positions_h2)
        if total_n < GK_MIN_SAMPLES:
            continue

        # Gerçek hakem: merkez şeritte yoğun → sahaya göre takım atama
        all_pos = positions_h1 + positions_h2
        frac_ctr = sum(1 for cx, cy in all_pos if CXL <= cx <= CXR) / len(all_pos)
        if frac_ctr >= REFEREE_CENTER_KEEP_MIN:
            continue

        correct_team, zone_ratio = _gk_team_from_positions(positions_h1, positions_h2)
        if correct_team is None:
            continue
        if correct_team == current_label:
            continue

        merged_tracks[tid]["label"] = correct_team
        merged_tracks[tid]["gk_ref_position_adjusted"] = True
        changes.append((tid, current_label, correct_team, zone_ratio))

    if changes and VERBOSE:
        print(f"\n  Referee→kaleci düzeltmesi (yalnızca referee kümesi, yarı+kale): "
              f"{len(changes)} track")
        for tid, old_lbl, new_lbl, zr in changes[:20]:
            print(f"    id {tid}  {old_lbl} → {new_lbl}  (kale bölgesi≈{zr:.0%})")
        if len(changes) > 20:
            print(f"    ... +{len(changes) - 20} daha")

    return merged_tracks


# ─── CORE: TEK CHUNK ÜZERİNDE KÜMELEYİCİ ────────────────────────────────────
def cluster_single_chunk(video_path, gameplay_json_path, out_dir):
    """
    Bir video + gameplay JSON çifti üzerinde forma rengi KMeans kümeleme yapar.
    Çıktı: track_labels.json, crop_manifest.json, crops/
    (Nihai review.mp4 pipeline sonunda birleştirilmiş etiketlerle üretilir.)
    """
    video_path         = Path(video_path)
    gameplay_json_path = Path(gameplay_json_path)
    out_dir            = Path(out_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if not gameplay_json_path.exists():
        raise FileNotFoundError(f"Gameplay JSON bulunamadı: {gameplay_json_path}")

    crops_dir    = out_dir / "crops"
    manifest_out = out_dir / "crop_manifest.json"
    labels_out   = out_dir / "track_labels.json"
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta   = load_json(gameplay_json_path)
    frames = meta["frames"]

    # ── Track örneklerini topla ──────────────────────────────────────────────
    # seq_idx: chunk video içindeki 0-bazlı pozisyon (original_frame_index DEĞİL)
    track_occurrences = defaultdict(list)
    for seq_idx, frame_info in enumerate(frames):
        for obj in frame_info["objects"]:
            tid = int(obj["track_id"])
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            if (y2 - y1) < MIN_BBOX_HEIGHT or (x2 - x1) < MIN_BBOX_WIDTH:
                continue
            track_occurrences[tid].append({
                "seq_idx":   seq_idx,   # chunk videosundaki frame pozisyonu
                "bbox_xyxy": obj["bbox_xyxy"],
                "confidence": obj["confidence"]
            })

    sampled_tracks = {}
    for tid, occs in track_occurrences.items():
        if len(occs) < MIN_TRACK_SAMPLES:
            continue
        if len(occs) <= MAX_SAMPLES_PER_TRACK:
            sampled_tracks[tid] = occs
        else:
            idxs = np.linspace(0, len(occs)-1, MAX_SAMPLES_PER_TRACK).astype(int)
            sampled_tracks[tid] = [occs[i] for i in idxs]

    if len(sampled_tracks) < 3:
        raise RuntimeError("Yeterli sayıda track yok. Video çok kısa ya da filtre çok sert.")

    # ── seq_idx → [(tid, sample_idx, bbox)] haritası ────────────────────────
    # seq_idx = chunk video içindeki 0-bazlı pozisyon
    frame_to_work = defaultdict(list)
    for tid, occs in sampled_tracks.items():
        for si, item in enumerate(occs):
            frame_to_work[item["seq_idx"]].append((tid, si, item["bbox_xyxy"]))
    needed = sorted(frame_to_work)

    # ── Tek video geçişi: crop çıkar + feature hesapla (frame cache YOK) ────
    manifest             = []
    track_feats_lists    = defaultdict(list)   # tid → [feature_vec, ...]
    track_stats_lists    = defaultdict(list)   # tid → [stats_dict, ...]
    track_crop_paths     = defaultdict(list)   # tid → [path, ...]
    track_torso_boxes    = defaultdict(list)   # tid → [torso_box, ...]

    # Her track için crops alt klasörünü önceden aç
    for tid in sampled_tracks:
        (crops_dir / f"track_{tid}").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cur = ptr = 0
    chunk_name = Path(video_path).stem
    with tqdm(total=len(needed), desc=f"Stage 4 | Crop scan ({chunk_name})",
              unit="frame", ncols=90) as pbar_scan:
     while ptr < len(needed):
        ret, frame = cap.read()
        if not ret:
            break
        if cur == needed[ptr]:
            for (tid, si, bbox) in frame_to_work[cur]:
                patch, torso_box = extract_torso_patch(frame, bbox)
                if patch is None:
                    continue
                out_path = crops_dir / f"track_{tid}" / f"frame_{cur:06d}_s{si:02d}.jpg"
                cv2.imwrite(str(out_path), patch)
                feat, stats = compute_feature(patch)
                if feat is None:
                    continue
                track_feats_lists[tid].append(feat)
                track_stats_lists[tid].append(stats)
                track_crop_paths[tid].append(str(out_path))
                track_torso_boxes[tid].append(torso_box)
                manifest.append({
                    "track_id":      tid,
                    "frame_index":   cur,
                    "bbox_xyxy":     bbox,
                    "torso_box_xyxy": torso_box,
                    "crop_path":     str(out_path),
                    "stats":         stats,
                })
            ptr += 1
            pbar_scan.update(1)
        cur += 1
    cap.release()

    # ── Feature vektörlerini topla ───────────────────────────────────────────
    track_features      = {}
    track_stats_map     = {}
    track_sample_counts = {}

    for tid, feats in track_feats_lists.items():
        if len(feats) < MIN_TRACK_SAMPLES:
            continue
        track_features[tid]      = np.mean(np.stack(feats), axis=0)
        track_sample_counts[tid] = len(feats)
        sl = track_stats_lists[tid]
        track_stats_map[tid] = {
            k: round(float(np.mean([s[k] for s in sl])), 4)
            for k in sl[0]
        }

    save_json(manifest_out, manifest)

    if len(track_features) < 3:
        raise RuntimeError("Kümeleme için yeterli kaliteli track kalmadı.")

    # ── KMeans ───────────────────────────────────────────────────────────────
    valid_ids = sorted(track_features)
    X = np.stack([track_features[t] for t in valid_ids])
    cluster_ids = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(X)

    cluster_to_tracks = defaultdict(list)
    for tid, cid in zip(valid_ids, cluster_ids):
        cluster_to_tracks[int(cid)].append(tid)

    cluster_summary = {}
    for cid, tids in cluster_to_tracks.items():
        agg = defaultdict(list)
        for tid in tids:
            for k, v in track_stats_map[tid].items():
                agg[k].append(v)
        cluster_summary[cid] = {
            "track_ids": tids, "track_count": len(tids),
            **{k: round(float(np.mean(v)), 4) for k, v in agg.items()}
        }

    # ── Küme isimlendirme: bilinen forma renklerine göre sabit eşikler ────────
    #
    # Galatasaray  → turuncu-kırmızı  (orange_ratio en yüksek)
    # Trabzonspor  → gri-siyah        (gray_ratio + dark_ratio yüksek)
    # Hakem        → açık mavi        (blue_ratio en yüksek)
    #
    # Strateji:
    #  1) Her küme için eşiğe dayalı sabit atama dene
    #  2) Atama çakışıyorsa veya eksikse göreli sıralamaya geri dön

    def _try_absolute_assignment(cs):
        """Sabit eşiklerle atama; başarısız olursa None döner."""
        scores = {}
        for cid, summ in cs.items():
            blue   = summ["blue_ratio"]
            orange = summ["orange_ratio"]
            gray   = summ.get("gray_ratio", 0)
            dark   = summ["dark_ratio"]
            if blue >= REFEREE_BLUE_MIN:
                scores[cid] = ("referee", blue)
            elif orange >= TEAM1_ORANGE_MIN:
                scores[cid] = ("team_1", orange)
            else:
                scores[cid] = ("team_2", gray + dark)
        labels = [v[0] for v in scores.values()]
        if len(set(labels)) == 3:          # tam 3 farklı etiket → başarılı
            return {cid: lbl for cid, (lbl, _) in scores.items()}
        return None

    abs_map = _try_absolute_assignment(cluster_summary)

    if abs_map is not None:
        name_map = abs_map
        if VERBOSE:
            print("  Küme isimlendirme: sabit eşik atama kullanıldı")
    else:
        # Göreli geri-dönüş: en çok mavi → hakem, en çok turuncu → team_1
        ref_cid = max(cluster_summary, key=lambda c: cluster_summary[c]["blue_ratio"])
        rem = [c for c in cluster_summary if c != ref_cid]
        t1_cid = max(rem, key=lambda c: cluster_summary[c]["orange_ratio"])
        t2_cid = [c for c in rem if c != t1_cid][0]
        name_map = {ref_cid: "referee", t1_cid: "team_1", t2_cid: "team_2"}
        if VERBOSE:
            print("  Küme isimlendirme: göreli sıralama kullanıldı (geri-dönüş)")

    # Debug: küme özet bilgisi
    if VERBOSE:
        for cid, lbl in name_map.items():
            s = cluster_summary[cid]
            print(f"  Küme {cid} → {lbl:8s}  "
                  f"orange={s['orange_ratio']:.3f}  "
                  f"blue={s['blue_ratio']:.3f}  "
                  f"gray={s.get('gray_ratio',0):.3f}  "
                  f"dark={s['dark_ratio']:.3f}  "
                  f"n={s['track_count']}")

    track_labels = {
        tid: {
            "cluster_id":     int(cid),
            "label":          name_map[int(cid)],
            "sample_count":   int(track_sample_counts[tid]),
            "feature_vector": [round(float(x), 5) for x in track_features[tid].tolist()],
            "stats":          track_stats_map[tid],
            "crop_examples":  track_crop_paths[tid][:5],
        }
        for tid, cid in zip(valid_ids, cluster_ids)
    }

    output = {
        "video_path":       str(video_path),
        "gameplay_json_path": str(gameplay_json_path),
        "cluster_name_map": {str(k): v for k, v in name_map.items()},
        "cluster_summary":  {str(k): v for k, v in cluster_summary.items()},
        "tracks":           {str(k): v for k, v in track_labels.items()},
    }
    save_json(labels_out, output)

    tqdm.write(f"  → chunk done: {out_dir.name}  |  tracks={len(valid_ids)}")
    return labels_out, manifest_out, crops_dir

# ─── MAIN: CHUNKED PIPELINE ──────────────────────────────────────────────────
def run_pipeline():
    for p in [SRC_VIDEO, SRC_JSON]:
        if not p.exists():
            raise FileNotFoundError(p)

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUT.mkdir(parents=True, exist_ok=True)

    meta   = load_json(SRC_JSON)
    frames = meta["frames"]
    total  = len(frames)
    chunk_size = math.ceil(total / N_CHUNKS)
    overlap    = max(0, int(total * CHUNK_OVERLAP_RATIO / N_CHUNKS))

    chunk_ranges = []
    for i in range(N_CHUNKS):
        # Her chunk sınırlarını overlap kadar genişlet
        s = max(0,     i * chunk_size - overlap)
        e = min(total, (i + 1) * chunk_size + overlap)
        if s < e:
            chunk_ranges.append((s, e))

    chunks_dir = WORKDIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStage 4 | Team Clustering — {len(chunk_ranges)} chunk, toplam {total} frame")
    print(f"  Video bölünüyor...")
    chunk_videos = split_video_exact(SRC_VIDEO, chunk_ranges, chunks_dir)

    labels_paths, manifest_paths, crop_dirs = [], [], []
    merged_tracks = {}

    chunk_pbar = tqdm(list(zip(chunk_ranges, chunk_videos)), desc="Stage 4 | Chunks",
                      unit="chunk", ncols=90)
    for idx, ((s, e), chunk_video) in enumerate(chunk_pbar, 1):
        chunk_pbar.set_postfix({"chunk": f"{idx}/{len(chunk_ranges)}", "frames": f"{s}..{e-1}"})
        chunk_json = chunks_dir / f"chunk_{idx:02d}.json"
        chunk_out  = WORKDIR / f"chunk_{idx:02d}"

        chunk_meta = dict(meta)
        chunk_meta["frames"] = frames[s:e]
        save_json(chunk_json, chunk_meta)

        lp, mp, cd = cluster_single_chunk(chunk_video, chunk_json, chunk_out)
        labels_paths.append(lp)
        manifest_paths.append(mp)
        crop_dirs.append(cd)

        tracks_data = load_json(lp)
        tracks, _ = normalize_track_labels(tracks_data)

        if idx == 1:
            merged_tracks = tracks
        else:
            aligned, flipped = align_chunk_to_merged(merged_tracks, tracks)
            tqdm.write(f"  chunk {idx}: flip={flipped}")
            for k, v in aligned.items():
                if k not in merged_tracks:
                    merged_tracks[k] = v

    # ── Sonuçları birleştir ──────────────────────────────────────────────────
    print("\nMerging results...")
    _w = int(meta.get("width") or 1920)
    _h = int(meta.get("height") or 1080)

    # Renk tabanlı etiket (konum düzeltmesinden önce) — gerekirse geri almak için
    for _tid, _rec in merged_tracks.items():
        _rec["label_from_clustering"] = _rec.get("label")

    print("  Yarı sınırı tespit ediliyor...")
    half_boundary = detect_half_boundary(SRC_VIDEO, SRC_JSON)

    merged_tracks = reclassify_referee_goalkeepers(
        merged_tracks, frames, _w, _h, half_boundary=half_boundary
    )
    save_json(FINAL_OUT / "track_labels.json", {"tracks": merged_tracks})
    merge_manifests(manifest_paths, FINAL_OUT / "crop_manifest.json")
    copy_crops(crop_dirs, FINAL_OUT / "crops")

    try:
        fps_hint = float(meta.get("fps") or 0)
        render_review_video_merged_labels(
            SRC_VIDEO, frames, merged_tracks, FINAL_OUT / "review.mp4", fps_hint=fps_hint or None
        )
    except Exception as exc:
        print(f"  review video (final) atlandı: {exc}")

    print("\nDONE")
    print("  track_labels :", FINAL_OUT / "track_labels.json")
    print("  crop_manifest:", FINAL_OUT / "crop_manifest.json")
    print("  crops        :", FINAL_OUT / "crops")
    print("  review video :", FINAL_OUT / "review.mp4")


def run_review_only():
    """Mevcut track_labels.json + gameplay → review.mp4 (kümeleme çalıştırmadan)."""
    tl_path = FINAL_OUT / "track_labels.json"
    for p in (SRC_VIDEO, SRC_JSON, tl_path):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    meta = load_json(SRC_JSON)
    frames = meta["frames"]
    data = load_json(tl_path)
    merged_tracks, _ = normalize_track_labels(data)
    fps_hint = float(meta.get("fps") or 0)
    out = FINAL_OUT / "review.mp4"
    render_review_video_merged_labels(
        SRC_VIDEO, frames, merged_tracks, out, fps_hint=fps_hint or None
    )
    print("\nDONE  review-only →", out)


if __name__ == "__main__":
    import sys

    if "--review-only" in sys.argv:
        run_review_only()
    else:
        run_pipeline()
