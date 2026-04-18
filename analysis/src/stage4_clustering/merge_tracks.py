"""
Stage 4c — Track Birleştirme (Track Merge)

Aynı oyuncunun occlusion sonrası yeni track ID aldığı durumları tespit eder.
Temporal olarak ardışık, uzamsal olarak yakın, aynı takım track'lerini
canonical_id altında birleştirir.

Her track'e canonical_id alanı eklenir — birleştirilen track'ler aynı
canonical_id'yi paylaşır. Downstream analysis bu ID'yi kullanabilir.

Giriş : track_labels_corrected.json, projection.json
Çıkış : track_labels_corrected.json (canonical_id alanı eklenerek güncellenir)
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

TRACK_LABELS_JSON = "/output/stage4_clustering/track_labels_corrected.json"
PROJECTION_JSON   = "/output/stage6_field/projection.json"

# ─── PARAMETRELER ─────────────────────────────────────────────────────────────
MAX_TEMPORAL_GAP_FRAMES = 90    # Frame cinsinden max boşluk (≈3.6s @ 25fps)
MAX_SPATIAL_GAP_M       = 5.0   # Track bitişi ile başlangıcı arasındaki max mesafe
MIN_TRACK_FRAMES        = 15    # Bu kadardan kısa track adaylar listesine alınmaz

def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

with open(TRACK_LABELS_JSON, "r", encoding="utf-8") as f:
    labels = json.load(f)
with open(PROJECTION_JSON, "r", encoding="utf-8") as f:
    proj = json.load(f)

tracks = labels["tracks"]
frames = proj["frames"]

# ─── HER TRACK İÇİN TEMPORAL + UZAMSAL ÖZET ──────────────────────────────────
print(f"\nStage 4c | Track Merge — {len(frames)} frame taranıyor")

track_info = defaultdict(lambda: {
    "first_frame": None, "last_frame": None,
    "first_xy": None, "last_xy": None,
    "frame_count": 0,
})

for frame_idx, fr in enumerate(
        tqdm(frames, desc="Stage 4c | Scan       ", unit="frame", ncols=90)):
    for obj in fr["players"] + fr["referees"]:
        tid = int(obj["track_id"])
        if str(tid) not in tracks:
            continue
        xy = obj.get("pitch_xy")
        if xy is None:
            continue
        info = track_info[tid]
        info["frame_count"] += 1
        if info["first_frame"] is None:
            info["first_frame"] = frame_idx
            info["first_xy"]    = xy
        info["last_frame"] = frame_idx
        info["last_xy"]    = xy

# ─── UNION-FIND ────────────────────────────────────────────────────────────────
parent = {}

def find(x):
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
        x = parent.get(x, x)
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry:
        parent[ry] = rx  # küçük ID canonical olsun (rx < ry garantisi yok — sonra min ile düzeltiriz)

# ─── MERGE ADİŞ ───────────────────────────────────────────────────────────────
# Aynı takımdan iki track:
#  - A bitmeden önce B başlamış (temporal gap ≤ MAX_TEMPORAL_GAP_FRAMES)
#  - A'nın son noktası ile B'nin ilk noktası uzamsal olarak yakın
candidates = [
    (tid, info) for tid, info in track_info.items()
    if (info["frame_count"] >= MIN_TRACK_FRAMES
        and info["first_frame"] is not None
        and info["last_frame"] is not None)
]
candidates.sort(key=lambda x: x[1]["first_frame"])

merge_count = 0
for i, (tid_a, info_a) in enumerate(
        tqdm(candidates, desc="Stage 4c | Merge scan ", unit="track", ncols=90)):

    label_a = tracks[str(tid_a)]["label"]
    if label_a not in ("team_1", "team_2"):
        continue

    for j in range(i + 1, len(candidates)):
        tid_b, info_b = candidates[j]

        # B zaten A'dan önce başladıysa atla
        if info_b["first_frame"] <= info_a["last_frame"]:
            continue
        # Temporal boşluk çok büyükse sıradaki B'ler daha da büyük → kır
        if info_b["first_frame"] - info_a["last_frame"] > MAX_TEMPORAL_GAP_FRAMES:
            break

        label_b = tracks[str(tid_b)]["label"]
        if label_b != label_a:
            continue

        # Uzamsal mesafe kontrolü
        if info_a["last_xy"] is None or info_b["first_xy"] is None:
            continue
        d = dist(info_a["last_xy"], info_b["first_xy"])
        if d <= MAX_SPATIAL_GAP_M:
            union(tid_a, tid_b)
            merge_count += 1

# ─── CANONICAL ID ATAMA ────────────────────────────────────────────────────────
# Her grubun en küçük track_id'si canonical olsun
groups = defaultdict(list)
for tid in track_info:
    groups[find(tid)].append(tid)

canonical_map = {}  # tid → canonical_id
for root, members in groups.items():
    canonical_id = min(members)
    for tid in members:
        canonical_map[tid] = canonical_id

# ─── LABELS'A canonical_id EKLE VE YAZ ────────────────────────────────────────
updated_tracks = {}
for tid_str, info in tracks.items():
    tid = int(tid_str)
    canonical_id = canonical_map.get(tid, tid)  # birleştirilmediyse kendisi
    updated_info = dict(info)
    updated_info["canonical_id"] = canonical_id
    updated_tracks[tid_str] = updated_info

labels["tracks"] = updated_tracks
if "track_merge" not in labels:
    labels["track_merge"] = {}
labels["track_merge"].update({
    "max_temporal_gap_frames": MAX_TEMPORAL_GAP_FRAMES,
    "max_spatial_gap_m":       MAX_SPATIAL_GAP_M,
    "min_track_frames":        MIN_TRACK_FRAMES,
    "merge_events":            merge_count,
    "total_tracks":            len(tracks),
    "canonical_tracks":        len(set(canonical_map.values())),
})

with open(TRACK_LABELS_JSON, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print("DONE")
print(f"  merge_events     = {merge_count}")
print(f"  total_tracks     = {len(tracks)}")
print(f"  canonical_tracks = {len(set(canonical_map.values()))}")
print(f"  out              = {TRACK_LABELS_JSON}")
