"""
Vision Module: Video Processing Pipeline
Handles object detection, tracking, team classification, and ball ownership.

ID Format:
- Sol Takım: L1, L2, L3, ..., L11 (L1 = kaleci)
- Sağ Takım: R1, R2, R3, ..., R11 (R1 = kaleci)
- Top: "ball"
"""
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Avoid runtime pip auto-installs (can trigger uvicorn reload loops in dev).
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from ultralytics import YOLO
from sklearn.cluster import KMeans

from ml_pipeline.detection.ball_interpolation import interpolate_ball_positions
from ml_pipeline.tracking.bytetrack import ByteTrackTracker
from ml_pipeline.tracking.id_stabilizer import TeamBasedIDStabilizer
from ml_pipeline.detection.ball_ownership import BallOwnershipProcessor
from ml_pipeline.detection.postprocess import draw_box
from ml_pipeline.detection.team_id import estimate_pitch_hsv as est_pitch_hsv
from ml_pipeline.detection.team_id import trim_outliers
from ml_pipeline.detection.team_classifier import get_grass_color, get_kits_colors, get_kits_classifier, classify_kits
from ml_pipeline.io.writer import write_meta
from schemas.types import FrameRecord, Track, get_team_from_id

# Detection parameters
DET_CONF = 0.25
DET_IOU = 0.50
DET_IMGSZ = 1280


def get_interpolated_P(frame_num, calibrations, calib_frames):
    """
    Verilen frame için interpolasyon yaparak P matrix döndür
    İki kalibrasyon arasında lineer interpolasyon yapar (run_soccer_line.py'den)
    """
    if len(calib_frames) == 0:
        return None
    
    # Tam eşleşme var mı?
    if frame_num in calibrations:
        return calibrations[frame_num]
    
    # En yakın önceki ve sonraki kalibrasyonu bul
    prev_frame = None
    next_frame = None
    
    for cf in calib_frames:
        if cf <= frame_num:
            prev_frame = cf
        if cf > frame_num and next_frame is None:
            next_frame = cf
            break
    
    # Sadece önceki varsa
    if prev_frame is not None and next_frame is None:
        return calibrations[prev_frame]
    
    # Sadece sonraki varsa
    if prev_frame is None and next_frame is not None:
        return calibrations[next_frame]
    
    # İkisi de varsa - interpolasyon yap
    if prev_frame is not None and next_frame is not None:
        # İki P matrix arasında lineer interpolasyon
        t = (frame_num - prev_frame) / (next_frame - prev_frame)
        P_prev = calibrations[prev_frame]
        P_next = calibrations[next_frame]
        
        # Lineer interpolasyon
        P_interp = (1 - t) * P_prev + t * P_next
        return P_interp
    
    return None

# Drawing configuration
DRAW_LABELS = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
BOX_COLORS = {
    0: (150, 50, 50), 1: (37, 47, 150), 2: (41, 248, 165), 3: (166, 196, 10),
    4: (155, 62, 157), 5: (123, 174, 213), 6: (217, 89, 204), 7: (22, 11, 15),
}

# Team classification parameters
COLOR_LOCK_FRAMES = 24
REF_LABELS = {3, 4}
REF_MIN_CONF = 0.35
MAIN_BAND_FRAC = 0.60
MAIN_NMS_IOU = 0.50
PROMOTE_ONLY_IF_MISSING = True
REF_V_MAX = 160
REF_S_MAX = 110
PERSIST_W = 0.65
REF_DIST_TAU = 32.0
SIDELINE_FRAC = 0.16
UPPER_FRAC = 0.45
S_MIN, V_MIN = 60, 50
REF_WARMUP_MIN = 3
TAU_REF_DELTAE_MIN = 18.0
TAU_REF_DELTAE_MAX = 35.0
MAIN_REF_LOCK_FRAMES = 8
SIDE_REF_LOCK_FRAMES = 10


def detect_and_track(video_path: str, model_path: str = "./weights/last.pt", 
                     output_dir: str = "./output",
                     progress_callback=None) -> Tuple[List[FrameRecord], Dict]:
    """
    Main vision pipeline: detection + tracking + team classification.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model weights
        output_dir: Directory for output files
        progress_callback: Optional callback(progress_percent, details) for progress updates
        
    Returns:
        (frames_list, metadata_dict)
        - frames_list: List of FrameRecord objects with tracks
        - metadata_dict: Video metadata (fps, width, height, teams, etc.)
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if not input_fps or input_fps <= 0:
        input_fps = 25.0
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Processing video: {width}x{height} @ {input_fps} fps, {total_frames} frames")
    
    # Warmup: grass color + team colors + referee template
    warmup_frames = []
    for _ in range(15):
        ok, f = cap.read()
        if not ok:
            break
        warmup_frames.append(cv2.resize(f, (width, height)))
    
    if not warmup_frames:
        cap.release()
        raise ValueError("No frames could be read from video")
    
    # Estimate pitch color
    grass_hsv_median = None
    if warmup_frames:
        H, S, V = est_pitch_hsv(warmup_frames)
        grass_hsv_median = np.array([[[H, S, V]]], dtype=np.uint8)
    
    # Build RAW class mapping
    RAW = _build_raw_ids(model)
    print(f"🎯 Class mapping: {RAW}")
    
    # Team classification warmup
    kits_clf, color_lock = _warmup_color_lock(
        cap, model, width, height, grass_hsv_median, 
        frames=COLOR_LOCK_FRAMES, RAW=RAW
    )
    
    centroids_bgr = None
    if kits_clf is not None:
        try:
            centroids_bgr = kits_clf.cluster_centers_.astype(np.float32)
        except Exception:
            centroids_bgr = None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Referee color template
    ref_mu_lab = None
    if color_lock is not None and color_lock.get("ref_mu_lab") is not None:
        ref_mu_lab = np.array(color_lock["ref_mu_lab"], dtype=np.float32)
    
    # Setup output
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)
    base_name = Path(video_path).stem
    
    # Use H.264 codec for browser compatibility - try multiple codecs
    track_vis_path = os.path.join(output_dir, "track_vis_out.mp4")
    
    # Try codecs in order of preference for browser compatibility
    codecs_to_try = [
        ('avc1', 'H.264 (avc1)'),
        ('H264', 'H.264 (H264)'),
        ('X264', 'H.264 (X264)'),
        ('mp4v', 'MPEG-4 (mp4v)'),
    ]
    
    output_video = None
    for codec, codec_name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_video = cv2.VideoWriter(track_vis_path, fourcc, input_fps, (width, height))
        if output_video.isOpened():
            print(f"✅ Using video codec: {codec_name}")
            break
        else:
            print(f"⚠️  {codec_name} codec not available, trying next...")
            output_video = None
    
    if output_video is None or not output_video.isOpened():
        raise RuntimeError("No suitable video codec found! Please install ffmpeg or OpenCV with H.264 support.")
    
    # NOT: Soccer Line 2D video artık pipeline.py'de ayrıca oluşturuluyor (birebir aynı sonuç için)
    
    # Initialize trackers
    tracker = ByteTrackTracker(fps=input_fps, names=model.names, track_ball=True)
    # NOT: ID Stabilization artık tracker içinde yapılıyor (TeamBasedIDStabilizer)
    ownership_processor = BallOwnershipProcessor(
        fps=input_fps, smoothing_window=0.6, distance_threshold=150,
        extreme_speed_threshold=25.0  # Şut seviyesinde hız - bu hızda ownership verilmez
    )
    
    # Process frames
    frames_list = []
    frame_idx = 0
    grass_hsv = grass_hsv_median
    
    # Ref tracking state
    last_main_ref_box = None
    main_ref_lock_t = 0
    last_side_ref_boxes = []
    side_ref_lock_t = 0
    
    print("🔄 Processing frames...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_idx += 1
        t = frame_idx / input_fps
        
        # Resize and detect
        annotated_frame = cv2.resize(frame, (width, height))
        result = model(
            annotated_frame, conf=DET_CONF, iou=DET_IOU, 
            imgsz=DET_IMGSZ, agnostic_nms=False, verbose=False
        )[0]
        
        # Process detections (team classification, referee detection)
        drawn = _process_detections(
            result, annotated_frame, kits_clf, color_lock, centroids_bgr,
            ref_mu_lab, grass_hsv, width, height, RAW,
            last_main_ref_box, main_ref_lock_t, last_side_ref_boxes, side_ref_lock_t
        )
        
        # Update ref tracking state
        main_ref_lock_t, side_ref_lock_t, last_main_ref_box, last_side_ref_boxes = \
            _update_ref_state(drawn, main_ref_lock_t, side_ref_lock_t)
        
        # Draw boxes
        for d in drawn:
            x1, y1, x2, y2 = d["bbox"]
            draw_box(annotated_frame, x1, y1, x2, y2, d["draw"])
        
        # Debug overlay
        _draw_debug_overlay(annotated_frame, kits_clf, color_lock, grass_hsv, drawn, ref_mu_lab, centroids_bgr)
        
        # Takım sınıflandırma fonksiyonu oluştur
        def team_classifier_fn(img, bbox):
            """Bbox içindeki oyuncunun takımını belirle (player ve goalkeeper için)."""
            best_iou_score = 0.0
            matched_team = None
            for d in drawn:
                # 0=Player-L, 1=Player-R, 2=GK-L, 3=GK-R
                # GK-L (2) -> Team 0, GK-R (3) -> Team 1
                if d["draw"] in [0, 1, 2, 3]:
                    score = _iou(bbox, d["bbox"])
                    if score > 0.50 and score > best_iou_score:
                        best_iou_score = score
                        # 0 ve 2 -> Team 0 (Left), 1 ve 3 -> Team 1 (Right)
                        matched_team = 0 if d["draw"] in [0, 2] else 1
            return matched_team
        
        # ByteTrack tracking (artık takım sınıflandırması ve ID stabilization burada)
        tracks = tracker.step(result, frame_idx=frame_idx, team_classifier=team_classifier_fn)
        
        # Ball ownership
        current_owner_id, ball_dist = ownership_processor.update(tracks)
        current_owner_team = None
        
        if current_owner_id is not None:
            owner_track = next((t for t in tracks if t.track_id == current_owner_id), None)
            if owner_track:
                # Yeni sistemde takım ID'den çıkarılıyor
                current_owner_team = get_team_from_id(current_owner_id)
                
                # Draw ownership line
                ball_tr = next((t for t in tracks if t.track_id == "ball"), None)
                if ball_tr:
                    bx_c = (int((ball_tr.bbox[0]+ball_tr.bbox[2])/2), int((ball_tr.bbox[1]+ball_tr.bbox[3])/2))
                    px_c = (int((owner_track.bbox[0]+owner_track.bbox[2])/2), int((owner_track.bbox[1]+owner_track.bbox[3])/2))
                    line_color = (255, 0, 0) if current_owner_team == 0 else (0, 0, 255) if current_owner_team == 1 else (255, 255, 255)
                    cv2.line(annotated_frame, bx_c, px_c, line_color, 2)
                    team_text = f"L" if current_owner_team == 0 else "R" if current_owner_team == 1 else "?"
                    label_text = f"Ball: {current_owner_id}"
                    cv2.putText(annotated_frame, label_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        
        # Store frame record
        frame_record = FrameRecord(
            frame_idx=frame_idx, t=t, tracks=tracks,
            ball_owner=current_owner_id, ball_owner_team=current_owner_team
        )
        frames_list.append(frame_record)
        
        # Draw ID overlays (L3, R5, ball formatında)
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            # Yeni format: direkt track_id göster (L3, R5, ball)
            label_txt = str(tr.track_id)
            # Takıma göre renk
            team = get_team_from_id(tr.track_id)
            if team == 0:
                color = (255, 100, 100)  # Kırmızımsı (L takımı)
            elif team == 1:
                color = (100, 100, 255)  # Mavimsi (R takımı)
            else:
                color = (0, 255, 255)    # Sarı (top)
            cv2.putText(annotated_frame, label_txt, (int(x1), max(int(y1)-5, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Video: Normal track video
        output_video.write(annotated_frame)
        
        # Report progress
        if progress_callback and total_frames > 0:
            progress_pct = (frame_idx / total_frames) * 100
            if frame_idx % 10 == 0:  # Her 10 frame'de bir rapor et (daha sık güncelleme)
                progress_callback(progress_pct, f"Frame {frame_idx}/{total_frames}")
        
        if frame_idx % 100 == 0:
            stats = tracker._id_stabilizer.get_stats() if tracker._id_stabilizer else {}
            print(f"[Frame {frame_idx}/{total_frames}] L={stats.get('left_team_count', 0)}, "
                  f"R={stats.get('right_team_count', 0)}, Total={stats.get('active_players', 0)}")
    
    # Cleanup
    output_video.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Vision processing complete: {len(frames_list)} frames")
    print(f"📹 Track visualization: {track_vis_path}")
    # POST-PROCESSING: Ball interpolation and ownership recalculation
    # This fills gaps where ball was not detected (e.g., occluded by players)
    print("🔄 Post-processing: Ball interpolation and ownership recalculation...")
    frames_list = _postprocess_ball_ownership(frames_list, input_fps)
    print(f"✅ Post-processing complete")
    # Build metadata
    metadata = {
        "fps": float(input_fps),
        "width": int(width),
        "height": int(height),
        "total_frames": len(frames_list),
        "left_cluster": int(color_lock["left_cluster"]) if color_lock and "left_cluster" in color_lock else None,
        "right_cluster": int(color_lock["right_cluster"]) if color_lock and "right_cluster" in color_lock else None,
        "grass_h": int(grass_hsv[0,0,0]) if grass_hsv is not None else None,
        "has_ref_mu": ref_mu_lab is not None,
    }
    
    # Write metadata
    write_meta(output_dir, video_name, input_fps, width, height, 
               conf=DET_CONF, iou=DET_IOU, model_name=str(model), extra=metadata)
    
    return frames_list, metadata


# ==================== HELPER FUNCTIONS ====================
# (Simplified versions - full implementation in original main.py)

def _build_raw_ids(model):
    """Build class ID mapping from model names."""
    names = {i: str(n).lower().strip() for i, n in model.names.items()}
    
    def find(*cands):
        cands_norm = [c.lower().replace('_', ' ').replace('-', ' ').strip() for c in cands]
        for cid, nm in names.items():
            nn = nm.lower().replace('_', ' ').replace('-', ' ').strip()
            if nn in cands_norm:
                return cid
            for c in cands_norm:
                if all(tok in nn for tok in c.split()):
                    return cid
        return None
    
    return {
        "player": find("player"),
        "goalkeeper": find("goalkeeper", "goal keeper", "goalie", "gk"),
        "ball": find("ball"),
        "main_ref": find("main referee", "main ref", "referee"),
        "side_ref": find("side referee", "assistant referee", "linesman"),
        "staff": find("staff", "coach"),
    }


def _warmup_color_lock(cap, model, width, height, grass_hsv, frames=100, RAW=None):
    """Perform team color classification warmup with position voting across multiple frames.
    
    IMPROVED: 
    - İlk 30 frame'i atlar (oyuncular henüz konumlanmamış olabilir)
    - Her iki tarafta da en az 3 oyuncu görünene kadar frame'i sayar
    - Yeterli güven skoru (vote farkı) olana kadar devam eder
    - Maksimum 300 frame'e kadar analiz edebilir
    """
    # Import here to avoid circular dependency
    from ml_pipeline.detection.team_classifier import get_grass_color, get_kits_colors, get_kits_classifier
    
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    all_colors = []  # Tüm renkler (KMeans için)
    
    # Her frame için pozisyon bilgisi
    frame_data = []  # [(frame_idx, colors, positions)]
    
    # IYILEŞTIRME: Parametreler
    MIN_PLAYERS_PER_SIDE = 3  # Her iki tarafta da en az 3 oyuncu olmalı
    MIN_VALID_FRAMES = 50  # En az 50 geçerli frame olmalı
    MAX_FRAMES = 300  # Maksimum 300 frame analiz et
    MIN_VOTE_DIFFERENCE = 10  # Sol ve sağ arasında en az 10 vote farkı olmalı
    
    frame_count = 0
    valid_frame_count = 0  # Her iki tarafta da yeterli oyuncu olan frame sayısı
    
    while frame_count < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        
        frame_count += 1
            
        fr = cv2.resize(f, (width, height))
        r = model(fr, conf=DET_CONF, iou=DET_IOU, imgsz=DET_IMGSZ, verbose=False)[0]
        
        # Extract player colors + positions (SADECE PLAYER - goalkeeper hariç)
        players_imgs, players_x_pos = [], []
        for box in r.boxes:
            label = int(box.cls.cpu().numpy()[0])
            # SADECE player'ları al - goalkeeper, hakem, staff, top hariç
            if RAW and label == RAW.get("player"):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                players_imgs.append(r.orig_img[y1:y2, x1:x2])
                # X pozisyonunu merkez noktadan hesapla
                x_center = (x1 + x2) / 2.0
                players_x_pos.append(x_center)
        
        if players_imgs:
            cols = get_kits_colors(players_imgs, grass_hsv, fr)
            if cols and len(cols) == len(players_x_pos):
                # IYILEŞTIRME: Her iki tarafta da yeterli oyuncu var mı kontrol et
                mid_x = width / 2.0
                left_count = sum(1 for x in players_x_pos if x < mid_x)
                right_count = sum(1 for x in players_x_pos if x >= mid_x)
                
                # Her iki tarafta da en az MIN_PLAYERS_PER_SIDE oyuncu varsa frame'i kabul et
                if left_count >= MIN_PLAYERS_PER_SIDE and right_count >= MIN_PLAYERS_PER_SIDE:
                    all_colors.extend(cols)
                    frame_data.append((frame_count, cols, players_x_pos))
                    valid_frame_count += 1
        
        # Yeterli geçerli frame toplandıysa dur
        if valid_frame_count >= frames:
            break
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    
    print(f"[TeamClassifier] Analyzed {frame_count} frames, {valid_frame_count} valid frames (both sides have {MIN_PLAYERS_PER_SIDE}+ players)")
    
    if len(all_colors) < 4:
        return None, {"left_cluster": 0, "right_cluster": 1, "ref_mu_lab": None}
    
    # Renkleri KMeans ile cluster'la
    colors_array = np.array(all_colors, dtype=np.float32)
    kc = trim_outliers(colors_array, "iqr", 1.5)
    clf = get_kits_classifier(kc)
    
    # POZİSYON BAZLI OYlama: frame boyunca sayaç tut
    left_cluster0_votes = 0  # Cluster 0 solda kaç kere çoğunlukta
    left_cluster1_votes = 0  # Cluster 1 solda kaç kere çoğunlukta
    right_cluster0_votes = 0  # Cluster 0 sağda kaç kere çoğunlukta
    right_cluster1_votes = 0  # Cluster 1 sağda kaç kere çoğunlukta
    
    # IYILEŞTIRME: Frame bazlı tutarlılık için ek sayaçlar
    consistent_frames = 0  # Sol ve sağda farklı cluster çoğunlukta olan frame'ler
    
    for frame_idx, colors, positions in frame_data:
        # colors ve positions aynı uzunlukta olmalı
        if len(colors) != len(positions) or len(colors) < 6:
            continue
        
        # Pozisyonlara göre sırala
        sorted_indices = np.argsort(positions)
        leftmost_3 = sorted_indices[:3]
        rightmost_3 = sorted_indices[-3:]
        
        # Renkleri cluster'la
        leftmost_colors = np.array([colors[i] for i in leftmost_3], dtype=np.float32)
        rightmost_colors = np.array([colors[i] for i in rightmost_3], dtype=np.float32)
        
        leftmost_clusters = clf.predict(leftmost_colors)
        rightmost_clusters = clf.predict(rightmost_colors)
        
        # Soldaki 3 oyuncuda hangi cluster çoğunlukta?
        left_c0 = np.sum(leftmost_clusters == 0)
        left_c1 = np.sum(leftmost_clusters == 1)
        
        # Sağdaki 3 oyuncuda hangi cluster çoğunlukta?
        right_c0 = np.sum(rightmost_clusters == 0)
        right_c1 = np.sum(rightmost_clusters == 1)
        
        # IYILEŞTIRME: Sadece net çoğunluk varsa say (en az 2/3)
        left_dominant = None
        right_dominant = None
        
        if left_c0 >= 2:
            left_dominant = 0
            left_cluster0_votes += 1
        elif left_c1 >= 2:
            left_dominant = 1
            left_cluster1_votes += 1
        
        if right_c0 >= 2:
            right_dominant = 0
            right_cluster0_votes += 1
        elif right_c1 >= 2:
            right_dominant = 1
            right_cluster1_votes += 1
        
        # Tutarlılık: Sol ve sağda farklı cluster'lar mı dominant?
        if left_dominant is not None and right_dominant is not None and left_dominant != right_dominant:
            consistent_frames += 1
    
    # MAKSIMUM DEĞERE GÖRE ATAMA
    # Cluster 0'ın sol ve sağ vote'larını karşılaştır
    # Cluster 1'in sol ve sağ vote'larını karşılaştır
    
    total_valid_votes = len(frame_data)
    consistency_ratio = consistent_frames / max(1, total_valid_votes)
    
    print(f"[TeamClassifier] Voting results over {len(frame_data)} valid frames:")
    print(f"  Cluster 0: LEFT={left_cluster0_votes}, RIGHT={right_cluster0_votes}")
    print(f"  Cluster 1: LEFT={left_cluster1_votes}, RIGHT={right_cluster1_votes}")
    print(f"  Consistency: {consistent_frames}/{total_valid_votes} frames ({consistency_ratio*100:.1f}%)")
    
    # 100 frame voting sonuçları (DEBUG)
    print(f"\n🗳️  Voting Sonuçları ({len(frame_data)} geçerli frame):")
    print(f"   Sol taraftaki oyuncular:")
    print(f"     Cluster 0: {left_cluster0_votes} oy")
    print(f"     Cluster 1: {left_cluster1_votes} oy")
    print(f"   Sağ taraftaki oyuncular:")
    print(f"     Cluster 0: {right_cluster0_votes} oy")
    print(f"     Cluster 1: {right_cluster1_votes} oy")
    print(f"   Tutarlılık: %{consistency_ratio*100:.1f}")
    
    # IYILEŞTIRME: Düşük tutarlılık uyarısı
    if consistency_ratio < 0.5:
        print(f"⚠️  UYARI: Düşük tutarlılık ({consistency_ratio*100:.1f}%). Takım ataması güvenilir olmayabilir!")
    
    # Maksimum oy hangisinde?
    votes = {
        'cluster0_left': left_cluster0_votes,
        'cluster0_right': right_cluster0_votes,
        'cluster1_left': left_cluster1_votes,
        'cluster1_right': right_cluster1_votes
    }
    
    max_vote_key = max(votes, key=votes.get)
    max_vote_value = votes[max_vote_key]
    
    # IYILEŞTIRME: Güven kontrolü - ikinci en yüksek ile fark yeterli mi?
    sorted_votes = sorted(votes.values(), reverse=True)
    vote_difference = sorted_votes[0] - sorted_votes[1] if len(sorted_votes) > 1 else sorted_votes[0]
    MIN_VOTE_MARGIN = max(5, total_valid_votes * 0.1)  # En az 5 veya %10 fark olmalı
    
    if vote_difference < MIN_VOTE_MARGIN:
        print(f"⚠️  UYARI: Vote farkı çok düşük ({vote_difference}). Daha fazla frame analiz edilmeli!")
    
    # En yüksek oya göre karar ver
    # MANTIK: Soldaki oyuncular → Sol kaleyi savunuyor → LEFT team
    #         Sağdaki oyuncular → Sağ kaleyi savunuyor → RIGHT team
    
    print(f"\n✅ Karar:")
    if max_vote_key == 'cluster0_left':
        # Cluster 0 en çok solda görünüyor → Cluster 0 = LEFT team
        left_cluster = 0
        print(f"   Cluster 0 = LEFT team (max: {max_vote_value} votes on LEFT)")
        print(f"   Cluster 1 = RIGHT team")
    elif max_vote_key == 'cluster0_right':
        # Cluster 0 en çok sağda görünüyor → Cluster 0 = RIGHT team
        left_cluster = 1  # LEFT team = Cluster 1
        print(f"   Cluster 0 = RIGHT team (max: {max_vote_value} votes on RIGHT)")
        print(f"   Cluster 1 = LEFT team")
    elif max_vote_key == 'cluster1_left':
        # Cluster 1 en çok solda görünüyor → Cluster 1 = LEFT team
        left_cluster = 1
        print(f"   Cluster 1 = LEFT team (max: {max_vote_value} votes on LEFT)")
        print(f"   Cluster 0 = RIGHT team")
    else:  # cluster1_right
        # Cluster 1 en çok sağda görünüyor → Cluster 1 = RIGHT team
        left_cluster = 0  # LEFT team = Cluster 0
        print(f"   Cluster 1 = RIGHT team (max: {max_vote_value} votes on RIGHT)")
        print(f"   Cluster 0 = LEFT team")
    
    return clf, {"left_cluster": left_cluster, "right_cluster": 1 - left_cluster, "ref_mu_lab": None}


def _process_detections(result, frame, kits_clf, color_lock, centroids_bgr, 
                       ref_mu_lab, grass_hsv, width, height, RAW,
                       last_main_ref_box, main_ref_lock_t, 
                       last_side_ref_boxes, side_ref_lock_t):
    """Process YOLO detections: team classification + referee detection."""
    from ml_pipeline.detection.team_classifier import get_kits_colors
    
    drawn = []
    
    for box in result.boxes:
        raw_label = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else 1.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        draw_label = raw_label
        
        if raw_label == RAW.get("player"):
            # SADECE RENK BAZLI takım sınıflandırma (pozisyon fallback YOK)
            draw_label = None  # Renk belirlenemezse None kalacak
            if kits_clf is not None and color_lock is not None:
                try:
                    player_img = result.orig_img[y1:y2, x1:x2]
                    if player_img.size > 0:
                        kit_colors = get_kits_colors([player_img], grass_hsv, frame)
                        if kit_colors:
                            import numpy as np
                            cluster = int(kits_clf.predict(np.asarray([kit_colors[0]], np.float32))[0])
                            # cluster 0 veya 1 -> draw_label olarak kullan
                            # color_lock'a göre left=0, right=1 eşlemesi
                            left_cluster = color_lock.get("left_cluster", 0)
                            if cluster == left_cluster:
                                draw_label = 0  # Left team
                            else:
                                draw_label = 1  # Right team
                except Exception:
                    pass  # Renk alınamazsa draw_label = None kalır
        elif raw_label == RAW.get("goalkeeper"):
            # Kaleci için POZİSYON BAZLI takım ataması yap
            # Sol taraftaki kaleci -> GK-L (2), Sağ taraftaki kaleci -> GK-R (3)
            bbox_center_x = (x1 + x2) / 2
            screen_left_threshold = width * 0.35
            screen_right_threshold = width * 0.65
            
            if bbox_center_x < screen_left_threshold:
                draw_label = 2  # GK-L (sol taraf)
            elif bbox_center_x > screen_right_threshold:
                draw_label = 3  # GK-R (sağ taraf)
            else:
                # Ortadaysa renk bazlı sınıflandır
                draw_label = None
                if kits_clf is not None and color_lock is not None:
                    try:
                        player_img = result.orig_img[y1:y2, x1:x2]
                        if player_img.size > 0:
                            kit_colors = get_kits_colors([player_img], grass_hsv, frame)
                            if kit_colors:
                                import numpy as np
                                cluster = int(kits_clf.predict(np.asarray([kit_colors[0]], np.float32))[0])
                                left_cluster = color_lock.get("left_cluster", 0)
                                if cluster == left_cluster:
                                    draw_label = 2  # Left GK
                                else:
                                    draw_label = 3  # Right GK
                    except Exception:
                        pass
        elif raw_label == RAW.get("ball"):
            draw_label = 4
        elif raw_label == RAW.get("main_ref"):
            draw_label = 5
        elif raw_label == RAW.get("side_ref"):
            draw_label = 6
        elif raw_label == RAW.get("staff"):
            draw_label = 7
        
        drawn.append({"bbox": (x1, y1, x2, y2), "raw": raw_label, "draw": draw_label, "conf": conf})
    
    return drawn


def _update_ref_state(drawn, main_ref_lock_t, side_ref_lock_t):
    """Update referee tracking state."""
    main_ix = [i for i, d in enumerate(drawn) if d["draw"] == 5]
    side_ix = [i for i, d in enumerate(drawn) if d["draw"] == 6]
    
    last_main_ref_box = drawn[main_ix[0]]["bbox"] if main_ix else None
    last_side_ref_boxes = [drawn[i]["bbox"] for i in side_ix[:2]]
    
    main_ref_lock_t = MAIN_REF_LOCK_FRAMES if main_ix else max(0, main_ref_lock_t - 1)
    side_ref_lock_t = SIDE_REF_LOCK_FRAMES if side_ix else max(0, side_ref_lock_t - 1)
    
    return main_ref_lock_t, side_ref_lock_t, last_main_ref_box, last_side_ref_boxes


def _draw_debug_overlay(frame, kits_clf, color_lock, grass_hsv, drawn, ref_mu_lab, centroids_bgr):
    """Draw debug info overlay."""
    n_main = sum(1 for d in drawn if d["draw"] == 5)
    n_side = sum(1 for d in drawn if d["draw"] == 6)
    left_cluster = color_lock["left_cluster"] if color_lock else -1
    grass_h = int(grass_hsv[0,0,0]) if grass_hsv is not None else -1
    
    cv2.putText(frame, 
        f"KITCLF={'ON' if kits_clf else 'OFF'} CL_LEFT={left_cluster} GH={grass_h} "
        f"MAIN={n_main} SIDE={n_side}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


def _iou(a, b):
    """Calculate IoU between two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)
def _postprocess_ball_ownership(frames_list: list, fps: float) -> list:
    """
    Post-process frames to interpolate missing ball detections and recalculate ownership.
    
    This handles cases where the ball disappears behind players for extended periods.
    After interpolation, ownership is recalculated using a fresh BallOwnershipProcessor.
    
    Args:
        frames_list: List of FrameRecord objects
        fps: Video FPS for ownership processor
        
    Returns:
        Updated frames_list with interpolated ball and recalculated ownership
    """
    import numpy as np
    from collections import deque, Counter
    
    # Step 1: Ball interpolation with extended max_gap
    # Fill gaps up to 50 frames (~2 seconds at 25fps)
    frames_list = interpolate_ball_positions(
        frames_list,
        max_gap=50,
        apply_smoothing=True,
        smoothing_window=5,
        smoothing_threshold=50.0
    )
    
    # Step 2: Recalculate ball ownership with improved parameters
    # Use larger close_distance_threshold to catch fast-moving balls near players
    ownership_processor = BallOwnershipProcessor(
        fps=fps,
        smoothing_window=0.5,           # Slightly shorter window for faster response
        distance_threshold=150,
        ball_speed_threshold=12.0,      # Increased from 8.0 - allow faster balls
        close_distance_threshold=70.0,  # Increased from 40.0 - catch balls near players
        extreme_speed_threshold=25.0    # Şut seviyesinde hız - bu hızda ownership verilmez
    )
    
    # Count stats
    interpolated_balls = 0
    ownership_changes = 0
    
    for frame in frames_list:
        # Find ball track
        ball_track = None
        for track in frame.tracks:
            if track.cls == 'ball' or track.track_id == 'ball':
                ball_track = track
                break
        
        # Check if this is an interpolated ball (no original score or very low)
        if ball_track and (ball_track.score is None or ball_track.score < 0.1):
            interpolated_balls += 1
        
        # Build tracks list for ownership processor
        # Need to create a simple object with required attributes
        class SimpleTrack:
            def __init__(self, track):
                self.track_id = track.track_id
                self.cls = track.cls
                self.bbox = track.bbox
        
        simple_tracks = [SimpleTrack(t) for t in frame.tracks]
        
        # Calculate ownership
        new_owner_id, ball_dist = ownership_processor.update(simple_tracks)
        new_owner_team = None
        
        if new_owner_id is not None:
            new_owner_team = get_team_from_id(new_owner_id)
        
        # Track changes
        if frame.ball_owner != new_owner_id:
            ownership_changes += 1
        
        # Update frame
        frame.ball_owner = new_owner_id
        frame.ball_owner_team = new_owner_team
    
    print(f"    📊 Interpolated {interpolated_balls} ball positions")
    print(f"    📊 Updated {ownership_changes} ownership assignments")
    
    return frames_list
