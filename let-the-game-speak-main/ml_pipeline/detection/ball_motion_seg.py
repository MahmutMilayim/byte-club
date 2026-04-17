import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any

# Dosya Yolları
DEFAULT_INPUT_FILE = "output/test_frames.jsonl"
DEFAULT_OUTPUT_FILE = "output/ball_segments.json"

# ═══════════════════════════════════════════════════════════════════════════════
# FUTBOL SAHASI SABİTLERİ (FIFA Standart: 105m x 68m)
# ═══════════════════════════════════════════════════════════════════════════════
FIELD_LENGTH = 105.0  # metre
FIELD_WIDTH = 68.0    # metre
HALF_FIELD = 52.5     # Orta saha çizgisi (metre)

# Kale ölçüleri
GOAL_Y_MIN = (FIELD_WIDTH - 7.32) / 2   # ~30.34m (kale üst direği)
GOAL_Y_MAX = (FIELD_WIDTH + 7.32) / 2   # ~37.66m (kale alt direği)

# Şut kabul aralığı (isabetsiz şutlar dahil)
SHOOT_Y_MIN = 12.0   # Şut bölgesi üst sınırı (metre) - genişletildi
SHOOT_Y_MAX = 56.0   # Şut bölgesi alt sınırı (metre) - genişletildi

# Yakın/uzak şut mesafe sınırları
CLOSE_SHOT_DISTANCE = 35.0  # metre - bu mesafeden yakınsa "yakın şut"

# ═══════════════════════════════════════════════════════════════════════════════
# KENAR ÇİZGİSİ KURALI (V5) - Sideline Rule
# Şut genellikle sahanın iç kısmından atılır, kenar çizgisinden değil
# ═══════════════════════════════════════════════════════════════════════════════
SIDELINE_MARGIN = 8.0  # metre - bu mesafeden yakınsa kenar çizgisi

# ═══════════════════════════════════════════════════════════════════════════════
# GERİ DÖNÜŞ KURALI (V5) - Bounce/Save Rule
# Top kaleye yakın bitip geri dönüyorsa = şut (kaleci kurtardı veya direkten döndü)
# ═══════════════════════════════════════════════════════════════════════════════
BOUNCE_NEAR_GOAL_X = 15.0  # metre - kale çizgisine bu kadar yakınsa "kaleye yakın"
BOUNCE_GOAL_ZONE_Y_MIN = 20.0  # metre - kale bölgesi Y alt sınırı
BOUNCE_GOAL_ZONE_Y_MAX = 48.0  # metre - kale bölgesi Y üst sınırı
BOUNCE_MIN_SPEED = 10.0  # px/frame - minimum hız (yavaş hareketleri filtrele)


def get_ball_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def is_shot_or_long_pass(start_x: float, start_y: float, end_x: float, end_y: float, verbose: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """
    futbol.html'deki algoritmayı Python'a çevirdik.
    
    Verilen başlangıç ve bitiş 2D saha koordinatlarına göre şut mu uzun pas mı belirler.
    
    KURALLAR:
    1. Hedef Y koordinatı 20-50m arasında olmalı (kale bölgesi)
    2. Y yönü kaleye doğru olmalı (top kale hizasında değilse)
    3. X yönü kaleye doğru olmalı
    4. Hedef X koordinatı şut sınırı içinde olmalı (mesafeye göre değişir)
    
    Args:
        start_x: Topun başlangıç X koordinatı (metre, 0-105)
        start_y: Topun başlangıç Y koordinatı (metre, 0-68)
        end_x: Hedefin X koordinatı (metre, 0-105)
        end_y: Hedefin Y koordinatı (metre, 0-68)
        verbose: Detaylı log yazdır
        
    Returns:
        (is_shot, reason, log_dict): (True/False, açıklama, detaylı log dictionary)
    """
    # Log dictionary oluştur
    log = {
        "coordinates": {
            "start_x": round(start_x, 1),
            "start_y": round(start_y, 1),
            "end_x": round(end_x, 1),
            "end_y": round(end_y, 1)
        },
        "rules": {},
        "result": None,
        "reason": None
    }
    
    if verbose:
        print(f"\n    📐 ŞUT/LONG_PASS ANALİZİ:")
        print(f"       Başlangıç: ({start_x:.1f}m, {start_y:.1f}m)")
        print(f"       Hedef:     ({end_x:.1f}m, {end_y:.1f}m)")
    
    # ─────────────────────────────────────────────────────────────
    # KURAL 0: SAHA DIŞI KONTROLÜ (end_x < 0 veya > 105 ise ŞUT!)
    # Top kale çizgisinden çıktıysa = şut (kaleciye çarpma, aut, gol)
    # ─────────────────────────────────────────────────────────────
    ball_crossed_left_goal = end_x < 0
    ball_crossed_right_goal = end_x > FIELD_LENGTH  # 105m
    
    if ball_crossed_left_goal or ball_crossed_right_goal:
        target_goal = "SOL" if ball_crossed_left_goal else "SAĞ"
        reason = f"{target_goal} KALE ÇİZGİSİNDEN ÇIKTI (end_x={end_x:.1f}m)"
        
        log["rules"]["rule0_ball_crossed_endline"] = {
            "passed": True,
            "description": f"Top {target_goal.lower()} kale çizgisinden çıktı",
            "end_x": round(end_x, 1)
        }
        log["result"] = "SHOT_CANDIDATE"
        log["reason"] = reason
        log["shot_type"] = "saha_dışı"
        log["target_goal"] = target_goal
        
        if verbose:
            print(f"       [KURAL 0] Top saha dışına çıktı (end_x={end_x:.1f}m) → ✅ ŞUT!")
            print(f"       🟢 SONUÇ: SHOT_CANDIDATE - {reason}")
        
        return True, reason, log
    
    # ─────────────────────────────────────────────────────────────
    # KURAL 0.5: KENAR ÇİZGİSİ KONTROLÜ (V5 - Sideline Rule)
    # Şut genellikle sahanın iç kısmından atılır
    # Kenar çizgisine çok yakın (start_y < 8 veya > 60) ise şut olamaz
    # ─────────────────────────────────────────────────────────────
    is_near_sideline = start_y < SIDELINE_MARGIN or start_y > (FIELD_WIDTH - SIDELINE_MARGIN)
    
    log["rules"]["rule0_5_sideline"] = {
        "passed": not is_near_sideline,
        "description": f"Başlangıç Y ({start_y:.1f}m) kenar çizgisine yakın değil",
        "sideline_margin": SIDELINE_MARGIN,
        "actual_start_y": round(start_y, 1),
        "field_width": FIELD_WIDTH
    }
    
    if verbose:
        status = "✅" if not is_near_sideline else "❌"
        print(f"       [KURAL 0.5] Başlangıç Y ({start_y:.1f}m) kenar çizgisinden uzak ({SIDELINE_MARGIN}m < Y < {FIELD_WIDTH - SIDELINE_MARGIN}m) → {status}")
    
    if is_near_sideline:
        reason = f"Kenar çizgisinden şut atılmaz (start_y={start_y:.1f}m)"
        log["result"] = "LONG_PASS"
        log["reason"] = reason
        log["failed_at_rule"] = "0.5"
        if verbose:
            print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
        return False, reason, log
    
    # ─────────────────────────────────────────────────────────────
    # KURAL 1: HEDEF Y KOORDİNATI KONTROLÜ (12m - 56m arası olmalı)
    # ─────────────────────────────────────────────────────────────
    target_in_shoot_zone = SHOOT_Y_MIN <= end_y <= SHOOT_Y_MAX
    
    log["rules"]["rule1_target_y_in_zone"] = {
        "passed": bool(target_in_shoot_zone),
        "description": f"Hedef Y ({end_y:.1f}m) ∈ [{SHOOT_Y_MIN}-{SHOOT_Y_MAX}m]",
        "required_range": [SHOOT_Y_MIN, SHOOT_Y_MAX],
        "actual_value": round(end_y, 1)
    }
    
    if verbose:
        status = "✅" if target_in_shoot_zone else "❌"
        print(f"       [KURAL 1] Hedef Y ({end_y:.1f}m) ∈ [{SHOOT_Y_MIN}-{SHOOT_Y_MAX}m] → {status}")
    
    if not target_in_shoot_zone:
        reason = f"Hedef Y ({end_y:.1f}m) şut aralığında değil ({SHOOT_Y_MIN}-{SHOOT_Y_MAX}m)"
        log["result"] = "LONG_PASS"
        log["reason"] = reason
        log["failed_at_rule"] = 1
        if verbose:
            print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
        return False, reason, log
    
    # ─────────────────────────────────────────────────────────────
    # KURAL 2: Y YÖNÜ KONTROLÜ
    # ─────────────────────────────────────────────────────────────
    y_direction_ok = True
    y_reason = ""
    y_description = "Top kale hizasında, Y yönü serbest"
    
    if start_y < SHOOT_Y_MIN:
        # Top şut bölgesinin üstünde - aşağı gitmeli
        y_description = f"Top üstte ({start_y:.1f}m), Y aşağı gitmeli"
        if end_y < start_y:
            y_direction_ok = False
            y_reason = f"Y yönü hatalı: Top üstte ({start_y:.1f}m), hedef yukarı ({end_y:.1f}m)"
    elif start_y > SHOOT_Y_MAX:
        # Top şut bölgesinin altında - yukarı gitmeli
        y_description = f"Top altta ({start_y:.1f}m), Y yukarı gitmeli"
        if end_y > start_y:
            y_direction_ok = False
            y_reason = f"Y yönü hatalı: Top altta ({start_y:.1f}m), hedef aşağı ({end_y:.1f}m)"
    
    log["rules"]["rule2_y_direction"] = {
        "passed": bool(y_direction_ok),
        "description": y_description,
        "start_y": round(start_y, 1),
        "end_y": round(end_y, 1)
    }
    
    if verbose:
        status = "✅" if y_direction_ok else "❌"
        if start_y < SHOOT_Y_MIN:
            print(f"       [KURAL 2] Top üstte, Y aşağı gitmeli (start={start_y:.1f}m, end={end_y:.1f}m) → {status}")
        elif start_y > SHOOT_Y_MAX:
            print(f"       [KURAL 2] Top altta, Y yukarı gitmeli (start={start_y:.1f}m, end={end_y:.1f}m) → {status}")
        else:
            print(f"       [KURAL 2] Top kale hizasında, Y yönü serbest → ✅")
    
    if not y_direction_ok:
        log["result"] = "LONG_PASS"
        log["reason"] = y_reason
        log["failed_at_rule"] = 2
        if verbose:
            print(f"       🔴 SONUÇ: LONG_PASS - {y_reason}")
        return False, y_reason, log
    
    # ─────────────────────────────────────────────────────────────
    # KURAL 3: X YÖNÜ VE KALE SEÇİMİ + MESAFE KONTROLÜ
    # ─────────────────────────────────────────────────────────────
    
    if start_x < HALF_FIELD:
        # ═══ SOL YARI SAHADA → SOL KALEYE ŞUT ═══
        target_goal = "SOL"
        if verbose:
            print(f"       [KURAL 3] Sol yarıda ({start_x:.1f}m < {HALF_FIELD}m) → SOL KALE hedef")
        
        # X yönü kontrolü: Sola gitmeli (X azalmalı)
        x_direction_ok = end_x < start_x
        
        log["rules"]["rule3a_x_direction"] = {
            "passed": bool(x_direction_ok),
            "description": f"Sol yarıda, X sola gitmeli (end < start)",
            "target_goal": "SOL",
            "start_x": round(start_x, 1),
            "end_x": round(end_x, 1),
            "condition": "end_x < start_x"
        }
        
        if verbose:
            status = "✅" if x_direction_ok else "❌"
            print(f"       [KURAL 3a] X sola gitmeli (end={end_x:.1f}m < start={start_x:.1f}m) → {status}")
        
        if not x_direction_ok:
            reason = f"X yönü hatalı: Sol yarıda ({start_x:.1f}m) ama sola gitmiyor ({end_x:.1f}m)"
            log["result"] = "LONG_PASS"
            log["reason"] = reason
            log["failed_at_rule"] = "3a"
            if verbose:
                print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
            return False, reason, log
        
        # Mesafeye göre şut sınırı
        is_close = start_x <= CLOSE_SHOT_DISTANCE
        shoot_threshold = 10.0 if is_close else 5.0
        x_threshold_ok = end_x <= shoot_threshold
        shot_type = "yakın" if is_close else "uzak"
        
        log["rules"]["rule3b_x_threshold"] = {
            "passed": bool(x_threshold_ok),
            "description": f"{shot_type} şut: Hedef X ≤ {shoot_threshold}m",
            "shot_type": shot_type,
            "threshold": shoot_threshold,
            "actual_end_x": round(end_x, 1),
            "condition": f"end_x <= {shoot_threshold}"
        }
        
        if verbose:
            status = "✅" if x_threshold_ok else "❌"
            print(f"       [KURAL 3b] {shot_type} şut: Hedef X ({end_x:.1f}m) ≤ {shoot_threshold}m → {status}")
        
        if not x_threshold_ok:
            reason = f"Hedef X ({end_x:.1f}m) sol kale sınırına ({shoot_threshold}m) yeterince yakın değil"
            log["result"] = "LONG_PASS"
            log["reason"] = reason
            log["failed_at_rule"] = "3b"
            if verbose:
                print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
            return False, reason, log
        
        result_reason = f"SOL KALEYE {shot_type.upper()} ŞUT"
        log["result"] = "SHOT_CANDIDATE"
        log["reason"] = result_reason
        log["shot_type"] = shot_type
        log["target_goal"] = "SOL"
        if verbose:
            print(f"       🟢 SONUÇ: SHOT_CANDIDATE - {result_reason}")
        return True, result_reason, log
        
    else:
        # ═══ SAĞ YARI SAHADA → SAĞ KALEYE ŞUT ═══
        target_goal = "SAĞ"
        if verbose:
            print(f"       [KURAL 3] Sağ yarıda ({start_x:.1f}m >= {HALF_FIELD}m) → SAĞ KALE hedef")
        
        # X yönü kontrolü: Sağa gitmeli (X artmalı)
        x_direction_ok = end_x > start_x
        
        log["rules"]["rule3a_x_direction"] = {
            "passed": bool(x_direction_ok),
            "description": f"Sağ yarıda, X sağa gitmeli (end > start)",
            "target_goal": "SAĞ",
            "start_x": round(start_x, 1),
            "end_x": round(end_x, 1),
            "condition": "end_x > start_x"
        }
        
        if verbose:
            status = "✅" if x_direction_ok else "❌"
            print(f"       [KURAL 3a] X sağa gitmeli (end={end_x:.1f}m > start={start_x:.1f}m) → {status}")
        
        if not x_direction_ok:
            reason = f"X yönü hatalı: Sağ yarıda ({start_x:.1f}m) ama sağa gitmiyor ({end_x:.1f}m)"
            log["result"] = "LONG_PASS"
            log["reason"] = reason
            log["failed_at_rule"] = "3a"
            if verbose:
                print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
            return False, reason, log
        
        # Mesafeye göre şut sınırı
        is_close = start_x >= (FIELD_LENGTH - CLOSE_SHOT_DISTANCE)  # 70m
        shoot_threshold = 95.0 if is_close else 100.0
        x_threshold_ok = end_x >= shoot_threshold
        shot_type = "yakın" if is_close else "uzak"
        
        log["rules"]["rule3b_x_threshold"] = {
            "passed": bool(x_threshold_ok),
            "description": f"{shot_type} şut: Hedef X ≥ {shoot_threshold}m",
            "shot_type": shot_type,
            "threshold": shoot_threshold,
            "actual_end_x": round(end_x, 1),
            "condition": f"end_x >= {shoot_threshold}"
        }
        
        if verbose:
            status = "✅" if x_threshold_ok else "❌"
            print(f"       [KURAL 3b] {shot_type} şut: Hedef X ({end_x:.1f}m) ≥ {shoot_threshold}m → {status}")
        
        if not x_threshold_ok:
            reason = f"Hedef X ({end_x:.1f}m) sağ kale sınırına ({shoot_threshold}m) yeterince yakın değil"
            log["result"] = "LONG_PASS"
            log["reason"] = reason
            log["failed_at_rule"] = "3b"
            if verbose:
                print(f"       🔴 SONUÇ: LONG_PASS - {reason}")
            return False, reason, log
        
        result_reason = f"SAĞ KALEYE {shot_type.upper()} ŞUT"
        log["result"] = "SHOT_CANDIDATE"
        log["reason"] = result_reason
        log["shot_type"] = shot_type
        log["target_goal"] = "SAĞ"
        if verbose:
            print(f"       🟢 SONUÇ: SHOT_CANDIDATE - {result_reason}")
        return True, result_reason, log

# --- ANA İŞLEM FONKSİYONU ---
def process_motion_segmentation(input_file=DEFAULT_INPUT_FILE, output_file=DEFAULT_OUTPUT_FILE):
    print("🔄 K3 Görevi Başlatılıyor (Motion Analysis)...")
    
    # 1. DOSYAYI YÜKLE
    frames = []
    with open(input_file, 'r') as f:
        for line in f:
            frames.append(json.loads(line))

    # 2. VERİYİ ÇIKAR (Extraction)
    # Hem piksel hem de 2D saha koordinatlarını çıkar
    data = []
    for frame in frames:
        tracks = frame.get('tracks', [])
        
        owner = frame.get('ball_owner', None)
        
        ball_track = next((t for t in tracks if t['cls'] in ['ball', 'sports ball']), None)
        
        if ball_track:
            cx, cy = get_ball_center(ball_track['bbox'])
            # 2D saha koordinatlarını da al (varsa)
            field_x = ball_track.get('field_x', None)
            field_y = ball_track.get('field_y', None)
            data.append({
                'frame': frame['frame_idx'], 
                'x': cx, 
                'y': cy, 
                'field_x': field_x,
                'field_y': field_y,
                'owner': owner
            })
        else:
            data.append({
                'frame': frame['frame_idx'], 
                'x': np.nan, 
                'y': np.nan, 
                'field_x': np.nan,
                'field_y': np.nan,
                'owner': owner
            })

    df = pd.DataFrame(data)
    # Guard against raw None values coming from JSON serialization.
    # Pandas diff() on object dtype with None can raise TypeError.
    df['field_x'] = pd.to_numeric(df['field_x'], errors='coerce')
    df['field_y'] = pd.to_numeric(df['field_y'], errors='coerce')
    
    # 4. FİZİKSEL HESAPLAMALAR
    df['x'] = df['x'].interpolate(method='linear').bfill().ffill()
    df['y'] = df['y'].interpolate(method='linear').bfill().ffill()
    df['dx'] = df['x'].diff().fillna(0)
    df['dy'] = df['y'].diff().fillna(0)
    df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed_smooth'] = df['speed'].rolling(window=5, center=True).mean().fillna(0)
    
    # V7.2: IŞINLANMA FİLTRESİ - Fiziksel olarak imkansız hareketleri tespit et
    # 2D saha koordinatlarında ardışık frame'ler arası mesafe kontrolü
    # En hızlı şut ~150 km/h = ~42 m/s, 25fps'de 1 frame = 40ms → max ~1.7m/frame
    # Güvenlik payı ile 3m/frame üstü = yanlış algılama (ışınlanma)
    MAX_FIELD_DISTANCE_PER_FRAME = 3.0  # metre
    
    df['field_dx'] = df['field_x'].diff().fillna(0)
    df['field_dy'] = df['field_y'].diff().fillna(0)
    df['field_speed'] = np.sqrt(df['field_dx']**2 + df['field_dy']**2)
    
    # Işınlanma olan frame'leri işaretle (field_x/y NaN değilse ve hız çok yüksekse)
    teleport_mask = (df['field_speed'] > MAX_FIELD_DISTANCE_PER_FRAME) & df['field_x'].notna() & df['field_y'].notna()
    teleport_count = teleport_mask.sum()
    
    if teleport_count > 0:
        print(f"    ⚠️ IŞINLANMA TESPİT EDİLDİ: {teleport_count} frame'de fiziksel olarak imkansız hareket")
        # Işınlanma olan frame'lerde 2D koordinatları NaN yap (yanlış algılama)
        df.loc[teleport_mask, 'field_x'] = np.nan
        df.loc[teleport_mask, 'field_y'] = np.nan

    # 5. SEGMENTASYON (Hassas Ayarlar - Quick Pass Detection)
    SPEED_THRESHOLD = 1.5  # Daha düşük eşik - yavaş pasları da yakala
    MIN_DISPLACEMENT = 15.0  # Daha düşük displacement - kısa pasları da yakala
    df['state'] = np.where(df['speed_smooth'] > SPEED_THRESHOLD, 'High', 'Low')
    
    # Owner değişikliklerini tespit et (None'dan gelen ve None'a giden geçişleri de dahil et)
    df['owner_changed'] = df['owner'].ne(df['owner'].shift()).fillna(False)
    
    # None olmayan owner değişikliklerini işaretle (gerçek pas geçişleri)
    df['real_owner_change'] = (df['owner_changed']) & (df['owner'].notna())
    
    segments = []
    current_segment = None
    
    for i, row in df.iterrows():
        state = row['state']
        owner_changed = row['owner_changed']
        real_owner_change = row['real_owner_change']
        
        if current_segment is None:
            # Segment başlatma koşulları:
            # 1. Hız yüksek (High state)
            # 2. VEYA gerçek owner değişimi oldu (None'dan ID'ye veya ID'den ID'ye)
            if state == 'High' or real_owner_change:
                current_segment = {'start_index': i}
        else:
            # Segment bitirme koşulları:
            # 1. Hız düşük oldu (Low state) VE bir süre geçti
            # 2. VEYA gerçek owner değişimi oldu (yeni bir pas başladı)
            # 3. VEYA son frame
            
            # Segment en az 2 frame sürmeli (çok kısa segmentleri önle)
            segment_duration = i - current_segment['start_index']
            
            if (state == 'Low' and segment_duration >= 2) or real_owner_change or i == len(df) - 1:
                start_idx = current_segment['start_index']
                end_idx = i - 1
                
                # Metrikler
                start_time = int(df.iloc[start_idx]['frame'])
                end_time = int(df.iloc[end_idx]['frame'])
                
                start_x, start_y = df.iloc[start_idx]['x'], df.iloc[start_idx]['y']
                end_x, end_y = df.iloc[end_idx]['x'], df.iloc[end_idx]['y']
                
                displacement = float(np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2))
                avg_speed = float(df.iloc[start_idx:end_idx+1]['speed_smooth'].mean())
                
                # Yön
                dir_x = (end_x - start_x) / displacement if displacement > 0 else 0
                dir_y = (end_y - start_y) / displacement if displacement > 0 else 0
                
                # Sahiplik - Başlangıç (Geliştirilmiş Lookback)
                # Kullanıcı isteği: "eğer null'dan geliyorsa bir önceki sahibi kimse topun o olsun"
                
                # 1. Önce yakın geçmişe bak (10 frame)
                lookback_idx = max(0, start_idx - 10)
                start_owners = df.iloc[lookback_idx:start_idx+1]['owner'].dropna()
                
                if not start_owners.empty:
                    start_owner = start_owners.iloc[-1]  # String ID (L3, R5, etc.)
                else:
                    # 2. Yakın geçmişte yoksa, daha geriye git (son geçerli sahibini bulana kadar)
                    # start_idx'ten 0'a kadar geriye doğru ara
                    prev_owners = df.iloc[0:start_idx]['owner'].dropna()
                    if not prev_owners.empty:
                        start_owner = prev_owners.iloc[-1]  # String ID
                    else:
                        # 3. Hiçbir geçmişte yoksa, segment içine bak (geleceğe)
                        segment_owners = df.iloc[start_idx:end_idx+1]['owner'].dropna()
                        start_owner = segment_owners.iloc[0] if not segment_owners.empty else None
                
                # Sahiplik - Bitiş (Sadece Forward - segment sonrasına bak)
                # İSTEK: Segment bitişinden sonra owner yoksa null kalsın
                lookforward_idx = min(len(df), end_idx + 10)
                end_owners = df.iloc[end_idx:lookforward_idx]['owner'].dropna()
                
                # Sadece forward'da bulursa kullan, bulamazsa NULL bırak
                end_owner = end_owners.iloc[0] if not end_owners.empty else None  # String ID

                # Filtrele ve Kaydet
                if displacement > MIN_DISPLACEMENT and (end_time - start_time) > 3:
                    # Segment türünü belirle
                    # HIZ TABANLI TURNOVER/PASS AYRIMI
                    HIGH_SPEED_THRESHOLD = 15.0  # px/frame
                    
                    # 2D saha koordinatlarını al (segment başlangıç ve bitiş)
                    start_field_x = df.iloc[start_idx]['field_x'] if pd.notna(df.iloc[start_idx]['field_x']) else None
                    start_field_y = df.iloc[start_idx]['field_y'] if pd.notna(df.iloc[start_idx]['field_y']) else None
                    end_field_x = df.iloc[end_idx]['field_x'] if pd.notna(df.iloc[end_idx]['field_x']) else None
                    end_field_y = df.iloc[end_idx]['field_y'] if pd.notna(df.iloc[end_idx]['field_y']) else None
                    
                    # 2D koordinat yoksa yakın frame'lerden bulmaya çalış
                    if start_field_x is None:
                        for offset in range(1, 6):
                            check_idx = start_idx + offset
                            if check_idx < len(df) and pd.notna(df.iloc[check_idx]['field_x']):
                                start_field_x = df.iloc[check_idx]['field_x']
                                start_field_y = df.iloc[check_idx]['field_y']
                                break
                    
                    if end_field_x is None:
                        for offset in range(1, 6):
                            check_idx = end_idx - offset
                            if check_idx >= 0 and pd.notna(df.iloc[check_idx]['field_x']):
                                end_field_x = df.iloc[check_idx]['field_x']
                                end_field_y = df.iloc[check_idx]['field_y']
                                break
                    
                    # Şut/uzun pas ayrımı için değişkenler
                    shot_analysis_reason = None
                    shot_analysis_log = None
                    
                    if start_owner is not None and end_owner is not None:
                        if start_owner == end_owner:
                            segment_type = "dribble"  # Top sürme (aynı oyuncu)
                        else:
                            # Takım değişikliği kontrolü
                            start_team = start_owner[0] if start_owner else None  # "L" veya "R"
                            end_team = end_owner[0] if end_owner else None  # "L" veya "R"
                            
                            if start_team != end_team:
                                # Takım değişti - ama hıza bak!
                                if avg_speed > HIGH_SPEED_THRESHOLD:
                                    segment_type = "pass"  # Yüksek hızda = pas (intercept edilmiş)
                                else:
                                    segment_type = "turnover_candidate"  # Düşük hızda = turnover adayı (sonra doğrulanacak)
                            else:
                                segment_type = "pass"  # Pas (aynı takım, farklı oyuncu)
                    elif start_owner is not None and end_owner is None:
                        # ═══════════════════════════════════════════════════════════════
                        # SHOT_CANDIDATE veya LONG_PASS AYRIMI (futbol.html algoritması)
                        # ═══════════════════════════════════════════════════════════════
                        # Top kayboldu - şut adayı mı uzun pas mı?
                        # 2D koordinatlar varsa futbol.html algoritmasını kullan
                        # NOT: Burada kesin şut DEĞİL, sadece ADAY belirliyoruz!
                        # Kesin şut tespiti shot_detection.py'da yapılacak.
                        
                        if start_field_x is not None and end_field_x is not None:
                            is_shot_candidate, reason, analysis_log = is_shot_or_long_pass(
                                start_field_x, start_field_y or FIELD_WIDTH/2,
                                end_field_x, end_field_y or FIELD_WIDTH/2
                            )
                            if is_shot_candidate:
                                segment_type = "shot_candidate"  # Şut ADAYI (kesin şut değil!)
                                shot_analysis_reason = reason
                                shot_analysis_log = analysis_log
                            else:
                                segment_type = "long_pass"  # Uzun pas (kaleye yönelmiyor)
                                shot_analysis_reason = reason
                                shot_analysis_log = analysis_log
                        else:
                            # 2D koordinat yok - yine de shot_candidate olarak işaretle
                            segment_type = "shot_candidate"
                            shot_analysis_reason = "2D koordinat yok, belirsiz"
                            shot_analysis_log = None
                    else:
                        segment_type = "unknown"  # Bilinmeyen
                    
                    segment_data = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_owner": start_owner,
                        "end_owner": end_owner,
                        "segment_type": segment_type,
                        "displacement": round(displacement, 2),
                        "direction_vector": [round(dir_x, 3), round(dir_y, 3)],
                        "average_speed": round(avg_speed, 2),
                        # 2D saha koordinatları
                        "start_field_x": round(start_field_x, 2) if start_field_x is not None else None,
                        "start_field_y": round(start_field_y, 2) if start_field_y is not None else None,
                        "end_field_x": round(end_field_x, 2) if end_field_x is not None else None,
                        "end_field_y": round(end_field_y, 2) if end_field_y is not None else None,
                    }
                    
                    # Şut analizi nedenini ekle (varsa)
                    if shot_analysis_reason:
                        segment_data["shot_analysis_reason"] = shot_analysis_reason
                    
                    # Şut analizi detaylı log'u ekle (varsa)
                    if shot_analysis_log is not None:
                        segment_data["shot_analysis_log"] = shot_analysis_log
                    
                    segments.append(segment_data)
                
                current_segment = None

    # ═══════════════════════════════════════════════════════════════════════════════
    # 5.5 POST-PROCESSING: GERİ DÖNÜŞ KURALI (V5 - Bounce/Save Rule)
    # Top kaleye yakın bitip geri dönüyorsa = şut (kaleci kurtardı veya direkten döndü)
    # Kural: pass/long_pass + kale bölgesinde bitiyor + geri dönüyor + hız > 10
    # ═══════════════════════════════════════════════════════════════════════════════
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        
        seg_type1 = seg1.get('segment_type', '')
        end_x1 = seg1.get('end_field_x')
        end_y1 = seg1.get('end_field_y')
        speed1 = seg1.get('average_speed', 0)
        
        start_x2 = seg2.get('start_field_x')
        end_x2 = seg2.get('end_field_x')
        
        # Gerekli veriler var mı?
        if end_x1 is None or end_y1 is None or start_x2 is None or end_x2 is None:
            continue
        
        # Koşul 1: İlk segment pass veya long_pass olmalı
        if seg_type1 not in ['pass', 'long_pass']:
            continue
        
        # Koşul 2: Kale bölgesinde bitiyor mu? (Y: 20-48)
        in_goal_zone = BOUNCE_GOAL_ZONE_Y_MIN <= end_y1 <= BOUNCE_GOAL_ZONE_Y_MAX
        if not in_goal_zone:
            continue
        
        # Koşul 3: Hız yeterli mi?
        if speed1 <= BOUNCE_MIN_SPEED:
            continue
        
        # Koşul 4: Kaleye yakın mı ve geri dönüyor mu?
        near_right_goal = end_x1 > (FIELD_LENGTH - BOUNCE_NEAR_GOAL_X)  # > 90m
        near_left_goal = end_x1 < BOUNCE_NEAR_GOAL_X  # < 15m
        
        bounced_from_right = near_right_goal and end_x2 < start_x2  # Sağdan sola döndü
        bounced_from_left = near_left_goal and end_x2 > start_x2  # Soldan sağa döndü
        
        if bounced_from_right or bounced_from_left:
            # ŞUT TESPİT EDİLDİ! (Kaleci kurtardı veya direkten döndü)
            target_goal = "SAĞ" if bounced_from_right else "SOL"
            segments[i]['segment_type'] = 'shot_candidate'
            segments[i]['shot_analysis_reason'] = f"{target_goal} KALEYE ŞUT (geri döndü - kaleci/direk)"
            segments[i]['shot_analysis_log'] = {
                "rule": "bounce_save_rule",
                "end_x": round(end_x1, 1),
                "end_y": round(end_y1, 1),
                "speed": round(speed1, 1),
                "bounced_back": True,
                "target_goal": target_goal
            }
            print(f"    🔄 GERİ DÖNÜŞ KURALI: Segment {i+1} shot_candidate olarak güncellendi (top {target_goal} kaleden geri döndü)")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 5.6 POST-PROCESSING: KALEYE ÇOK YAKIN ŞUT KURALI (V6 - Near Goal Shot Rule)
    # Top kale çizgisine çok yakın bitiyorsa (>94m veya <11m) ve uzun mesafeden geldiyse = şut
    # Kaleci/defans topu alsa bile bu bir şut denemesidir
    # 
    # FUTBOL MANTIĞI:
    # - Kale çizgisine 11m yakınlık (94m veya 11m) = şut bölgesi (ceza sahası sınırı)
    # - Minimum 12m mesafe = kısa pasları hariç tutar (kaleciye geri pas vs.)
    # - Y: 20-48m = kale bölgesi (kale direkleri 30-38m, ±10m tolerans)
    # - Cross'ları hariç tutar (Y < 20 veya Y > 48 = kanat bölgesi)
    # ═══════════════════════════════════════════════════════════════════════════════
    NEAR_GOAL_THRESHOLD_RIGHT = 94.0  # Sağ kale için minimum end_x (ceza sahası sınırı)
    NEAR_GOAL_THRESHOLD_LEFT = 11.0   # Sol kale için maksimum end_x (ceza sahası sınırı)
    NEAR_GOAL_MIN_DISTANCE = 12.0     # Minimum şut mesafesi (metre) - kısa pasları hariç tut
    NEAR_GOAL_Y_MIN = 20.0            # Kale bölgesi Y minimum (cross'ları hariç tut)
    NEAR_GOAL_Y_MAX = 48.0            # Kale bölgesi Y maksimum (cross'ları hariç tut)
    
    for i, seg in enumerate(segments):
        seg_type = seg.get('segment_type', '')
        
        # Zaten shot_candidate veya dribble ise atla
        if seg_type in ['shot_candidate', 'dribble']:
            continue
        
        end_x = seg.get('end_field_x')
        start_x = seg.get('start_field_x')
        end_y = seg.get('end_field_y')
        
        if end_x is None or start_x is None or end_y is None:
            continue
        
        # Kale bölgesinde mi?
        if not (NEAR_GOAL_Y_MIN <= end_y <= NEAR_GOAL_Y_MAX):
            continue
        
        distance = abs(end_x - start_x)
        
        # Mesafe yeterli mi?
        if distance <= NEAR_GOAL_MIN_DISTANCE:
            continue
        
        # Sağ kale kontrolü
        if end_x > NEAR_GOAL_THRESHOLD_RIGHT and start_x >= HALF_FIELD and start_x < end_x:
            segments[i]['segment_type'] = 'shot_candidate'
            segments[i]['shot_analysis_reason'] = f"SAĞ KALEYE YAKIN ŞUT (end_x={end_x:.1f}m, mesafe={distance:.1f}m)"
            segments[i]['shot_analysis_log'] = {
                "rule": "near_goal_shot_rule",
                "end_x": round(end_x, 1),
                "end_y": round(end_y, 1),
                "start_x": round(start_x, 1),
                "distance": round(distance, 1),
                "target_goal": "SAĞ"
            }
            print(f"    🎯 KALEYE YAKIN ŞUT: Segment {i+1} shot_candidate (end_x={end_x:.1f}m, mesafe={distance:.1f}m)")
        
        # Sol kale kontrolü
        elif end_x < NEAR_GOAL_THRESHOLD_LEFT and start_x < HALF_FIELD and start_x > end_x:
            segments[i]['segment_type'] = 'shot_candidate'
            segments[i]['shot_analysis_reason'] = f"SOL KALEYE YAKIN ŞUT (end_x={end_x:.1f}m, mesafe={distance:.1f}m)"
            segments[i]['shot_analysis_log'] = {
                "rule": "near_goal_shot_rule",
                "end_x": round(end_x, 1),
                "end_y": round(end_y, 1),
                "start_x": round(start_x, 1),
                "distance": round(distance, 1),
                "target_goal": "SOL"
            }
            print(f"    🎯 KALEYE YAKIN ŞUT: Segment {i+1} shot_candidate (end_x={end_x:.1f}m, mesafe={distance:.1f}m)")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 5.7 POST-PROCESSING: SEGMENT ARASI KALEYE IŞINLANMA KURALI (V7 - Teleport Goal Rule)
    # Top bir segmentten diğerine aniden kale bölgesine ışınlandıysa = ŞUT + GOL!
    # Örnek: Segment A end_x=11.8m → Segment B start_x=-0.1m (kale içi) = GOL!
    # Bu durum genellikle şu şekilde olur:
    # - Şut atıldı
    # - Top havada iken tespit edilemedi (YOLO kaybetti)
    # - Top kaleye girdi ve kaleci/ağda göründü
    # ═══════════════════════════════════════════════════════════════════════════════
    TELEPORT_GOAL_LEFT_THRESHOLD = 5.0   # Sol kale çizgisi (0m'den 5m'e kadar)
    TELEPORT_GOAL_RIGHT_THRESHOLD = 100.0  # Sağ kale çizgisi (100m'den 105m'e kadar)
    TELEPORT_MIN_JUMP = 8.0  # Minimum pozisyon atlama (metre)
    TELEPORT_GOAL_Y_MIN = 28.0  # Kale Y minimum
    TELEPORT_GOAL_Y_MAX = 40.0  # Kale Y maksimum
    
    print(f"    🔍 Teleport gol kontrolü: {len(segments)} segment analiz ediliyor...")
    
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        end_x1 = seg1.get('end_field_x')
        end_y1 = seg1.get('end_field_y')
        start_x2 = seg2.get('start_field_x')
        start_y2 = seg2.get('start_field_y')
        end_owner1 = seg1.get('end_owner')
        start_owner2 = seg2.get('start_owner')
        seg_type1 = seg1.get('segment_type', '')
        
        # Gerekli veriler var mı?
        if end_x1 is None or start_x2 is None or start_y2 is None:
            continue
        
        # Segment 1 zaten shot_candidate ise atla
        if seg_type1 == 'shot_candidate':
            continue
        
        # DEBUG: Her segment çiftini yazdır
        print(f"    [DEBUG] Seg {i+1}→{i+2}: end_x1={end_x1:.1f}m, start_x2={start_x2:.1f}m, start_y2={start_y2:.1f}m, type1={seg_type1}")
        
        # SOL KALE KONTROLÜ: Birden kale içine ışınlandı mı?
        if start_x2 < TELEPORT_GOAL_LEFT_THRESHOLD and TELEPORT_GOAL_Y_MIN <= start_y2 <= TELEPORT_GOAL_Y_MAX:
            jump_distance = end_x1 - start_x2
            if jump_distance >= TELEPORT_MIN_JUMP:
                # ŞUT + GOL TESPİT EDİLDİ!
                # Dribble segmentini de shot_candidate yap (şut çekilmeden önce topla ilerleme)
                segments[i]['segment_type'] = 'shot_candidate'
                segments[i]['shot_analysis_reason'] = f"SOL KALEYE ŞUT + GOL (top kaleye ışınlandı: {end_x1:.1f}m → {start_x2:.1f}m)"
                segments[i]['shot_analysis_log'] = {
                    "rule": "teleport_goal_rule",
                    "end_x_seg1": round(end_x1, 1),
                    "start_x_seg2": round(start_x2, 1),
                    "start_y_seg2": round(start_y2, 1),
                    "jump_distance": round(jump_distance, 1),
                    "target_goal": "SOL",
                    "is_goal": True,
                    "original_segment_type": seg_type1
                }
                segments[i]['teleport_goal'] = True  # Gol işareti
                print(f"    ⚽ IŞINLANMA GOLU: Segment {i+1} shot_candidate (top {end_x1:.1f}m → {start_x2:.1f}m kaleye ışınlandı)")
        
        # SAĞ KALE KONTROLÜ: Birden kale içine ışınlandı mı?
        elif start_x2 > TELEPORT_GOAL_RIGHT_THRESHOLD and TELEPORT_GOAL_Y_MIN <= start_y2 <= TELEPORT_GOAL_Y_MAX:
            jump_distance = start_x2 - end_x1
            if jump_distance >= TELEPORT_MIN_JUMP:
                # ŞUT + GOL TESPİT EDİLDİ!
                segments[i]['segment_type'] = 'shot_candidate'
                segments[i]['shot_analysis_reason'] = f"SAĞ KALEYE ŞUT + GOL (top kaleye ışınlandı: {end_x1:.1f}m → {start_x2:.1f}m)"
                segments[i]['shot_analysis_log'] = {
                    "rule": "teleport_goal_rule",
                    "end_x_seg1": round(end_x1, 1),
                    "start_x_seg2": round(start_x2, 1),
                    "start_y_seg2": round(start_y2, 1),
                    "jump_distance": round(jump_distance, 1),
                    "target_goal": "SAĞ",
                    "is_goal": True
                }
                segments[i]['teleport_goal'] = True  # Gol işareti
                print(f"    ⚽ IŞINLANMA GOLU: Segment {i+1} shot_candidate (top {end_x1:.1f}m → {start_x2:.1f}m kaleye ışınlandı)")

    # 6. POST-PROCESSING: endOwner'ları düzelt
    # Kural: Bir segment'in endOwner'ı, bir sonraki segment'in startOwner'ı olmalı
    # Eğer sonraki segment yoksa, segment bitişinden sonraki frame'lere bak
    for i in range(len(segments)):
        if i < len(segments) - 1:
            # Sonraki segment var
            next_start_owner = segments[i + 1].get('start_owner')
            if next_start_owner is not None:
                # Sonraki segment'in startOwner'ı varsa, bu segment'in endOwner'ı o olur
                segments[i]['end_owner'] = next_start_owner
            else:
                # Sonraki segment'in startOwner'ı null ise, bu segment'in endOwner'ı da null
                segments[i]['end_owner'] = None
        else:
            # SON SEGMENT - ŞUT TESPİTİ İÇİN ÖZEL KONTROL
            # Segment bitiminden sonraki frame'lerde owner'ın None olup olmadığına bak
            # Eğer çoğunlukla None ise = top oyunculardan uzaklaştı = ŞUT
            end_time = segments[i].get('end_time', 0)
            # Frame index bul (end_time frame numarası)
            end_frame_indices = df[df['frame'] == end_time].index
            if len(end_frame_indices) > 0:
                end_idx = end_frame_indices[0]
                # Segment bitiminden video sonuna kadar BAK (15 değil!)
                # Şut sonrası top genellikle uzun süre sahipsiz kalır
                remaining_frames = len(df) - end_idx
                lookforward_count = min(remaining_frames, 30)  # Max 30 frame ileriye bak
                
                if lookforward_count > 0:
                    end_owners_series = df.iloc[end_idx:end_idx + lookforward_count]['owner']
                    null_count = end_owners_series.isna().sum()
                    total_count = len(end_owners_series)
                    
                    # Eğer %70'den fazlası None ise = ŞUT (top uzaklaştı)
                    if null_count / total_count >= 0.7:
                        segments[i]['end_owner'] = None  # Şut olarak işaretle
                    else:
                        # Az sayıda None var, owner'ı bul
                        end_owners = end_owners_series.dropna()
                        if not end_owners.empty:
                            segments[i]['end_owner'] = end_owners.iloc[0]
                        else:
                            segments[i]['end_owner'] = None
                else:
                    segments[i]['end_owner'] = None
            else:
                segments[i]['end_owner'] = None
    
    # 7. POST-PROCESSING: segment_type'ları güncelle (endOwner değiştiği için)
    # TAKIM DEĞİŞİKLİĞİ KONTROLÜ:
    # - Takım değişikliği olan tüm segment'ler "team_change_candidate" olarak işaretlenir
    # - Sonra topu alan takımın pas yapıp yapmadığına bakılır
    
    for seg in segments:
        # ❗ ÖNEMLİ: Eğer önceki kurallar shot_candidate yaptıysa, korumaya al!
        # Sadece spesifik kuralları kontrol et (bounce rule ve near goal rule)
        reason = seg.get('shot_analysis_reason', '')
        if reason and ('geri döndü' in reason or 'KALEYE YAKIN ŞUT' in reason):
            seg['segment_type'] = "shot_candidate"  # Önceki kurallardan gelen şut adayı
            continue  # Bu segment'e dokunma!
        
        start_owner = seg.get('start_owner')
        end_owner = seg.get('end_owner')
        avg_speed = seg.get('average_speed', 0)
        
        if start_owner is not None and end_owner is not None:
            if start_owner == end_owner:
                seg['segment_type'] = "dribble"  # Top sürme (aynı oyuncu)
            else:
                # Takım değişikliği kontrolü
                start_team = start_owner[0] if start_owner else None  # "L" veya "R"
                end_team = end_owner[0] if end_owner else None  # "L" veya "R"
                
                if start_team != end_team:
                    # Takım değişti - topu alan takımın pas yapıp yapmadığını kontrol edeceğiz
                    seg['segment_type'] = "team_change_candidate"
                    seg['avg_speed_at_change'] = avg_speed
                else:
                    seg['segment_type'] = "pass"  # Pas (aynı takım, farklı oyuncu)
        elif start_owner is not None and end_owner is None:
            # ❗ ÖNEMLI: Eğer zaten long_pass olarak işaretlenmişse, shot_candidate'e çevirme!
            # is_shot_or_long_pass() fonksiyonu önceden karar verdi
            if seg.get('segment_type') != "long_pass":
                seg['segment_type'] = "shot_candidate"  # Şut adayı (top kayboldu)
        else:
            seg['segment_type'] = "unknown"  # Bilinmeyen
    
    # 7.5 TAKIM DEĞİŞİKLİĞİ DOĞRULAMA VE SEGMENT BİRLEŞTİRME
    # Mantık: Takım değişikliği olduğunda, topu alan takım (end_team) en az 1 pas yapmalı
    # Eğer topu alan takım pas yaparsa → turnover (gerçek top kaybı, diğer takım kontrolü ele geçirdi)
    # Eğer top hemen geri dönerse → geçici dokunuş, segment'leri birleştir (L2→R7→L9 = L2→L9)
    # Eğer son segment ise → turnover (video bittiği için doğrulayamıyoruz ama kabul ediyoruz)
    
    # Silinecek segment index'lerini takip et
    segments_to_remove = set()
    
    for i, seg in enumerate(segments):
        if seg.get('segment_type') == "team_change_candidate":
            start_owner = seg.get('start_owner')
            end_owner = seg.get('end_owner')
            start_team = start_owner[0] if start_owner else None  # Topu kaybeden takım
            end_team = end_owner[0] if end_owner else None  # Topu alan takım
            
            # Son segment mi?
            is_last_segment = (i == len(segments) - 1)
            
            if is_last_segment:
                # Son segment - video bittiği için turnover olarak kabul et
                seg['segment_type'] = "turnover"
                seg['turnover_reason'] = "video_end"
            else:
                # Sonraki segment'lere bak: topu alan takım (end_team) kendi içinde pas yaptı mı?
                confirmed_turnover = False
                merge_needed = False
                merge_target_idx = None
                
                # Sonraki segment'leri kontrol et
                for j in range(i + 1, len(segments)):
                    next_seg = segments[j]
                    next_start_owner = next_seg.get('start_owner')
                    next_end_owner = next_seg.get('end_owner')
                    
                    if next_start_owner is None:
                        continue
                    
                    next_start_team = next_start_owner[0] if next_start_owner else None
                    next_end_team = next_end_owner[0] if next_end_owner else None
                    
                    # Sonraki segment topu alan takımla (end_team) mı başlıyor?
                    if next_start_team == end_team:
                        # Dribble mi? (aynı oyuncu devam ediyor)
                        if next_start_owner == next_end_owner:
                            # Dribble - takım topu kontrol ediyor, devam et
                            continue
                        # Aynı takım içinde pas mı? (farklı oyuncu ama aynı takım)
                        elif next_end_owner is not None and next_start_team == next_end_team:
                            # Evet! Topu alan takım kendi içinde pas yaptı - turnover doğrulandı
                            confirmed_turnover = True
                            seg['turnover_reason'] = "team_pass_confirmed"
                            break
                        elif next_end_team is not None and next_start_team != next_end_team:
                            # Top tekrar el değiştirdi - topu alan takım pas yapamadı
                            # Bu geçici bir dokunuş, segment'leri birleştir
                            merge_needed = True
                            merge_target_idx = j
                            break
                        else:
                            # Bilinmeyen durum
                            break
                    else:
                        # Sonraki segment farklı takımla başlıyor - bu olmamalı normalde
                        merge_needed = True
                        merge_target_idx = j
                        break
                
                if confirmed_turnover:
                    seg['segment_type'] = "turnover"
                elif merge_needed and merge_target_idx is not None:
                    # Topu alan takım pas yapamadı - segment'leri birleştir
                    # Örnek: L2→R7 (team_change_candidate), R7→L9 (takım değişti)
                    # R7 topu aldı ama hemen L9'a kaybetti - R takımı pas yapamadı
                    # Sonuç: L2→L9 olmalı (R7 geçici dokunuş)
                    
                    next_seg = segments[merge_target_idx]
                    next_end_owner = next_seg.get('end_owner')
                    next_end_team = next_end_owner[0] if next_end_owner else None
                    
                    # Eğer top başlangıç takımına (start_team) geri döndüyse
                    if next_end_team == start_team:
                        # Segment'leri birleştir: Bu segment'in start_owner + sonraki segment'in end_owner
                        seg['end_owner'] = next_end_owner
                        seg['end_time'] = next_seg.get('end_time')
                        seg['segment_type'] = "pass"
                        seg['intercepted'] = True
                        seg['intercept_reason'] = "temporary_loss_recovered"
                        
                        # Sonraki segment'i sil
                        segments_to_remove.add(merge_target_idx)
                    else:
                        # Top farklı bir takıma gitti - intercept edilmiş pas
                        seg['segment_type'] = "pass"
                        seg['intercepted'] = True
                        seg['intercept_reason'] = "no_team_pass_after"
                else:
                    # Turnover doğrulanamadı ve birleştirme de yapılamadı
                    seg['segment_type'] = "pass"
                    seg['intercepted'] = True
                    seg['intercept_reason'] = "undetermined"
    
    # Silinecek segment'leri kaldır (sondan başa doğru)
    for idx in sorted(segments_to_remove, reverse=True):
        if idx < len(segments):
            del segments[idx]
    
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=4)
        
    print(f"✅ Dosya kaydedildi: {output_file}")
    
    # 8. TURNOVER'LARI AYRI DOSYAYA KAYDET
    turnovers = [seg for seg in segments if seg.get('segment_type') == 'turnover']
    
    # Turnover istatistikleri
    left_to_right_turnovers = [t for t in turnovers if t.get('start_owner', '')[0:1] == 'L']
    right_to_left_turnovers = [t for t in turnovers if t.get('start_owner', '')[0:1] == 'R']
    
    # Oyuncu bazlı top kaybı sayısı
    turnover_by_player = {}
    for t in turnovers:
        player = t.get('start_owner')
        if player:
            turnover_by_player[player] = turnover_by_player.get(player, 0) + 1
    
    turnover_output = {
        "total_turnovers": len(turnovers),
        "left_team_losses": len(left_to_right_turnovers),  # Sol takımın top kaybı
        "right_team_losses": len(right_to_left_turnovers),  # Sağ takımın top kaybı
        "turnovers_by_player": turnover_by_player,
        "turnovers": turnovers
    }
    
    # Turnover dosya yolunu oluştur
    turnover_output_file = output_file.replace('.json', '_turnovers.json')
    if turnover_output_file == output_file:
        turnover_output_file = output_file.replace('.json', '') + '_turnovers.json'
    
    with open(turnover_output_file, 'w') as f:
        json.dump(turnover_output, f, indent=4)
    
    print(f"⚽ Top Kaybı Raporu: {turnover_output_file}")
    print(f"   📊 Toplam top kaybı: {len(turnovers)}")
    print(f"   🔵 Sol takım top kaybı: {len(left_to_right_turnovers)}")
    print(f"   🔴 Sağ takım top kaybı: {len(right_to_left_turnovers)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = DEFAULT_INPUT_FILE
        output_file = DEFAULT_OUTPUT_FILE
    process_motion_segmentation(input_file, output_file)
