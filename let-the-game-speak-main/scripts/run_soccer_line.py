"""
Soccer Line - Tek komutla saha çizgisi tespiti ve çizimi + 2D görünüm
Usage: python scripts/run_soccer_line.py <video_path> [--output <output_path>]
"""
import sys
import os
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOCCER_LINE_PATH = os.path.join(PROJECT_ROOT, "soccer_line")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SOCCER_LINE_PATH)

# NOT: os.chdir artık burada yapılmıyor - _load_models içinde yapılacak

import cv2
import torch
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# YOLO for player/ball detection
from ultralytics import YOLO

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict
)

# Saha çizgileri (dünya koordinatları - metre)
LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],   # Sol ceza sahası üst
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], # Sol ceza sahası sağ
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],   # Sol ceza sahası alt
    [[88.5, 54.16, 0.], [105., 54.16, 0.]], # Sağ ceza sahası üst
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], # Sağ ceza sahası sol
    [[88.5, 13.84, 0.], [105., 13.84, 0.]], # Sağ ceza sahası alt
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],   # Sol kale üst çizgi
    [[0., 37.66, 0.], [0., 37.66, -2.44]],      # Sol kale sol direk
    [[0., 30.34, 0.], [0., 30.34, -2.44]],      # Sol kale sağ direk
    [[105., 37.66, -2.44], [105., 30.34, -2.44]], # Sağ kale üst çizgi
    [[105., 30.34, 0.], [105., 30.34, -2.44]],    # Sağ kale sol direk
    [[105., 37.66, 0.], [105., 37.66, -2.44]],    # Sağ kale sağ direk
    [[52.5, 0., 0.], [52.5, 68, 0.]],       # Orta saha çizgisi
    [[0., 68., 0.], [105., 68., 0.]],       # Üst çizgi
    [[0., 0., 0.], [0., 68., 0.]],          # Sol çizgi
    [[105., 0., 0.], [105., 68., 0.]],      # Sağ çizgi
    [[0., 0., 0.], [105., 0., 0.]],         # Alt çizgi
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],    # Sol kale alanı üst
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],   # Sol kale alanı sağ
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],    # Sol kale alanı alt
    [[99.5, 43.16, 0.], [105., 43.16, 0.]], # Sağ kale alanı üst
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]], # Sağ kale alanı sol
    [[99.5, 24.84, 0.], [105., 24.84, 0.]], # Sağ kale alanı alt
]

# Field Transform için import
ML_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "ml_pipeline")
sys.path.insert(0, ML_PIPELINE_PATH)
sys.path.insert(0, os.path.join(ML_PIPELINE_PATH, "core"))

try:
    from field_transform import FieldTransform
except ImportError:
    # Fallback - inline tanım
    class FieldTransform:
        FIELD_LENGTH = 105.0
        FIELD_WIDTH = 68.0
        def __init__(self):
            self.P = None
            self.H = None
        def set_projection_matrix(self, P, frame_size):
            self.P = P
            self.H = np.column_stack([P[:, 0], P[:, 1], P[:, 3]])
        def pixel_to_field(self, px, py):
            if self.H is None: return None
            try:
                H_inv = np.linalg.inv(self.H)
                pixel = np.array([px, py, 1.0])
                w = H_inv @ pixel
                if abs(w[2]) < 1e-6: return None
                return (w[0]/w[2] + 52.5, w[1]/w[2] + 34)
            except: return None


class SoccerLineProcessor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model_kp = None
        self.model_line = None
        self.yolo_model = None
        self.transform = T.Resize((540, 960))
        self.field_transform = FieldTransform()  # 2D dönüşüm için
        self._original_cwd = os.getcwd()  # Mevcut dizini kaydet
        self._load_models()
    
    def _load_models(self):
        """Model yükle"""
        print("🔄 Soccer Line modelleri yükleniyor...")
        
        # Soccer Line dizinine geç (relative path'ler için)
        os.chdir(SOCCER_LINE_PATH)
        
        try:
            # Config dosyaları
            cfg_kp = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
            cfg_line = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))
            
            # Weights (support both legacy SV_* and README/downloaded MV_* names)
            kp_candidates = [
                "weights/SV_kp",
                "weights/MV_kp",
            ]
            line_candidates = [
                "weights/SV_lines",
                "weights/MV_lines",
            ]
            weights_kp = next((p for p in kp_candidates if os.path.exists(p)), None)
            weights_line = next((p for p in line_candidates if os.path.exists(p)), None)
            if not weights_kp or not weights_line:
                tried = ", ".join(kp_candidates + line_candidates)
                raise FileNotFoundError(
                    f"Soccer Line weight dosyalari bulunamadi. Denenen yollar: {tried}. "
                    f"Once `python soccer_line/download_weights.py` calistirin."
                )
            print(f"📦 Soccer Line weights: kp={weights_kp}, line={weights_line}")
            
            # Keypoint model
            self.model_kp = get_cls_net(cfg_kp)
            state_kp = torch.load(weights_kp, map_location=self.device)
            self.model_kp.load_state_dict(state_kp)
            self.model_kp.to(self.device)
            self.model_kp.eval()
            
            # Line model
            self.model_line = get_cls_net_l(cfg_line)
            state_line = torch.load(weights_line, map_location=self.device)
            self.model_line.load_state_dict(state_line)
            self.model_line.to(self.device)
            self.model_line.eval()
            
            # YOLO model for player/ball detection
            yolo_path = os.path.join(PROJECT_ROOT, "weights", "last.pt")
            if os.path.exists(yolo_path):
                print("🔄 YOLO modeli yükleniyor...")
                self.yolo_model = YOLO(yolo_path)
                print("✅ YOLO modeli yüklendi!")
            else:
                print(f"⚠️ YOLO modeli bulunamadı: {yolo_path}")
            
            print("✅ Modeller yüklendi!")
        finally:
            # Her zaman orijinal dizine dön
            os.chdir(self._original_cwd)
    
    def detect_objects(self, frame):
        """
        Frame'deki oyuncuları ve topu tespit et
        Returns: list of detections
        """
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            
            if conf < 0.3:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Ayak pozisyonu (bbox'ın alt merkezi)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Alt kenar
            
            # Top için merkez kullan
            if 'ball' in cls_name.lower():
                foot_y = (y1 + y2) / 2
            
            detections.append({
                'class': cls_name,
                'bbox': [x1, y1, x2, y2],
                'foot': (foot_x, foot_y),
                'conf': conf
            })
        
        return detections
    
    def draw_tracks_on_2d(self, field, tracks, P, scale_x, scale_y, margin=0):
        """
        frames.jsonl'den gelen track'leri 2D sahaya çiz
        tracks: [{'track_id': 'R1', 'cls': 'player', 'bbox': [...], ...}, ...]
        
        KALECİ TAKIMI SAHA POZİSYONUNA GÖRE BELİRLENİR:
        - Sol yarı sahadaki kaleci -> L1 (mavi)
        - Sağ yarı sahadaki kaleci -> R1 (kırmızı)
        """
        for track in tracks:
            bbox = track.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = bbox
            cls_name = track.get('cls', '').lower()
            track_id = track.get('track_id', '').upper()
            
            # Ayak pozisyonu (bbox'ın alt merkezi)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # Alt kenar
            
            # Top için merkez kullan
            if 'ball' in cls_name:
                foot_y = (y1 + y2) / 2
            
            # Piksel -> Saha koordinatı
            field_pos = self.pixel_to_field(P, foot_x, foot_y)
            if field_pos is None:
                continue
            
            fx, fy = field_pos
            
            # Saha koordinatı -> 2D piksel (margin offset ile)
            x2d = int((fx + margin) * scale_x)
            y2d = int(fy * scale_y)
            
            # Saha içinde mi?
            if not (0 <= x2d < field.shape[1] and 0 <= y2d < field.shape[0]):
                continue
            
            # KALECİ İÇİN SAHA POZİSYONUNA GÖRE TAKIM BELİRLE
            # L1 veya R1 ise kaleci - saha pozisyonuna göre takım
            original_track_id = track_id
            if track_id in ('L1', 'R1'):
                # Saha merkezi 52.5m - sol yarı = L, sağ yarı = R
                if fx < 52.5:
                    track_id = 'L1'  # Sol yarıdaki kaleci -> L takımı
                else:
                    track_id = 'R1'  # Sağ yarıdaki kaleci -> R takımı
            
            # Renk ve boyut belirle - TRACK_ID'DEN TAKIM BİLGİSİ
            if 'ball' in cls_name or track_id == 'BALL':
                color = (0, 255, 255)  # Sarı - TOP
                size = 6
                border_color = (0, 0, 0)
            elif track_id == 'L1':
                # L1 = Sol takım kalecisi = Mavi + Sarı
                color = (0, 255, 255)  # Sarı iç (BGR)
                size = 14
                border_color = (255, 0, 0)  # Mavi çerçeve
            elif track_id == 'R1':
                # R1 = Sağ takım kalecisi = Kırmızı + Sarı
                color = (0, 255, 255)  # Sarı iç (BGR)
                size = 14
                border_color = (0, 0, 255)  # Kırmızı çerçeve
            elif track_id.startswith('R'):
                # R = Right team = Kırmızı takım
                color = (0, 0, 255)  # Kırmızı (BGR)
                size = 10
                border_color = (255, 255, 255)
            elif track_id.startswith('L'):
                # L = Left team = Mavi takım
                color = (255, 0, 0)  # Mavi (BGR)
                size = 10
                border_color = (255, 255, 255)
            elif 'goalkeeper' in cls_name or 'keeper' in cls_name:
                color = (0, 200, 0)  # Yeşil - KALECİ
                size = 14
                border_color = (255, 255, 255)
            elif 'referee' in cls_name:
                color = (0, 0, 0)  # Siyah - HAKEM
                size = 10
                border_color = (0, 255, 255)
            else:
                color = (128, 128, 128)  # Gri
                size = 8
                border_color = (255, 255, 255)
            
            # Çiz
            cv2.circle(field, (x2d, y2d), size + 3, border_color, -1)
            cv2.circle(field, (x2d, y2d), size, color, -1)
            
            # Top için ekstra vurgu
            if 'ball' in cls_name or track_id == 'BALL':
                cv2.circle(field, (x2d, y2d), size + 6, (0, 255, 255), 3)
        
        return field
    
    def draw_detections_on_2d(self, field, detections, P, scale_x, scale_y, margin=0):
        """
        Tespit edilen oyuncuları/topu 2D sahaya çiz (YOLO direct detection)
        """
        for det in detections:
            foot_x, foot_y = det['foot']
            cls_name = det['class'].lower()
            
            # Piksel -> Saha koordinatı
            field_pos = self.pixel_to_field(P, foot_x, foot_y)
            if field_pos is None:
                continue
            
            fx, fy = field_pos
            
            # Saha koordinatı -> 2D piksel (margin offset ile)
            x2d = int((fx + margin) * scale_x)
            y2d = int(fy * scale_y)
            
            # Saha içinde mi?
            if not (0 <= x2d < field.shape[1] and 0 <= y2d < field.shape[0]):
                continue
            
            # Renk ve boyut belirle
            if 'ball' in cls_name:
                color = (0, 255, 255)  # Sarı - TOP
                size = 12
                border_color = (0, 0, 0)
            elif 'goalkeeper' in cls_name or 'keeper' in cls_name:
                color = (0, 200, 0)  # Yeşil - KALECİ
                size = 14
                border_color = (255, 255, 255)
            elif 'referee' in cls_name:
                color = (0, 0, 0)  # Siyah - HAKEM
                size = 10
                border_color = (0, 255, 255)
            elif 'player' in cls_name:
                color = (255, 0, 0)  # Mavi - OYUNCU
                size = 10
                border_color = (255, 255, 255)
            else:
                color = (128, 128, 128)  # Gri
                size = 8
                border_color = (255, 255, 255)
            
            # Çiz
            cv2.circle(field, (x2d, y2d), size + 3, border_color, -1)
            cv2.circle(field, (x2d, y2d), size, color, -1)
            
            # Top için ekstra vurgu
            if 'ball' in cls_name:
                cv2.circle(field, (x2d, y2d), size + 6, (0, 255, 255), 3)
        
        return field
    
    def get_projection_matrix(self, params):
        """Kamera parametrelerinden projection matrix hesapla"""
        cam_params = params["cam_params"]
        x_focal = cam_params['x_focal_length']
        y_focal = cam_params['y_focal_length']
        pp = np.array(cam_params['principal_point'])
        pos = np.array(cam_params['position_meters'])
        rot = np.array(cam_params['rotation_matrix'])
        
        It = np.eye(4)[:-1]
        It[:, -1] = -pos
        Q = np.array([
            [x_focal, 0, pp[0]],
            [0, y_focal, pp[1]],
            [0, 0, 1]
        ])
        P = Q @ (rot @ It)
        return P
    
    def project_point(self, P, world_point):
        """3D dünya noktasını 2D piksele dönüştür"""
        w = np.array([world_point[0] - 105/2, world_point[1] - 68/2, world_point[2], 1])
        p = P @ w
        if p[2] != 0:
            p = p / p[2]
        # Sonsuz veya çok büyük değerleri kontrol et
        x, y = p[0], p[1]
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return None
        if abs(x) > 1e6 or abs(y) > 1e6:
            return None
        return (int(x), int(y))
    
    def calibrate_frame(self, frame):
        """Tek frame için kamera kalibrasyonu yap"""
        h_orig, w_orig = frame.shape[:2]
        
        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = TF.to_tensor(pil_img).float().unsqueeze(0)
        
        if tensor.size()[-1] != 960:
            tensor = self.transform(tensor)
        
        tensor = tensor.to(self.device)
        b, c, h, w = tensor.size()
        
        # Inference
        with torch.no_grad():
            heatmaps_kp = self.model_kp(tensor)
            heatmaps_line = self.model_line(tensor)
        
        # Get keypoints and lines
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps_kp[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_line[:, :-1, :, :])
        
        kp_dict = coords_to_dict(kp_coords, threshold=0.3434)
        lines_dict = coords_to_dict(line_coords, threshold=0.7867)
        
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
        
        # Calibrate
        cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
        cam.update(kp_dict, lines_dict)
        params = cam.heuristic_voting(refine_lines=True)
        
        return params
    
    def create_2d_field(self, width=600, height=400):
        """
        Boş 2D saha görünümü oluştur
        Saha: 105m x 68m -> width x height piksel
        Kaleler için ekstra margin bırak
        """
        # Kale için margin (her iki tarafta 3m)
        margin = 3.0
        total_length = 105.0 + 2 * margin  # 111m
        total_width = 68.0
        
        field = np.zeros((height, width, 3), dtype=np.uint8)
        field[:] = (34, 139, 34)  # Yeşil zemin
        
        # Ölçek faktörleri (margin dahil)
        scale_x = width / total_length
        scale_y = height / total_width
        
        def to_2d(fx, fy):
            """Saha koordinatını 2D piksele çevir (margin offset ile)"""
            return (int((fx + margin) * scale_x), int(fy * scale_y))
        
        white = (255, 255, 255)
        
        # Dış çizgiler
        cv2.rectangle(field, to_2d(0, 0), to_2d(105, 68), white, 2)
        
        # Orta çizgi
        cv2.line(field, to_2d(52.5, 0), to_2d(52.5, 68), white, 2)
        
        # Orta daire
        center = to_2d(52.5, 34)
        radius = int(9.15 * scale_x)
        cv2.circle(field, center, radius, white, 2)
        cv2.circle(field, center, 3, white, -1)  # Merkez nokta
        
        # Sol ceza sahası (16.5m x 40.32m)
        cv2.rectangle(field, to_2d(0, 13.84), to_2d(16.5, 54.16), white, 2)
        
        # Sağ ceza sahası
        cv2.rectangle(field, to_2d(88.5, 13.84), to_2d(105, 54.16), white, 2)
        
        # Sol kale alanı (5.5m x 18.32m)
        cv2.rectangle(field, to_2d(0, 24.84), to_2d(5.5, 43.16), white, 2)
        
        # Sağ kale alanı
        cv2.rectangle(field, to_2d(99.5, 24.84), to_2d(105, 43.16), white, 2)
        
        # KALELER (7.32m genişlik, 2.44m derinlik) - BEYAZ DOLGULU
        # Sol kale
        cv2.rectangle(field, to_2d(-2.5, 30.34), to_2d(0, 37.66), (255, 255, 255), 2)  # Çerçeve
        cv2.rectangle(field, to_2d(-2.5, 30.34), to_2d(0, 37.66), (220, 220, 220), -1)  # İç dolgu
        # Sağ kale
        cv2.rectangle(field, to_2d(105, 30.34), to_2d(107.5, 37.66), (255, 255, 255), 2)
        cv2.rectangle(field, to_2d(105, 30.34), to_2d(107.5, 37.66), (220, 220, 220), -1)
        
        # Penaltı noktaları
        cv2.circle(field, to_2d(11, 34), 3, white, -1)  # Sol
        cv2.circle(field, to_2d(94, 34), 3, white, -1)  # Sağ
        
        return field, scale_x, scale_y, margin
    
    def draw_on_2d_field(self, field, positions, scale_x, scale_y, P=None):
        """
        2D saha üzerine pozisyonları çiz
        positions: [(pixel_x, pixel_y, color, size, label), ...]
        """
        field_copy = field.copy()
        
        for pos in positions:
            if len(pos) >= 4:
                px, py, color, size = pos[:4]
                label = pos[4] if len(pos) > 4 else None
                
                # Piksel -> Saha koordinatı
                if P is not None:
                    field_pos = self.pixel_to_field(P, px, py)
                    if field_pos:
                        fx, fy = field_pos
                        # Saha koordinatı -> 2D piksel
                        x2d = int(fx * scale_x)
                        y2d = int(fy * scale_y)
                        
                        # Saha içinde mi?
                        if 0 <= x2d < field_copy.shape[1] and 0 <= y2d < field_copy.shape[0]:
                            cv2.circle(field_copy, (x2d, y2d), size, color, -1)
                            if label:
                                cv2.putText(field_copy, label, (x2d + 5, y2d - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return field_copy
    
    def draw_camera_view_on_2d(self, field, P, frame_w, frame_h, scale_x, scale_y, margin=0):
        """
        Kameranın gördüğü alanı 2D sahada göster
        Ekranın 4 köşesini saha koordinatlarına çevirip dörtgen çiz
        """
        corners = [
            (0, frame_h),           # Sol alt
            (frame_w, frame_h),     # Sağ alt
            (frame_w, frame_h//2),  # Sağ orta
            (0, frame_h//2),        # Sol orta
        ]
        
        field_corners = []
        for px, py in corners:
            field_pos = self.pixel_to_field(P, px, py)
            if field_pos:
                fx, fy = field_pos
                x2d = int((fx + margin) * scale_x)
                y2d = int(fy * scale_y)
                # Sınırla
                x2d = max(0, min(field.shape[1]-1, x2d))
                y2d = max(0, min(field.shape[0]-1, y2d))
                field_corners.append((x2d, y2d))
        
        if len(field_corners) >= 3:
            pts = np.array(field_corners, np.int32)
            # Yarı saydam alan
            overlay = field.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 0))  # Sarı
            cv2.addWeighted(overlay, 0.3, field, 0.7, 0, field)
            # Kenar çizgisi
            cv2.polylines(field, [pts], True, (255, 200, 0), 2)
        
        return field
    
    def draw_field_lines(self, frame, P):
        """Saha çizgilerini çiz"""
        h, w = frame.shape[:2]
        
        for i, line in enumerate(LINES_COORDS):
            p1 = self.project_point(P, line[0])
            p2 = self.project_point(P, line[1])
            
            # None kontrol
            if p1 is None or p2 is None:
                continue
            
            # OpenCV clipLine kullan - otomatik ekran sınırlarına kırpar
            try:
                clipped, cp1, cp2 = cv2.clipLine((0, 0, w, h), p1, p2)
                if clipped:
                    cv2.line(frame, cp1, cp2, (0, 255, 0), 2)
            except:
                continue
        
        # Orta daire
        r = 9.15
        pts = []
        for ang in np.linspace(0, 360, 100):
            ang_rad = np.deg2rad(ang)
            world_pt = [52.5 + r * np.sin(ang_rad), 34 + r * np.cos(ang_rad), 0]
            p = self.project_point(P, world_pt)
            if p is not None:
                pts.append(p)
        
        # Daire noktalarını ekran içine filtrele ve çiz
        valid_pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
        if len(valid_pts) > 2:
            pts_arr = np.array(valid_pts, np.int32)
            cv2.polylines(frame, [pts_arr], False, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, input_path, output_path, sample_interval=1.0, frames_data=None, events=None, only_2d=False):
        """
        Video işle ve saha çizgilerini çiz
        1. Önce sample frame'lerden kalibrasyon yap
        2. Sonra tüm video'yu kalibrasyonlarla çiz (interpolasyon ile)
        
        frames_data: dict of {frame_idx: [tracks]} from frames.jsonl (for team colors)
        events: dict with 'shots' and 'passes' lists for overlay
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {input_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Sample interval'a göre frame atlama hesapla
        frame_skip = int(fps * sample_interval)
        sampled_frames = list(range(0, total_frames, frame_skip))
        
        print(f"🎬 Video: {width}x{height} @ {fps}fps, {total_frames} frames ({duration:.1f}s)")
        print(f"⏱️ Her {sample_interval}s'de 1 frame = {len(sampled_frames)} frame kalibre edilecek")
        
        # AŞAMA 1: Kalibrasyon frame'lerini topla
        print("\n📐 AŞAMA 1: Kalibrasyon yapılıyor...")
        calibrations = {}  # frame_num -> P matrix
        
        for frame_num in tqdm(sampled_frames, desc="Calibrating"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                params = self.calibrate_frame(frame)
                if params is not None:
                    P = self.get_projection_matrix(params)
                    calibrations[frame_num] = P
            except Exception as e:
                pass
        
        print(f"✅ {len(calibrations)}/{len(sampled_frames)} frame kalibre edildi")
        
        if len(calibrations) == 0:
            print("❌ Hiçbir frame kalibre edilemedi!")
            cap.release()
            return False
        
        # Kalibrasyonları sakla (2D dönüşüm için)
        self.last_calibrations = calibrations
        self.last_calib_frames = sorted(calibrations.keys())
        self.fps = fps
        
        # output_path yoksa sadece kalibrasyon yap, video oluşturma
        if output_path is None:
            print("✅ Kalibrasyon tamamlandı (video kaydedilmeyecek)")
            cap.release()
            return True
        
        # AŞAMA 2: Video çizimi
        def open_video_writer(path, fps, size):
            # Prefer H.264 for browser compatibility, fallback to mp4v.
            codecs_to_try = [
                ('avc1', 'H.264 (avc1)'),
                ('H264', 'H.264 (H264)'),
                ('X264', 'H.264 (X264)'),
                ('mp4v', 'MPEG-4 (mp4v)'),
            ]
            for codec, codec_name in codecs_to_try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(path, fourcc, fps, size)
                if writer.isOpened():
                    print(f"✅ Using video codec: {codec_name}")
                    return writer
                print(f"⚠️  {codec_name} codec not available, trying next...")
            return None

        if only_2d:
            print("\n🎨 AŞAMA 2: Sadece 2D saha videosu çiziliyor...")
            field_2d_height = height
            field_2d_width = int(field_2d_height * 105 / 68)
            out = open_video_writer(output_path, fps, (field_2d_width, field_2d_height))
            if out is None or not out.isOpened():
                print(f"❌ VideoWriter açılamadı: {output_path}")
                cap.release()
                return False
            field_template, scale_x, scale_y, margin = self.create_2d_field(field_2d_width, field_2d_height)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            calib_frames = sorted(calibrations.keys())
            for frame_num in tqdm(range(total_frames), desc="Drawing 2D"):
                ret, frame = cap.read()
                if not ret:
                    break
                P = self.get_interpolated_P(frame_num, calibrations, calib_frames)
                if P is not None:
                    field_2d = field_template.copy()
                    field_2d = self.draw_camera_view_on_2d(field_2d, P, width, height, scale_x, scale_y, margin)
                    detections = self.detect_objects(frame)
                    referees = [d for d in detections if 'referee' in d['class'].lower()]
                    if frames_data and (frame_num + 1) in frames_data:
                        tracks = frames_data[frame_num + 1]
                        field_2d = self.draw_tracks_on_2d(field_2d, tracks, P, scale_x, scale_y, margin)
                        if referees:
                            field_2d = self.draw_detections_on_2d(field_2d, referees, P, scale_x, scale_y, margin)
                    else:
                        field_2d = self.draw_detections_on_2d(field_2d, detections, P, scale_x, scale_y, margin)
                    if events:
                        _, field_2d = self.draw_event_overlay(frame, field_2d, frame_num + 1, events, fps)
                else:
                    field_2d = field_template.copy()
                out.write(field_2d)
            cap.release()
            out.release()
            import os
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"\n✅ 2D Video kaydedildi: {output_path}")
                print(f"📊 {total_frames} frame çizildi, {len(calibrations)} kalibrasyon kullanıldı")
                print(f"📦 Dosya boyutu: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"\n❌ 2D Video dosyası oluşturulamadı: {output_path}")
                return False
        else:
            print("\n🎨 AŞAMA 2: Tüm video çiziliyor (yan yana görünüm)...")
            field_2d_height = height
            field_2d_width = int(field_2d_height * 105 / 68)  # Saha oranı
            total_width = width + field_2d_width
            out = open_video_writer(output_path, fps, (total_width, height))
            if out is None or not out.isOpened():
                print(f"❌ VideoWriter açılamadı: {output_path}")
                cap.release()
                return False
            field_template, scale_x, scale_y, margin = self.create_2d_field(field_2d_width, field_2d_height)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            calib_frames = sorted(calibrations.keys())
            for frame_num in tqdm(range(total_frames), desc="Drawing"):
                ret, frame = cap.read()
                if not ret:
                    break
                P = self.get_interpolated_P(frame_num, calibrations, calib_frames)
                if P is not None:
                    frame = self.draw_field_lines(frame, P)
                    field_2d = field_template.copy()
                    field_2d = self.draw_camera_view_on_2d(field_2d, P, width, height, scale_x, scale_y, margin)
                    detections = self.detect_objects(frame)
                    referees = [d for d in detections if 'referee' in d['class'].lower()]
                    if frames_data and (frame_num + 1) in frames_data:
                        tracks = frames_data[frame_num + 1]
                        field_2d = self.draw_tracks_on_2d(field_2d, tracks, P, scale_x, scale_y, margin)
                        if referees:
                            field_2d = self.draw_detections_on_2d(field_2d, referees, P, scale_x, scale_y, margin)
                    else:
                        field_2d = self.draw_detections_on_2d(field_2d, detections, P, scale_x, scale_y, margin)
                    if events:
                        frame, field_2d = self.draw_event_overlay(frame, field_2d, frame_num + 1, events, fps)
                else:
                    field_2d = field_template.copy()
                combined = np.hstack([frame, field_2d])
                out.write(combined)
            cap.release()
            out.release()
            import os
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"\n✅ Video kaydedildi: {output_path}")
                print(f"📊 {total_frames} frame çizildi, {len(calibrations)} kalibrasyon kullanıldı")
                print(f"📦 Dosya boyutu: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"\n❌ Video dosyası oluşturulamadı: {output_path}")
                return False
    
    def get_interpolated_P(self, frame_num, calibrations, calib_frames):
        """
        Verilen frame için interpolasyon yaparak P matrix döndür
        İki kalibrasyon arasında lineer interpolasyon yapar
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
    
    def pixel_to_field(self, P, pixel_x, pixel_y):
        """
        Piksel koordinatını 2D saha koordinatına dönüştür
        
        Args:
            P: 3x4 projection matrix
            pixel_x, pixel_y: Piksel koordinatları
            
        Returns:
            (field_x, field_y) metre cinsinden veya None
        """
        # Homography: z=0 düzlemi için P'nin sütunlarını al
        H = np.column_stack([P[:, 0], P[:, 1], P[:, 3]])
        
        try:
            H_inv = np.linalg.inv(H)
            pixel = np.array([pixel_x, pixel_y, 1.0])
            world_h = H_inv @ pixel
            
            if abs(world_h[2]) < 1e-6:
                return None
            
            # Homojen -> kartezyen
            world_x = world_h[0] / world_h[2]
            world_y = world_h[1] / world_h[2]
            
            # Saha merkezini ekle (projection'da çıkarılmıştı)
            field_x = world_x + 105.0 / 2
            field_y = world_y + 68.0 / 2
            
            return (field_x, field_y)
        except:
            return None
    
    def draw_event_overlay(self, frame, field_2d, frame_idx, events, fps):
        """
        GOL ve PAS eventlerini video üzerine çiz
        
        Args:
            frame: Gerçek video frame'i
            field_2d: 2D saha görünümü
            frame_idx: Mevcut frame indexi (1-indexed)
            events: {'shots': [...], 'passes': [...]} formatında event listesi
            fps: Video FPS
            
        Returns:
            frame, field_2d: Çizilmiş frame'ler
        """
        # Event gösterim süresi (frame sayısı olarak)
        display_frames = int(fps * 2.0)  # 2 saniye göster
        
        h, w = frame.shape[:2]
        
        # --- SHOT EVENTS ---
        shots = events.get('shots', [])
        for shot in shots:
            shot_start_frame = shot.get('frame_idx', shot.get('frame', shot.get('segment_start_frame', 0)))
            goal_frame = shot.get('goal_frame', None)
            is_goal = shot.get('is_goal', False)
            shooter_id = shot.get('shooter_id', shot.get('segment_shooter_id', '?'))
            team = shot.get('shooter_team', shot.get('team', ''))
            start_pos = shot.get('start_pos', [0, 0])
            end_pos = shot.get('end_pos', [0, 0])
            target_side = shot.get('target_side', 'RIGHT')
            
            # Gösterim zamanı
            if is_goal and goal_frame:
                result_display_start = goal_frame
            else:
                result_display_start = shot_start_frame + int(fps * 1.0)
            result_display_end = result_display_start + int(fps * 2.5)
            
            if result_display_start <= frame_idx <= result_display_end:
                progress = (frame_idx - result_display_start) / (result_display_end - result_display_start)
                alpha = max(0.4, 1.0 - progress * 0.6)
                
                # GOL ise yeşil GOL, değilse sarı ŞUT
                if is_goal:
                    text = "GOL"
                    color = (0, 255, 0)  # Yeşil
                    bg_color = (0, 100, 0)
                else:
                    text = "SUT"
                    color = (0, 255, 255)  # Sarı
                    bg_color = (0, 100, 100)
                
                # --- GERÇEK VIDEO ÜZERİNE ÇİZ ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3.0
                thickness = 5
                
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                overlay = frame.copy()
                x1 = (w - text_w) // 2 - 30
                y1 = 40
                x2 = x1 + text_w + 60
                y2 = y1 + text_h + 50
                cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                cv2.putText(frame, text, ((w - text_w) // 2, y1 + text_h + 20), 
                           font, font_scale, color, thickness)
                
                # Ok çiz (şut yönü) - daha belirgin
                arrow_y = y2 + 40
                arrow_len = 120
                if target_side == "RIGHT":
                    cv2.arrowedLine(frame, (w // 2 - arrow_len, arrow_y), (w // 2 + arrow_len, arrow_y), 
                                   color, 6, tipLength=0.25)
                    # Ok gölgesi
                    cv2.arrowedLine(frame, (w // 2 - arrow_len + 2, arrow_y + 2), (w // 2 + arrow_len + 2, arrow_y + 2), 
                                   (0, 0, 0), 6, tipLength=0.25)
                else:
                    cv2.arrowedLine(frame, (w // 2 + arrow_len, arrow_y), (w // 2 - arrow_len, arrow_y), 
                                   color, 6, tipLength=0.25)
                    cv2.arrowedLine(frame, (w // 2 + arrow_len + 2, arrow_y + 2), (w // 2 - arrow_len + 2, arrow_y + 2), 
                                   (0, 0, 0), 6, tipLength=0.25)
                
                # --- 2D SAHA ---
                h2, w2 = field_2d.shape[:2]
                font_scale_2d = 1.2
                (tw2, th2), _ = cv2.getTextSize(text, font, font_scale_2d, 3)
                
                overlay_2d = field_2d.copy()
                cv2.rectangle(overlay_2d, (10, 10), (10 + tw2 + 30, 10 + th2 + 20), bg_color, -1)
                cv2.addWeighted(overlay_2d, alpha, field_2d, 1 - alpha, 0, field_2d)
                cv2.putText(field_2d, text, (20, 10 + th2 + 8), font, font_scale_2d, color, 3)
                
                # 2D sahada da ok çiz
                arrow_y_2d = 10 + th2 + 45
                arrow_len_2d = 60
                if target_side == "RIGHT":
                    cv2.arrowedLine(field_2d, (20, arrow_y_2d), (20 + arrow_len_2d, arrow_y_2d), 
                                   color, 3, tipLength=0.3)
                else:
                    cv2.arrowedLine(field_2d, (20 + arrow_len_2d, arrow_y_2d), (20, arrow_y_2d), 
                                   color, 3, tipLength=0.3)
        
        # --- PASS EVENTS ---
        passes = events.get('passes', [])
        pass_display_frames = int(fps * 1.5)  # 1.5 saniye göster
        for pas in passes:
            pass_frame = pas.get('frame', pas.get('start_frame', 0))
            passer_id = pas.get('passer_id', '?')
            receiver_id = pas.get('receiver_id', '?')
            team = pas.get('team', '')
            
            # Frame aralığında mı?
            if pass_frame <= frame_idx <= pass_frame + pass_display_frames:
                progress = (frame_idx - pass_frame) / pass_display_frames
                alpha = max(0.3, 1.0 - progress * 0.7)
                
                text = "PAS"
                color = (255, 200, 0)  # Sarı
                bg_color = (100, 80, 0)
                
                detail_text = f"#{passer_id} → #{receiver_id}"
                if team:
                    detail_text += f" ({team})"
                
                # --- GERÇEK VIDEO (sağ üst köşe) ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Sağ üst köşe
                x1 = w - text_w - 50
                y1 = 50
                x2 = w - 10
                y2 = y1 + text_h + 30
                
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                cv2.putText(frame, text, (x1 + 10, y1 + text_h + 10), 
                           font, font_scale, color, thickness)
                
                # Detail (altında)
                font_scale_small = 0.7
                (dw, dh), _ = cv2.getTextSize(detail_text, font, font_scale_small, 2)
                cv2.putText(frame, detail_text, (x1 + 10, y2 + dh + 5),
                           font, font_scale_small, (255, 255, 255), 2)
                
                # --- 2D SAHA (sağ üst köşe) ---
                h2, w2 = field_2d.shape[:2]
                font_scale_2d = 0.6
                (tw2, th2), _ = cv2.getTextSize(text, font, font_scale_2d, 2)
                
                x1_2d = w2 - tw2 - 30
                y1_2d = 10
                
                overlay_2d = field_2d.copy()
                cv2.rectangle(overlay_2d, (x1_2d, y1_2d), (w2 - 10, y1_2d + th2 + 15), bg_color, -1)
                cv2.addWeighted(overlay_2d, alpha, field_2d, 1 - alpha, 0, field_2d)
                
                cv2.putText(field_2d, text, (x1_2d + 5, y1_2d + th2 + 5), font, font_scale_2d, color, 2)
        
        return frame, field_2d

    def process_video_with_2d(self, input_path, output_path, sample_interval=1.0):
        """
        Video işle, 2D koordinatları da kaydet
        Returns: calibrations dict (frame_num -> P matrix)
        """
        # Normal process_video çağır
        self.process_video(input_path, output_path, sample_interval)
        
        # Kalibrasyon verilerini döndür
        return self.last_calibrations if hasattr(self, 'last_calibrations') else {}


def main():
    parser = argparse.ArgumentParser(description="Soccer Line - Saha çizgisi tespiti")
    parser.add_argument("video", help="Video dosyası")
    parser.add_argument("--output", "-o", help="Çıktı dosyası (varsayılan: output/<video>_field_lines.mp4)")
    parser.add_argument("--frames", "-f", help="frames.jsonl dosyası (takım bilgisi için)")
    parser.add_argument("--shot-events", help="shot_events.json dosyası")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--interval", "-i", type=float, default=1.0, 
                       help="Kaç saniyede bir frame al (varsayılan: 1.0)")
    parser.add_argument("--test-2d", action="store_true",
                       help="2D dönüşüm testi yap")
    
    args = parser.parse_args()
    
    # Input video
    video_path = args.video
    if not os.path.exists(video_path):
        video_path = os.path.join(PROJECT_ROOT, args.video)
    
    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {args.video}")
        return 1
    
    # frames.jsonl dosyası
    frames_data = None
    if args.frames:
        frames_path = args.frames
        if not os.path.exists(frames_path):
            frames_path = os.path.join(PROJECT_ROOT, args.frames)
        if os.path.exists(frames_path):
            import json
            frames_data = {}
            with open(frames_path, 'r') as f:
                for line in f:
                    frame = json.loads(line)
                    frames_data[frame['frame_idx']] = frame.get('tracks', [])
            print(f"📊 frames.jsonl yüklendi: {len(frames_data)} frame")
        else:
            print(f"⚠️ frames.jsonl bulunamadı: {args.frames}")
    
    # shot_events.json dosyası
    events_data = {'shots': [], 'passes': []}
    if args.shot_events:
        shot_path = args.shot_events
        if not os.path.exists(shot_path):
            shot_path = os.path.join(PROJECT_ROOT, args.shot_events)
        if os.path.exists(shot_path):
            import json
            with open(shot_path, 'r') as f:
                shots = json.load(f)
            events_data['shots'] = shots
            print(f"⚽ shot_events yüklendi: {len(shots)} şut")
        else:
            print(f"⚠️ shot_events bulunamadı: {args.shot_events}")
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}_field_lines.mp4")
    
    print(f"📹 Input: {video_path}")
    print(f"📂 Output: {output_path}")
    
    # Process
    processor = SoccerLineProcessor(device=args.device)
    processor.process_video(video_path, output_path, sample_interval=args.interval, frames_data=frames_data, events=events_data)
    
    # 2D Test
    if args.test_2d and hasattr(processor, 'last_calibrations'):
        print("\n" + "="*50)
        print("🔄 2D DÖNÜŞÜM TESTİ")
        print("="*50)
        
        # İlk kalibrasyonu al
        calib_frames = processor.last_calib_frames
        if calib_frames:
            P = processor.last_calibrations[calib_frames[0]]
            
            # Test pikselleri (ekran üzerindeki noktalar)
            test_pixels = [
                (960, 540),   # Ekran merkezi
                (100, 800),   # Sol alt
                (1820, 800),  # Sağ alt
                (960, 200),   # Üst orta
            ]
            
            print(f"\n📐 P Matrix (frame {calib_frames[0]}):")
            print(P)
            
            print("\n📍 Piksel -> Saha Koordinatı Dönüşümü:")
            for px, py in test_pixels:
                field_pos = processor.pixel_to_field(P, px, py)
                if field_pos:
                    fx, fy = field_pos
                    print(f"   Piksel ({px:4d}, {py:4d}) -> Saha ({fx:6.1f}m, {fy:5.1f}m)")
                else:
                    print(f"   Piksel ({px:4d}, {py:4d}) -> Dönüşüm başarısız")
            
            # Kale pozisyonlarını test et
            print("\n⚽ Kale Pozisyonları (saha koordinatı -> piksel):")
            left_goal = processor.project_point(P, [0, 34, 0])
            right_goal = processor.project_point(P, [105, 34, 0])
            print(f"   Sol kale (0, 34) -> Piksel {left_goal}")
            print(f"   Sağ kale (105, 34) -> Piksel {right_goal}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
