"""
Soccer Line Integration for Main Pipeline
Her saniye 1 frame kalibrasyon yaparak saha koordinatı dönüşümü sağlar
"""
import os
import sys
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Callable
from PIL import Image
import torchvision.transforms.functional as TF

# Soccer line path'ini ekle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOCCER_LINE_PATH = os.path.join(PROJECT_ROOT, "soccer_line")

# Orijinal working directory'yi sakla
_original_cwd = os.getcwd()


class SoccerLineCalibrator:
    """
    Soccer Line kamera kalibrasyonu
    Pipeline'da frame'leri 2D saha koordinatlarına dönüştürür
    """
    
    # Saha boyutları (metre)
    FIELD_LENGTH = 105.0
    FIELD_WIDTH = 68.0
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Soccer Line models
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_kp = None
        self.model_line = None
        self.transform = None
        self.initialized = False
        
        # Kalibrasyon cache (frame_idx -> P matrix)
        self.calibrations: Dict[int, np.ndarray] = {}
        self.calib_frames: list = []
        
        # Aktif kalibrasyon
        self.current_P: Optional[np.ndarray] = None
        self.frame_size: Optional[Tuple[int, int]] = None
        
        self._load_models()
    
    def _load_models(self):
        """Load HRNet models for keypoint and line detection"""
        original_cwd = os.getcwd()
        
        try:
            # Soccer line dizinine geç
            os.chdir(SOCCER_LINE_PATH)
            
            # Soccer line path'ini en başa ekle
            sys.path.insert(0, SOCCER_LINE_PATH)
            
            import yaml
            
            # Import'ları doğrudan yap - model klasörü (models DEĞİL!)
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                coords_to_dict,
                complete_keypoints
            )
            
            # Store imports for later use
            self._FramebyFrameCalib = FramebyFrameCalib
            self._get_kp_heatmap = get_keypoints_from_heatmap_batch_maxpool
            self._get_line_heatmap = get_keypoints_from_heatmap_batch_maxpool_l
            self._coords_to_dict = coords_to_dict
            self._complete_keypoints = complete_keypoints
            
            # YAML Config dosyalarını yükle (run_soccer_line.py gibi)
            cfg_kp = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
            cfg_line = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))
            
            # Load keypoint model
            weights_kp = "weights/SV_kp"
            self.model_kp = get_cls_net(cfg_kp)
            state_kp = torch.load(weights_kp, map_location=self.device)
            self.model_kp.load_state_dict(state_kp)
            self.model_kp.to(self.device)
            self.model_kp.eval()
            
            # Load line model
            weights_line = "weights/SV_lines"
            self.model_line = get_cls_net_l(cfg_line)
            state_line = torch.load(weights_line, map_location=self.device)
            self.model_line.load_state_dict(state_line)
            self.model_line.to(self.device)
            self.model_line.eval()
            
            # Transform for resizing
            self.transform = torch.nn.Upsample(size=(540, 960), mode='bilinear', align_corners=True)
            
            self.initialized = True
            print(f"[SoccerLine] ✅ Models loaded on {self.device}")
            
        except Exception as e:
            import traceback
            print(f"[SoccerLine] Failed to load models: {e}")
            traceback.print_exc()
            self.initialized = False
        finally:
            # Orijinal dizin ve path'e geri dön
            os.chdir(original_cwd)
            # Soccer line path'ini koru (kalibrasyon için gerekli)
    
    def calibrate_frame(self, frame: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        """
        Tek frame için kamera kalibrasyonu yap
        
        Args:
            frame: BGR frame
            frame_idx: Frame index
            
        Returns:
            P matrix (3x4) or None
        """
        if not self.initialized:
            return None
        
        try:
            os.chdir(SOCCER_LINE_PATH)
            
            h_orig, w_orig = frame.shape[:2]
            self.frame_size = (w_orig, h_orig)
            
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
            kp_coords = self._get_kp_heatmap(heatmaps_kp[:, :-1, :, :])
            line_coords = self._get_line_heatmap(heatmaps_line[:, :-1, :, :])
            
            kp_dict = self._coords_to_dict(kp_coords, threshold=0.3434)
            lines_dict = self._coords_to_dict(line_coords, threshold=0.7867)
            
            kp_dict, lines_dict = self._complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
            
            # Calibrate
            cam = self._FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
            cam.update(kp_dict, lines_dict)
            params = cam.heuristic_voting(refine_lines=True)
            
            if params is None:
                return None
            
            # Get projection matrix
            P = self._get_projection_matrix(params)
            
            # Cache
            self.calibrations[frame_idx] = P
            self.calib_frames = sorted(self.calibrations.keys())
            self.current_P = P
            
            return P
            
        except Exception as e:
            print(f"[SoccerLine] Calibration failed for frame {frame_idx}: {e}")
            return None
        finally:
            os.chdir(_original_cwd)
    
    def _get_projection_matrix(self, params: Dict) -> np.ndarray:
        """Extract projection matrix from calibration params"""
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
    
    def get_P_for_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Verilen frame için P matrix döndür (interpolasyon ile)
        """
        if len(self.calib_frames) == 0:
            return self.current_P
        
        # Tam eşleşme
        if frame_idx in self.calibrations:
            return self.calibrations[frame_idx]
        
        # En yakın önceki ve sonraki kalibrasyonu bul
        prev_frame = None
        next_frame = None
        
        for cf in self.calib_frames:
            if cf <= frame_idx:
                prev_frame = cf
            if cf > frame_idx and next_frame is None:
                next_frame = cf
                break
        
        # Sadece önceki varsa
        if prev_frame is not None and next_frame is None:
            return self.calibrations[prev_frame]
        
        # Sadece sonraki varsa
        if prev_frame is None and next_frame is not None:
            return self.calibrations[next_frame]
        
        # İkisi de varsa - interpolasyon
        if prev_frame is not None and next_frame is not None:
            t = (frame_idx - prev_frame) / (next_frame - prev_frame)
            P_prev = self.calibrations[prev_frame]
            P_next = self.calibrations[next_frame]
            return (1 - t) * P_prev + t * P_next
        
        return self.current_P
    
    def pixel_to_field(self, pixel_x: float, pixel_y: float, frame_idx: Optional[int] = None) -> Optional[Tuple[float, float]]:
        """
        Piksel koordinatını saha koordinatına dönüştür
        
        Args:
            pixel_x, pixel_y: Piksel koordinatları
            frame_idx: Frame index (None ise en son kalibrasyon kullanılır)
            
        Returns:
            (field_x, field_y) metre cinsinden veya None
        """
        P = self.get_P_for_frame(frame_idx) if frame_idx is not None else self.current_P
        
        if P is None:
            return None
        
        try:
            # Homography (z=0 için)
            H = np.column_stack([P[:, 0], P[:, 1], P[:, 3]])
            H_inv = np.linalg.inv(H)
            
            # Piksel -> dünya koordinatı
            pixel = np.array([pixel_x, pixel_y, 1.0])
            world_h = H_inv @ pixel
            
            if abs(world_h[2]) < 1e-6:
                return None
            
            world_x = world_h[0] / world_h[2]
            world_y = world_h[1] / world_h[2]
            
            # Saha merkezini offset olarak ekle
            field_x = world_x + self.FIELD_LENGTH / 2
            field_y = world_y + self.FIELD_WIDTH / 2
            
            # Saha sınırları kontrolü (tolerans ile)
            margin = 15
            if not (-margin <= field_x <= self.FIELD_LENGTH + margin and
                    -margin <= field_y <= self.FIELD_WIDTH + margin):
                return None
            
            return (field_x, field_y)
            
        except np.linalg.LinAlgError:
            return None
    
    def get_pixel_to_field_func(self) -> Callable:
        """
        ID stabilizer için pixel_to_field fonksiyonu döndür
        """
        def pixel_to_field_wrapper(pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
            return self.pixel_to_field(pixel_x, pixel_y)
        
        return pixel_to_field_wrapper


def create_calibrator(device: str = "cuda") -> Optional[SoccerLineCalibrator]:
    """
    SoccerLine calibrator oluştur
    Başarısız olursa None döner
    """
    try:
        calibrator = SoccerLineCalibrator(device=device)
        if calibrator.initialized:
            return calibrator
        return None
    except Exception as e:
        print(f"[SoccerLine] Could not create calibrator: {e}")
        return None
