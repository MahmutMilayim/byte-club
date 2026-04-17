"""
Goal Zone Detection using soccer_line calibration.

Şut anında kale pozisyonunu tespit eder.
Her şut için o frame'deki kale koordinatlarını döner.
"""
import sys
import os
import cv2
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# soccer_line path
SOCCER_LINE_PATH = Path(__file__).parent.parent.parent / "soccer_line"

# Kale çizgisi koordinatları (dünya koordinatları - metre)
# Sol kale: x=0, sağ kale: x=105
# Kale genişliği: 7.32m (y: 30.34 - 37.66)
# Kale yüksekliği: 2.44m (z: 0 to -2.44)
GOAL_LINES = {
    'left': [
        [[0., 37.66, -2.44], [0., 30.34, -2.44]],  # Üst çizgi
        [[0., 37.66, 0.], [0., 37.66, -2.44]],     # Sol direk
        [[0., 30.34, 0.], [0., 30.34, -2.44]],     # Sağ direk
    ],
    'right': [
        [[105., 37.66, -2.44], [105., 30.34, -2.44]],  # Üst çizgi
        [[105., 30.34, 0.], [105., 30.34, -2.44]],     # Sol direk
        [[105., 37.66, 0.], [105., 37.66, -2.44]],     # Sağ direk
    ]
}

# Kale köşeleri (polygon için)
GOAL_CORNERS = {
    'left': [
        [0., 30.34, 0.],      # Alt sol
        [0., 37.66, 0.],      # Alt sağ
        [0., 37.66, -2.44],   # Üst sağ
        [0., 30.34, -2.44],   # Üst sol
    ],
    'right': [
        [105., 30.34, 0.],
        [105., 37.66, 0.],
        [105., 37.66, -2.44],
        [105., 30.34, -2.44],
    ]
}


class GoalZoneDetector:
    """Detects goal zone pixel coordinates from a video frame using soccer_line."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_kp = None
        self.model_line = None
        self.transform = T.Resize((540, 960))
        self.loaded = False
        
        self._load_models()
    
    def _load_models(self):
        """Load soccer_line models."""
        try:
            # Add soccer_line to path
            sys.path.insert(0, str(SOCCER_LINE_PATH))
            
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            
            weights_kp = SOCCER_LINE_PATH / "weights" / "MV_kp"
            weights_line = SOCCER_LINE_PATH / "weights" / "MV_lines"
            config_kp = SOCCER_LINE_PATH / "config" / "hrnetv2_w48.yaml"
            config_line = SOCCER_LINE_PATH / "config" / "hrnetv2_w48_l.yaml"
            
            if not all(p.exists() for p in [weights_kp, weights_line, config_kp, config_line]):
                # Silent fail - don't print warning
                return
            
            # Load configs
            cfg = yaml.safe_load(open(config_kp, 'r'))
            cfg_l = yaml.safe_load(open(config_line, 'r'))
            
            # Keypoint model
            self.model_kp = get_cls_net(cfg)
            state = torch.load(weights_kp, map_location=self.device)
            self.model_kp.load_state_dict(state)
            self.model_kp.to(self.device)
            self.model_kp.eval()
            
            # Line model
            self.model_line = get_cls_net_l(cfg_l)
            state_l = torch.load(weights_line, map_location=self.device)
            self.model_line.load_state_dict(state_l)
            self.model_line.to(self.device)
            self.model_line.eval()
            
            self.loaded = True
            print("✅ GoalZoneDetector: soccer_line models loaded")
            
        except Exception as e:
            print(f"❌ GoalZoneDetector load failed: {e}")
            import traceback
            traceback.print_exc()
    
    def detect_goal_zone(self, frame: np.ndarray, target_side: str = "LEFT") -> Optional[Dict]:
        """
        Detect goal zone from a single frame.
        
        Args:
            frame: BGR image (numpy array)
            target_side: "LEFT" or "RIGHT"
            
        Returns:
            Dict with goal zone info or None
        """
        if not self.loaded:
            return None
        
        try:
            sys.path.insert(0, str(SOCCER_LINE_PATH))
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints,
                coords_to_dict
            )
            
            h_orig, w_orig = frame.shape[:2]
            
            # Preprocess frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = TF.to_tensor(pil_img).float().unsqueeze(0)
            
            # Resize if needed
            if tensor.size()[-1] != 960:
                tensor = self.transform(tensor)
            
            tensor = tensor.to(self.device)
            b, c, h, w = tensor.size()
            
            # Initialize calibrator
            cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
            
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
            
            # Get camera calibration
            cam.update(kp_dict, lines_dict)
            params = cam.heuristic_voting(refine_lines=True)
            
            if params is None:
                return None
            
            # Get projection matrix
            P = self._get_projection_matrix(params)
            
            # Project goal corners to image
            side = 'left' if target_side == "LEFT" else 'right'
            corners = GOAL_CORNERS[side]
            
            pixel_corners = []
            for world_pt in corners:
                px = self._project_point(P, world_pt)
                if px is not None:
                    pixel_corners.append(px)
            
            if len(pixel_corners) < 4:
                return None
            
            # Calculate bounds
            xs = [p[0] for p in pixel_corners]
            ys = [p[1] for p in pixel_corners]
            
            return {
                'polygon': pixel_corners,
                'x_min': min(xs),
                'x_max': max(xs),
                'y_min': min(ys),
                'y_max': max(ys),
                'center': (sum(xs)/4, sum(ys)/4),
                'valid': True,
                'projection_matrix': P
            }
            
        except Exception as e:
            print(f"❌ detect_goal_zone error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_projection_matrix(self, params: Dict) -> np.ndarray:
        """Get projection matrix from camera parameters."""
        cam = params["cam_params"]
        x_focal = cam['x_focal_length']
        y_focal = cam['y_focal_length']
        principal = np.array(cam['principal_point'])
        position = np.array(cam['position_meters'])
        rotation = np.array(cam['rotation_matrix'])
        
        It = np.eye(4)[:-1]
        It[:, -1] = -position
        Q = np.array([
            [x_focal, 0, principal[0]],
            [0, y_focal, principal[1]],
            [0, 0, 1]
        ])
        return Q @ (rotation @ It)
    
    def _project_point(self, P: np.ndarray, world_pt: List[float]) -> Optional[Tuple[int, int]]:
        """Project 3D world point to 2D image pixel."""
        x, y, z = world_pt
        # World coords centered at field center
        pt = np.array([x - 105/2, y - 68/2, z, 1])
        
        img_pt = P @ pt
        if img_pt[2] <= 0:
            return None
        
        img_pt /= img_pt[2]
        return (int(img_pt[0]), int(img_pt[1]))
    
    def draw_goal_zone(self, frame: np.ndarray, goal_zone: Dict, color=(0, 255, 0), thickness=3) -> np.ndarray:
        """Draw goal zone polygon on frame."""
        if not goal_zone or not goal_zone.get('valid'):
            return frame
        
        polygon = goal_zone['polygon']
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw outline
        cv2.polylines(frame, [pts], True, color, thickness)
        
        # Draw corner points
        for pt in polygon:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        
        return frame


# Singleton instance
_detector = None

def get_detector(device: str = "cpu") -> GoalZoneDetector:
    """Get or create singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = GoalZoneDetector(device)
    return _detector


def detect_goal_zone_from_video(video_path: str, frame_idx: int, target_side: str = "LEFT") -> Optional[Dict]:
    """
    Convenience function to detect goal zone from a specific frame.
    
    Args:
        video_path: Path to video
        frame_idx: Frame number
        target_side: "LEFT" or "RIGHT"
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    detector = get_detector()
    return detector.detect_goal_zone(frame, target_side)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python goal_zone.py <video_path> <frame_idx> [LEFT|RIGHT]")
        sys.exit(1)
    
    video = sys.argv[1]
    frame_idx = int(sys.argv[2])
    side = sys.argv[3] if len(sys.argv) > 3 else "LEFT"
    
    # Test detection
    result = detect_goal_zone_from_video(video, frame_idx, side)
    print(f"Goal zone at frame {frame_idx}, side {side}:")
    print(result)
    
    # Draw and save test image
    if result and result.get('valid'):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            detector = get_detector()
            frame = detector.draw_goal_zone(frame, result)
            cv2.imwrite(f"goal_zone_test_frame_{frame_idx}.jpg", frame)
            print(f"Saved: goal_zone_test_frame_{frame_idx}.jpg")

