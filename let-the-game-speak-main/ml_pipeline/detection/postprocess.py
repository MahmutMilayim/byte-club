# ml_pipeline/detection/postprocess.py
from typing import Tuple
import cv2

DRAW_LABELS = ["Player-L", "Player-R", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
BOX_COLORS = {
    0:(150,50,50), 1:(37,47,150), 2:(41,248,165), 3:(166,196,10),
    4:(155,62,157), 5:(123,174,213), 6:(217,89,204), 7:(22,11,15)
}

def draw_box(frame,
             x1:int, y1:int, x2:int, y2:int,
             label_idx,
             thickness:int=2,
             font_scale:float=0.7):
    """
    Tek bir yardimci ile dikdortgen ve etiket cizer.
    
    label_idx None ise cizim yapilmaz (renk tespiti basarisiz).
    """
    # Renk tespiti basarisiz - bu framede cizim yapma
    if label_idx is None:
        return
    
    label_idx = max(0, min(int(label_idx), len(DRAW_LABELS)-1))
    color = BOX_COLORS.get(label_idx, (0,255,0))
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    cv2.putText(frame, DRAW_LABELS[label_idx], (x1, max(0, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
