import json
from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import (
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
    complete_keypoints,
    coords_to_dict,
)

LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
    [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]],
    [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68., 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]],
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WEIGHTS_KP = "/work/PnLCalib/weights/SV_kp"
WEIGHTS_LINE = "/work/PnLCalib/weights/SV_lines"
CFG_KP = "/work/PnLCalib/config/hrnetv2_w48.yaml"
CFG_LINE = "/work/PnLCalib/config/hrnetv2_w48_l.yaml"

KEYFRAME_DIR = Path("/output/stage6_field/keyframes")
OUT_JSON_DIR = Path("/output/stage6_field/keyframe_json")
OUT_IMG_DIR = Path("/output/stage6_field/keyframe_projected")
BANK_JSON = Path("/output/stage6_field/homography_bank_raw.json")

KP_THRESHOLD = 0.3434
LINE_THRESHOLD = 0.7867
PNL_REFINE = False

MIN_KP_COUNT = 4
MIN_LINE_COUNT = 2
MAX_REP_ERR = 15.0

transform2 = T.Resize((540, 960))

def normalize_for_json(obj):
    if isinstance(obj, dict):
        return {k: normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'], dtype=np.float64)
    position_meters = np.array(cam_params['position_meters'], dtype=np.float64)
    rotation = np.array(cam_params['rotation_matrix'], dtype=np.float64)

    It = np.eye(4, dtype=np.float64)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]], dtype=np.float64)
    P = Q @ (rotation @ It)
    return P

def project_field_overlay(frame, P):
    frame = frame.copy()
    for line in LINES_COORDS:
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1], dtype=np.float64)
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1], dtype=np.float64)

        if abs(i1[-1]) < 1e-9 or abs(i2[-1]) < 1e-9:
            continue

        i1 /= i1[-1]
        i2 /= i2[-1]

        if not np.isfinite(i1).all() or not np.isfinite(i2).all():
            continue

        # numpy scalar → Python float → Python int (cv2.line requires native Python int)
        x1, y1 = float(i1[0]), float(i1[1])
        x2, y2 = float(i2[0]), float(i2[1])
        # Skip lines that project way off-screen (degenerate calibration)
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1e6:
            continue

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        try:
            frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)
        except cv2.error:
            pass
    return frame

def count_detected_points(d):
    cnt = 0
    for _, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            cnt += 1
    return cnt

def count_detected_lines(d):
    cnt = 0
    for _, v in d.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            cnt += 1
    return cnt

def sanity_score(final_params_dict):
    if final_params_dict is None:
        return 0.0
    try:
        P = projection_from_cam_params(final_params_dict)
        if not np.isfinite(P).all():
            return 0.0
        return 1.0
    except Exception:
        return 0.0

def load_models():
    cfg = yaml.safe_load(open(CFG_KP, 'r'))
    cfg_l = yaml.safe_load(open(CFG_LINE, 'r'))

    loaded_state = torch.load(WEIGHTS_KP, map_location=DEVICE)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(DEVICE)
    model.eval()

    loaded_state_l = torch.load(WEIGHTS_LINE, map_location=DEVICE)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(DEVICE)
    model_l.eval()

    return model, model_l

def run_one(image_path, model, model_l):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Image okunamadi: {image_path}")

    h_original, w_original = frame.shape[:2]
    cam = FramebyFrameCalib(iwidth=w_original, iheight=h_original, denormalize=True)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensor = F.to_tensor(pil).float().unsqueeze(0)
    if tensor.size()[-1] != 960:
        tensor = transform2(tensor)
    tensor = tensor.to(DEVICE)
    _, _, h, w = tensor.size()

    with torch.no_grad():
        heatmaps = model(tensor)
        heatmaps_l = model_l(tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])

    kp_dict = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
    lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=PNL_REFINE)

    kp_count = count_detected_points(kp_dict)
    line_count = count_detected_lines(lines_dict)

    rep_err = None
    if final_params_dict is not None and "rep_err" in final_params_dict:
        try:
            rep_err = float(final_params_dict["rep_err"])
        except Exception:
            rep_err = None

    sane = sanity_score(final_params_dict)

    accepted = (
        final_params_dict is not None
        and kp_count >= MIN_KP_COUNT
        and line_count >= MIN_LINE_COUNT
        and rep_err is not None
        and rep_err <= MAX_REP_ERR
        and sane > 0.5
    )

    output = {
        "image_path": str(image_path),
        "device": DEVICE,
        "kp_threshold": KP_THRESHOLD,
        "line_threshold": LINE_THRESHOLD,
        "pnl_refine": PNL_REFINE,
        "image_width": w_original,
        "image_height": h_original,
        "kp_count": kp_count,
        "line_count": line_count,
        "rep_err": rep_err,
        "sanity_score": sane,
        "accepted": accepted,
        "kp_dict": normalize_for_json(kp_dict),
        "lines_dict": normalize_for_json(lines_dict),
        "final_params_dict": normalize_for_json(final_params_dict) if final_params_dict is not None else None,
    }

    projected_img = None
    if final_params_dict is not None:
        P = projection_from_cam_params(final_params_dict)
        projected_img = project_field_overlay(frame, P)

    return output, projected_img

def main():
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    BANK_JSON.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(KEYFRAME_DIR.glob("*.png"))
    if len(image_paths) == 0:
        raise RuntimeError("Keyframe bulunamadi")

    model, model_l = load_models()

    bank = {
        "device": DEVICE,
        "kp_threshold": KP_THRESHOLD,
        "line_threshold": LINE_THRESHOLD,
        "pnl_refine": PNL_REFINE,
        "min_kp_count": MIN_KP_COUNT,
        "min_line_count": MIN_LINE_COUNT,
        "max_rep_err": MAX_REP_ERR,
        "frames": []
    }

    already_done = {p.stem for p in OUT_JSON_DIR.glob("*.json")}
    remaining = [p for p in image_paths if p.stem not in already_done]

    # Reload already-processed frames into bank so the final bank.json is complete
    for p in image_paths:
        if p.stem in already_done:
            with open(OUT_JSON_DIR / f"{p.stem}.json", encoding="utf-8") as f:
                out = json.load(f)
            bank["frames"].append({
                "image_path": out["image_path"],
                "accepted": out["accepted"],
                "kp_count": out["kp_count"],
                "line_count": out["line_count"],
                "rep_err": out["rep_err"],
                "sanity_score": out["sanity_score"],
                "final_params_dict": out["final_params_dict"],
            })

    skipped = len(already_done)
    print(f"\nStage 6 | Keyframe Calibration — {len(image_paths)} keyframe "
          f"({skipped} zaten tamamlandı, {len(remaining)} işlenecek)")

    for image_path in tqdm(remaining, desc="Stage 6 | Calibration  ",
                           unit="frame", ncols=90):
        output, projected_img = run_one(image_path, model, model_l)

        stem = image_path.stem
        out_json = OUT_JSON_DIR / f"{stem}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        if projected_img is not None:
            out_img = OUT_IMG_DIR / f"{stem}_projected.png"
            cv2.imwrite(str(out_img), projected_img)

        bank["frames"].append({
            "image_path": output["image_path"],
            "accepted": output["accepted"],
            "kp_count": output["kp_count"],
            "line_count": output["line_count"],
            "rep_err": output["rep_err"],
            "sanity_score": output["sanity_score"],
            "final_params_dict": output["final_params_dict"],
        })

    bank["frames"].sort(key=lambda x: x["image_path"])
    with open(BANK_JSON, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    accepted_count = sum(1 for fr in bank["frames"] if fr["accepted"])
    print("DONE")
    print("total_keyframes =", len(bank["frames"]))
    print("accepted_keyframes =", accepted_count)
    print("bank_json =", BANK_JSON)

if __name__ == "__main__":
    main()
