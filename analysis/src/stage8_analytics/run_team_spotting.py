import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


DEFAULT_TEAMSPOT_ROOT = "/workspace/third_party/sn-teamspotting"
DEFAULT_VIDEO_PATH = "/output/stage3_filter/gameplay.mp4"
DEFAULT_OUT_JSON = "/output/stage8_analytics/team_spotting_raw.json"
DEFAULT_MODEL_NAME = "SoccerNetBall_baseline"
DEFAULT_SIZE = (796, 448)
FPS_TARGET = 25.0
STRIDE_SNB = 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teamspot-root", default=DEFAULT_TEAMSPOT_ROOT)
    ap.add_argument("--video-path", default=DEFAULT_VIDEO_PATH)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device_arg):
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_imports(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class VideoClipDataset(IterableDataset):
    def __init__(self, video_path, clip_len, overlap_len, stride, size):
        self.video_path = str(video_path)
        self.clip_len = int(clip_len)
        self.overlap_len = int(overlap_len)
        self.stride = int(stride)
        self.size = tuple(size)
        self.pop_len = self.clip_len - self.overlap_len
        stream = cv2.VideoCapture(self.video_path)
        self.video_len = int(stream.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_fps = float(stream.get(cv2.CAP_PROP_FPS) or 0.0)
        stream.release()

    def __iter__(self):
        stream = cv2.VideoCapture(self.video_path)
        buffer = deque()
        i = -5 * self.stride
        while True:
            if i < 0:
                if i % self.stride == 0:
                    frame = np.zeros((self.size[1], self.size[0], 3), np.uint8)
                    buffer.append(torch.from_numpy(frame).permute(2, 0, 1))
                i += 1
                continue

            ret, frame = stream.read()
            if not ret:
                break
            if i % self.stride != 0:
                i += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            buffer.append(torch.from_numpy(frame).permute(2, 0, 1))
            i += 1

            if len(buffer) == self.clip_len:
                yield torch.stack(list(buffer)), (i + self.stride - 1) // self.stride - self.clip_len
                for _ in range(self.pop_len):
                    if buffer:
                        buffer.popleft()
        stream.release()


def build_runtime_args(cli_args, config):
    args = argparse.Namespace()
    args.frame_dir = config["frame_dir"]
    args.save_dir = config["save_dir"]
    args.store_dir = os.path.join(config["save_dir"], "StoreClips", config["dataset"])
    args.store_mode = "load"
    args.batch_size = int(cli_args.batch_size)
    args.clip_len = int(config["clip_len"])
    args.crop_dim = int(config["crop_dim"])
    if args.crop_dim <= 0:
        args.crop_dim = None
    args.dataset = config["dataset"]
    args.event_team = bool(config.get("event_team", True))
    args.radi_displacement = int(config["radi_displacement"])
    args.epoch_num_frames = int(config["epoch_num_frames"])
    args.feature_arch = config["feature_arch"]
    args.learning_rate = float(config["learning_rate"])
    args.mixup = bool(config["mixup"])
    args.modality = config["modality"]
    args.num_classes = int(config["num_classes"])
    args.num_epochs = int(config["num_epochs"])
    args.warm_up_epochs = int(config["warm_up_epochs"])
    args.start_val_epoch = int(config["start_val_epoch"])
    args.temporal_arch = config["temporal_arch"]
    args.n_layers = int(config["n_layers"])
    args.sgp_ks = int(config["sgp_ks"])
    args.sgp_r = int(config["sgp_r"])
    args.only_test = True
    args.criterion = config["criterion"]
    args.num_workers = int(cli_args.num_workers)
    args.joint_train = config.get("joint_train")
    return args


def load_model(teamspot_root, model_name, runtime_args, device):
    from model.model import TDEEDModel
    from util.dataset import load_classes

    class_path = Path(teamspot_root) / "data" / runtime_args.dataset / "class.txt"
    classes = load_classes(str(class_path), event_team=runtime_args.event_team)
    ckpt_path = (
        Path(teamspot_root)
        / "checkpoints"
        / model_name.split("_", 1)[0]
        / model_name
        / "checkpoint_best.pt"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = TDEEDModel(device=device, args=runtime_args)
    if runtime_args.joint_train:
        joint_class_path = (
            Path(teamspot_root)
            / "data"
            / runtime_args.joint_train["dataset"]
            / "class.txt"
        )
        joint_train_classes = load_classes(
            str(joint_class_path),
            event_team=runtime_args.event_team,
        )
        if runtime_args.event_team:
            n_classes = [len(classes) // 2 + 1, len(joint_train_classes) // 2 + 1]
        else:
            n_classes = [len(classes) + 1, len(joint_train_classes) + 1]
        model._model.update_pred_head(n_classes)
        model._num_classes = int(np.array(n_classes).sum())
    state = torch.load(str(ckpt_path), map_location=device)
    model.load(state)
    return model, classes, ckpt_path


def process_frame_predictions_local(classes, scores, threshold):
    classes_inv = {v: k for k, v in classes.items()}
    events = []
    for frame_idx in range(scores.shape[0]):
        for class_idx, label in classes_inv.items():
            score = float(scores[frame_idx, class_idx])
            if score < threshold:
                continue
            if "-" in label:
                base_label, side = label.rsplit("-", 1)
                events.append(
                    {
                        "label": base_label,
                        "side": side,
                        "frame": int(frame_idx),
                        "score": score,
                    }
                )
            else:
                events.append(
                    {
                        "label": label,
                        "frame": int(frame_idx),
                        "score": score,
                    }
                )
    return [{"video": "gameplay", "events": events, "fps": FPS_TARGET / STRIDE_SNB}]


def serialise_events(pred_events, stride):
    items = []
    for event in pred_events[0].get("events", []):
        items.append(
            {
                "frame": int(event["frame"]) * int(stride),
                "frame_stride_space": int(event["frame"]),
                "label": str(event["label"]).upper(),
                "side": str(event.get("side") or "").lower() or None,
                "confidence": round(float(event["score"]), 6),
            }
        )
    items.sort(key=lambda item: (item["frame"], -item["confidence"]))
    return items


def main():
    args = parse_args()
    ensure_imports(args.teamspot_root)
    device = resolve_device(args.device)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_cap = cv2.VideoCapture(args.video_path)
    if not video_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")
    video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(video_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_cap.release()
    if video_len <= 0:
        raise RuntimeError(f"Video has no frames: {args.video_path}")

    config = load_json(
        Path(args.teamspot_root)
        / "config"
        / "SoccerNetBall"
        / f"{args.model_name}.json"
    )
    runtime_args = build_runtime_args(args, config)
    model, classes, ckpt_path = load_model(args.teamspot_root, args.model_name, runtime_args, device)

    overlap_len = runtime_args.clip_len // 4 * 3
    dataset = VideoClipDataset(
        args.video_path,
        clip_len=runtime_args.clip_len,
        overlap_len=overlap_len,
        stride=STRIDE_SNB,
        size=DEFAULT_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    score_len = max(1, int(np.ceil(video_len / STRIDE_SNB)))
    predictions = np.zeros((score_len, len(classes) + 1), np.float32)
    support = np.zeros((score_len,), np.int32)

    print(
        f"\nStage 8 v4 | Team Spotting - {args.model_name}"
        f" | video_len={video_len} fps={video_fps:.4f} stride={STRIDE_SNB}"
    )
    for frames, starts in tqdm(loader, desc="Stage 8 v4 | Team Spotting", unit="clip", ncols=90):
        _, batch_pred_scores = model.predict(frames, use_amp=(device == "cuda"))
        for i in range(frames.shape[0]):
            pred_scores = batch_pred_scores[i]
            start = int(starts[i].item())
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= predictions.shape[0]:
                end = predictions.shape[0]
                pred_scores = pred_scores[: end - start, :]
            if end <= start:
                continue
            predictions[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0).astype(np.int32)

    safe_support = np.maximum(support, 1)
    norm_scores = predictions / safe_support[:, None]

    from util.eval import WINDOWS_SNB, soft_non_maximum_supression
    from util.io import store_json_snb

    pred_events = process_frame_predictions_local(classes, norm_scores, args.threshold)
    pred_events = soft_non_maximum_supression(
        pred_events,
        window=WINDOWS_SNB[1],
        threshold=args.threshold,
    )
    serialised = serialise_events(pred_events, STRIDE_SNB)

    pred_dir = out_path.parent / "sn_teamspotting_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    store_json_snb(
        str(pred_dir),
        [
            {
                "video": "gameplay",
                "events": [
                    {
                        "label": event["label"],
                        "team": event["side"],
                        "frame": int(event["frame"] // STRIDE_SNB),
                        "score": float(event["confidence"]),
                    }
                    for event in serialised
                ],
            }
        ],
        stride=STRIDE_SNB,
    )

    payload = {
        "model_name": args.model_name,
        "video_path": args.video_path,
        "video_frames": int(video_len),
        "video_fps": round(video_fps, 6),
        "expected_fps": FPS_TARGET,
        "stride": STRIDE_SNB,
        "clip_len": int(runtime_args.clip_len),
        "overlap_len": int(overlap_len),
        "checkpoint_path": str(ckpt_path),
        "postprocess": "soft_non_maximum_supression",
        "threshold": float(args.threshold),
        "prediction_dir": str(pred_dir),
        "events": serialised,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("out =", out_path)
    print("events =", len(serialised))


if __name__ == "__main__":
    main()
