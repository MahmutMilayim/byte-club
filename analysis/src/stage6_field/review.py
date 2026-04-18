import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


VIDEO_PATH = "/output/stage3_filter/gameplay.mp4"
TRACK_LABELS_JSON_PATH = "/output/stage4_clustering/track_labels_corrected.json"
PROJECTION_JSON_PATH = "/output/stage6_field/projection.json"
OUT_DIR = "/output/stage6_field"
OUT_VIDEO_NAME = "review_overlay.mp4"

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0
MINIMAP_W = 420
MINIMAP_H = 272
BOTTOM_MARGIN = 24
PANEL_PAD = 14
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-path", default=VIDEO_PATH)
    ap.add_argument("--track-labels-json-path", default=TRACK_LABELS_JSON_PATH)
    ap.add_argument("--projection-json-path", default=PROJECTION_JSON_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--out-name", default=OUT_VIDEO_NAME)
    return ap.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pitch_to_canvas(x, y, x0, y0, w, h):
    return x0 + int((x / FIELD_LENGTH_M) * w), y0 + int((y / FIELD_WIDTH_M) * h)


def draw_pitch(canvas, x0, y0, w, h):
    lc = (255, 255, 255)
    x1, y1 = x0 + w, y0 + h
    cv2.rectangle(canvas, (x0, y0), (x1, y1), lc, 2)
    mx = x0 + w // 2
    cv2.line(canvas, (mx, y0), (mx, y1), lc, 2)
    cv2.circle(canvas, pitch_to_canvas(52.5, 34.0, x0, y0, w, h), int((9.15 / FIELD_WIDTH_M) * h), lc, 2)
    for xa, ya, xb, yb in [
        (0.0, 13.84, 16.5, 54.16),
        (88.5, 13.84, 105.0, 54.16),
        (0.0, 24.84, 5.5, 43.16),
        (99.5, 24.84, 105.0, 43.16),
    ]:
        cv2.rectangle(canvas, pitch_to_canvas(xa, ya, x0, y0, w, h), pitch_to_canvas(xb, yb, x0, y0, w, h), lc, 2)


def alpha_rect(frame, x1, y1, x2, y2, color=(20, 20, 20), alpha=0.28):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def label_color(label):
    return {
        "team_1": (255, 0, 0),
        "team_2": (0, 140, 255),
        "referee": (0, 255, 255),
    }.get(label, (180, 180, 180))


def main():
    args = parse_args()

    track_labels = load_json(args.track_labels_json_path)
    projection = load_json(args.projection_json_path)

    id_to_label = {int(k): v["label"] for k, v in track_labels["tracks"].items()}
    frames = projection["frames"]
    if not frames:
        raise RuntimeError("Projection has no frames to render")

    start_frame = int(frames[0]["frame_index"])
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {args.video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    writer = cv2.VideoWriter(out_path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (video_w, video_h))
    if not writer.isOpened():
        raise RuntimeError(f"Video could not be written: {out_path.as_posix()}")

    panel_w = MINIMAP_W + 2 * PANEL_PAD
    panel_h = MINIMAP_H + 2 * PANEL_PAD
    panel_x1 = (video_w - panel_w) // 2
    panel_y1 = video_h - panel_h - BOTTOM_MARGIN
    panel_x2, panel_y2 = panel_x1 + panel_w, panel_y1 + panel_h
    map_x0, map_y0 = panel_x1 + PANEL_PAD, panel_y1 + PANEL_PAD

    print(f"\nStage 6 | Render Review - {len(frames)} frames")
    expected_frame_index = start_frame
    for fr in tqdm(frames, desc="Stage 6 | Review", unit="frame", ncols=90):
        frame_index = int(fr["frame_index"])
        while expected_frame_index < frame_index:
            ok, _ = cap.read()
            if not ok:
                break
            expected_frame_index += 1

        ok, frame = cap.read()
        if not ok:
            break
        expected_frame_index = frame_index + 1

        for player in fr["players"]:
            x1, y1, x2, y2 = map(int, player["bbox_xyxy"])
            col = label_color(id_to_label.get(int(player["track_id"]), player["label"]))
            thickness = 3 if player.get("projection_trusted") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)
            cv2.putText(frame, f"{player['label']} {player['track_id']}", (x1, max(18, y1 - 6)), FONT, 0.48, col, 2, cv2.LINE_AA)

        alpha_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (map_x0, map_y0), (map_x0 + MINIMAP_W, map_y0 + MINIMAP_H), (40, 120, 40), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        draw_pitch(frame, map_x0, map_y0, MINIMAP_W, MINIMAP_H)

        for player in fr["players"]:
            xy = player.get("pitch_xy")
            if xy is None or not (0.0 <= xy[0] <= FIELD_LENGTH_M and 0.0 <= xy[1] <= FIELD_WIDTH_M):
                continue
            cx, cy = pitch_to_canvas(xy[0], xy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
            col = (255, 0, 0) if player["label"] == "team_1" else (0, 140, 255)
            radius = 6 if player.get("projection_trusted") else 4
            cv2.circle(frame, (cx, cy), radius, col, -1)

        for referee in fr["referees"]:
            xy = referee.get("pitch_xy")
            if xy is None or not (0.0 <= xy[0] <= FIELD_LENGTH_M and 0.0 <= xy[1] <= FIELD_WIDTH_M):
                continue
            cx, cy = pitch_to_canvas(xy[0], xy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        ball = fr["ball"]
        ball_xy = ball.get("pitch_xy_trusted") or ball.get("pitch_xy_smoothed")
        if ball_xy is not None and 0.0 <= ball_xy[0] <= FIELD_LENGTH_M and 0.0 <= ball_xy[1] <= FIELD_WIDTH_M:
            cx, cy = pitch_to_canvas(ball_xy[0], ball_xy[1], map_x0, map_y0, MINIMAP_W, MINIMAP_H)
            col = (255, 255, 255) if ball.get("ball_ground_trusted") else (170, 170, 170)
            cv2.circle(frame, (cx, cy), 4, col, -1)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)

        if ball.get("image_xy") is not None:
            bx, by = map(int, ball["image_xy"])
            ball_col = (255, 255, 255) if ball.get("ball_ground_trusted") else (0, 255, 255)
            cv2.circle(frame, (bx, by), 6, ball_col, -1)
            cv2.circle(frame, (bx, by), 10, (0, 0, 0), 2)

        cv2.putText(frame, "2D Pitch Map", (panel_x1 + 10, panel_y1 - 8), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_index} | H {fr['homography_quality_score']:.3f} | {fr['homography_selection_source']}", (24, 36), FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"ball: {ball.get('ball_state')} | trusted={int(bool(ball.get('ball_ground_trusted')))}", (24, 68), FONT, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()

    print("DONE")
    print("out =", out_path.as_posix())
    print("frames =", len(frames))


if __name__ == "__main__":
    main()
