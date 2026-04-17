"""
calibrate_field.py
==================
Videonun belirli bir frame'ini açar, saha köşelerine tıklarsın,
config.py'deki MANUAL_CALIB_POINTS'i otomatik günceller.

Kullanım:
    python calibrate_field.py --video mac.mp4
    python calibrate_field.py --video mac.mp4 --frame 500

Tıklama sırası (önemli!):
    1. Sol-üst köşe  (kale arkası sol)
    2. Sağ-üst köşe (kale arkası sağ)
    3. Sağ-alt köşe
    4. Sol-alt köşe

Daha fazla nokta = daha doğru homografi.
4 nokta yeterli ama 8-12 nokta daha iyi sonuç verir.
"""

import cv2
import sys
import json
import argparse
from pathlib import Path

# Dünya koordinatları — tıklama sırasına göre eşleşecek saha noktaları
# Standart 4 köşe:
DEFAULT_WORLD_PTS = [
    [0.0,   68.0],   # sol-üst (soldaki kale çizgisi üst direği tarafı)
    [105.0, 68.0],   # sağ-üst
    [105.0, 0.0 ],   # sağ-alt
    [0.0,   0.0 ],   # sol-alt
]

# Daha fazla nokta istersen bunları kullan (calibrate_field.py --mode extended):
EXTENDED_WORLD_PTS = [
    [0.0,   68.0],   # sol-üst köşe
    [105.0, 68.0],   # sağ-üst köşe
    [105.0, 0.0 ],   # sağ-alt köşe
    [0.0,   0.0 ],   # sol-alt köşe
    [52.5,  68.0],   # orta çizgi üst
    [52.5,  0.0 ],   # orta çizgi alt
    [16.5,  68.0],   # sol ceza sahası üst
    [88.5,  68.0],   # sağ ceza sahası üst
]

clicked_points = []
frame_display  = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        n = len(clicked_points)
        # Noktayı çiz
        cv2.circle(frame_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_display, str(n), (x+10, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon", frame_display)
        print(f"  Nokta {n}: piksel=({x}, {y})")


def get_frame(video_path: str, frame_no: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Frame {frame_no} okunamadı")
    return frame


def update_config(pixel_pts, world_pts):
    """config.py'deki MANUAL_CALIB_POINTS'i günceller."""
    config_path = Path("config.py")
    if not config_path.exists():
        print("  config.py bulunamadı — calib.json olarak kaydediliyor")
        with open("calib.json", "w") as f:
            json.dump({"pixel": pixel_pts, "world": world_pts}, f, indent=2)
        return

    content = config_path.read_text(encoding="utf-8")

    new_block = (
        f"MANUAL_CALIB_POINTS = {{\n"
        f"    \"pixel\": {pixel_pts},\n"
        f"    \"world\": {world_pts},\n"
        f"}}"
    )

    # Eski MANUAL_CALIB_POINTS bloğunu değiştir
    import re
    # Çok satırlı bloğu bul ve değiştir
    pattern = r'MANUAL_CALIB_POINTS\s*=\s*None'
    if re.search(pattern, content):
        content = re.sub(pattern, new_block, content)
    else:
        # Zaten var, güncelle
        pattern2 = r'MANUAL_CALIB_POINTS\s*=\s*\{[^}]*\}'
        if re.search(pattern2, content, re.DOTALL):
            content = re.sub(pattern2, new_block, content, flags=re.DOTALL)
        else:
            content += f"\n\n{new_block}\n"

    config_path.write_text(content, encoding="utf-8")
    print(f"\n  ✅ config.py güncellendi!")


def main():
    global clicked_points, frame_display

    p = argparse.ArgumentParser()
    p.add_argument("--video",  default=None,
                   help="Video yolu (config.py VIDEO_PATH varsayilan)")
    p.add_argument("--frame",  type=int, default=200,
                   help="Kaç numaralı frame kullanılsın (default: 200)")
    p.add_argument("--mode",   choices=["simple", "extended"], default="simple",
                   help="simple=4 nokta, extended=8 nokta")
    args = p.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config as cfg
    if args.video is None:
        args.video = cfg.VIDEO_PATH

    world_pts = DEFAULT_WORLD_PTS if args.mode == "simple" else EXTENDED_WORLD_PTS
    n_points  = len(world_pts)

    print(f"\n{'='*55}")
    print(f"  SAHA KALİBRASYON ARACI")
    print(f"{'='*55}")
    print(f"  Video : {args.video}")
    print(f"  Frame : {args.frame}")
    print(f"  Mod   : {args.mode} ({n_points} nokta)")
    print(f"\n  Tıklama sırası:")
    labels = [
        "Sol-ust kose  (sol kale arkasi, ust kenar cizgisi)",
        "Sag-ust kose  (sag kale arkasi, ust kenar cizgisi)",
        "Sag-alt: alt kenar cizgisinin en sag gorunen noktasi",
        "Sol-alt: alt kenar cizgisinin en sol gorunen noktasi",
        "Orta cizgi ust",
        "Orta cizgi alt",
        "Sol ceza sahasi ust",
        "Sag ceza sahasi ust",
    ]
    for i, lbl in enumerate(labels[:n_points]):
        print(f"  {i+1}. {lbl}")

    print(f"\n  Tüm noktaları tıkladıktan sonra ENTER'a bas.")
    print(f"  Yanlış tıkladıysan 'r' tuşuna bas (sıfırla).")
    print(f"{'='*55}\n")

    frame = get_frame(args.video, args.frame)
    frame_display = frame.copy()

    # Pencere boyutunu ayarla
    h, w = frame.shape[:2]
    scale = min(1.0, 1400 / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    frame_display = cv2.resize(frame_display, (disp_w, disp_h))

    # Tıklama ölçeği düzeltmesi
    scale_x = w / disp_w
    scale_y = h / disp_h

    cv2.namedWindow("Kalibrasyon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kalibrasyon", disp_w, disp_h)
    cv2.setMouseCallback("Kalibrasyon", mouse_callback)

    # Talimatları frame üzerine yaz
    cv2.putText(frame_display,
                f"Siradaki nokta: 1 - {labels[0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Kalibrasyon", frame_display)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            # Sıfırla
            clicked_points = []
            frame_display = cv2.resize(frame.copy(), (disp_w, disp_h))
            cv2.imshow("Kalibrasyon", frame_display)
            print("  Sıfırlandı, tekrar tıklayabilirsin.")

        elif key == 13 or key == ord('q'):  # ENTER veya q
            break

        # Talimat güncelle
        if len(clicked_points) < n_points:
            overlay = frame_display.copy()
            next_lbl = labels[len(clicked_points)] if len(clicked_points) < len(labels) else ""
            cv2.putText(overlay,
                        f"Siradaki: {len(clicked_points)+1} - {next_lbl}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Kalibrasyon", overlay)

        if len(clicked_points) >= n_points:
            cv2.putText(frame_display,
                        "Tamamlandi! ENTER'a bas.",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Kalibrasyon", frame_display)

    cv2.destroyAllWindows()

    if len(clicked_points) < 4:
        print(f"\n  ❌ Yeterli nokta tıklanmadı ({len(clicked_points)}/4)")
        sys.exit(1)

    # Ölçeği orijinal frame boyutuna geri al
    actual_pts = [
        [int(x * scale_x), int(y * scale_y)]
        for x, y in clicked_points[:n_points]
    ]

    print(f"\n  Tıklanan noktalar (orijinal çözünürlük):")
    for i, (px, w_pt) in enumerate(zip(actual_pts, world_pts[:n_points])):
        print(f"    {i+1}. piksel={px}  →  saha={w_pt}m")

    update_config(actual_pts, world_pts[:n_points])

    print(f"\n  Artık pipeline.py'yi çalıştırabilirsin:")
    print(f"  python pipeline.py --video \"{args.video}\"")


if __name__ == "__main__":
    main()
