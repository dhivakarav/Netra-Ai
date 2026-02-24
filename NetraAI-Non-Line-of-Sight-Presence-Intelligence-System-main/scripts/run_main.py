import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def clamp_roi(roi, w, h):
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 2, x1))
    y1 = max(0, min(h - 2, y1))
    x2 = max(1, min(w - 1, x2))
    y2 = max(1, min(h - 1, y2))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return (x1, y1, x2, y2)


def _to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def blur_score_laplacian(bgr) -> float:
    g = _to_gray(bgr)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def brightness_mean(bgr) -> float:
    g = _to_gray(bgr)
    return float(np.mean(g))

def contrast_std(bgr) -> float:
    g = _to_gray(bgr)
    return float(np.std(g))

def edge_density(bgr, t1=60, t2=160) -> float:
    g = _to_gray(bgr)
    edges = cv2.Canny(g, t1, t2)
    return float(np.mean(edges > 0))  

def dark_channel_mean(bgr, ksize=15) -> float:
    b, g, r = cv2.split(bgr)
    m = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dark = cv2.erode(m, kernel)
    return float(np.mean(dark) / 255.0) 

def glare_score(bgr) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float(np.mean(v >= 245))  

def classify_view(bgr) -> dict:
    """
    Heuristic visibility classifier: clear / fog / rain / glare / blur / dark / blocked.
    """
    blur = blur_score_laplacian(bgr)
    br = brightness_mean(bgr)
    ctr = contrast_std(bgr)
    ed = edge_density(bgr)
    dc = dark_channel_mean(bgr)
    gl = glare_score(bgr)

    blocked = (ctr < 8 and ed < 0.01)
    dark = (br < 18)
    blurry = (blur < 45 and not blocked)
    foggy = (ctr < 22 and ed < 0.03 and dc > 0.22 and not dark and not blocked)
    glare = (gl > 0.03 and not blocked)
    rainy = (ed > 0.08 and ctr < 40 and not foggy and not blocked and not dark)

    labels = []
    if blocked: labels.append("blocked")
    if dark: labels.append("dark")
    if glare: labels.append("glare")
    if foggy: labels.append("fog")
    if rainy: labels.append("rain")
    if blurry: labels.append("blur")

    primary = labels[0] if labels else "clear"
    return {
        "primary": primary,
        "labels": labels,
        "metrics": {
            "blur_var": blur,
            "brightness": br,
            "contrast_std": ctr,
            "edge_density": ed,
            "dark_channel": dc,
            "glare_frac": gl,
        }
    }

def choose_shadow_thr(view_primary: str) -> float:
    if view_primary == "blocked":
        return 0.90
    if view_primary in ("fog", "rain"):
        return 0.50
    if view_primary in ("dark", "glare"):
        return 0.56
    if view_primary == "blur":
        return 0.60
    return 0.65

def choose_motion_thr(view_primary: str) -> int:
    if view_primary in ("fog", "rain"):
        return 22
    if view_primary == "glare":
        return 20
    if view_primary == "dark":
        return 12
    return 18


class ShadowModelAdapter:
    """
    Replace shadow_infer() with a real pretrained shadow segmentation model when you have one.
    Output: float32 map in [0,1] shape (H,W) for ROI.
    """
    def __init__(self):
        pass

    def shadow_infer(self, roi_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prob = 1.0 - (gray.astype(np.float32) / 255.0)
        return prob


class AutoCalibrator:
    """
    Learns baseline shadow mass and sets threshold using percentile.
    """
    def __init__(self, warmup_sec=60, fps_assume=12):
        self.warmup_frames = max(1, int(warmup_sec * fps_assume))
        self.values = []
        self.ready = False
        self.thresh = 0.65

    def update(self, shadow_prob: np.ndarray, motion_mask: np.ndarray | None):
        if motion_mask is not None:
            m = (motion_mask > 0).astype(np.float32)
            score = float((shadow_prob * m).mean())
        else:
            score = float(shadow_prob.mean())

        if not self.ready:
            self.values.append(score)
            if len(self.values) >= self.warmup_frames:
                arr = np.array(self.values, dtype=np.float32)
                self.thresh = float(np.percentile(arr, 99.5))
                self.ready = True
        return score, self.ready, self.thresh


COCO_ANIMALS = {"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--rtsp", type=str, default=None)
    ap.add_argument("--roi", nargs=4, type=int, default=None)

    ap.add_argument("--yolo", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--warmup_sec", type=int, default=60)
    ap.add_argument("--fps_assume", type=int, default=12)

    ap.add_argument("--min_shadow_area", type=int, default=700)
    ap.add_argument("--shadow_prob_thresh_min", type=float, default=0.35)

    ap.add_argument("--use_motion_gate", action="store_true")
    ap.add_argument("--motion_thresh", type=int, default=None, help="Override; else adaptive by visibility")

    args = ap.parse_args()

    if args.cam is None and args.rtsp is None:
        raise SystemExit("Provide --cam 0 OR --rtsp <url>")

    src = args.rtsp if args.rtsp else args.cam
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if args.rtsp is None else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    # Force 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    yolo = YOLO(args.yolo)
    shadow_model = ShadowModelAdapter()
    calib = AutoCalibrator(warmup_sec=args.warmup_sec, fps_assume=args.fps_assume)

    prev_gray = None

    print("Live pipeline (720p): YOLO (human/animal) + ShadowProb + Auto-calibration + Visibility detection")
    print("Keys: q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        H, W = frame.shape[:2]
        roi = (0, 0, W - 1, H - 1) if args.roi is None else clamp_roi(tuple(args.roi), W, H)
        x1, y1, x2, y2 = roi
        roi_bgr = frame[y1:y2, x1:x2]

        view = classify_view(roi_bgr)
        view_primary = view["primary"]

        shadow_thr_view = choose_shadow_thr(view_primary)
        motion_thr_view = choose_motion_thr(view_primary)
        motion_thr = int(args.motion_thresh) if args.motion_thresh is not None else motion_thr_view

        motion_mask = None
        if args.use_motion_gate:
            g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = g
                motion_mask = np.zeros_like(g, dtype=np.uint8)
            else:
                d = cv2.absdiff(g, prev_gray)
                prev_gray = g
                motion_mask = (d >= motion_thr).astype(np.uint8) * 255
                motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=2)

        shadow_prob = shadow_model.shadow_infer(roi_bgr).astype(np.float32)
        shadow_prob = np.clip(shadow_prob, 0.0, 1.0)

        shadow_score, ready, thr_auto = calib.update(shadow_prob, motion_mask)

        thr = max(float(args.shadow_prob_thresh_min), min(thr_auto, shadow_thr_view))

        res = yolo.predict(roi_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        names = res.names

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

        obj_mask = np.zeros((roi_bgr.shape[0], roi_bgr.shape[1]), dtype=np.uint8)
        kept_det = 0

        for b in res.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            label = str(names.get(cls, cls))
            if label != "person" and label not in COCO_ANIMALS:
                continue

            kept_det += 1
            bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(overlay, (bx1 + x1, by1 + y1), (bx2 + x1, by2 + y1), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"{label} {conf:.2f}",
                (bx1 + x1, max(0, by1 + y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            h_obj = max(1, by2 - by1)
            w_obj = max(1, bx2 - bx1)
            sx1 = max(0, bx1 - int(0.2 * w_obj))
            sx2 = min(roi_bgr.shape[1], bx2 + int(0.2 * w_obj))
            sy1 = min(roi_bgr.shape[0] - 1, by2)
            sy2 = min(roi_bgr.shape[0], by2 + int(1.2 * h_obj))
            if sy2 > sy1 and sx2 > sx1:
                obj_mask[sy1:sy2, sx1:sx2] = 255

        shadow_mask = (shadow_prob >= thr).astype(np.uint8) * 255

        if motion_mask is not None:
            shadow_mask = cv2.bitwise_and(shadow_mask, motion_mask)

        if kept_det > 0:
            shadow_mask = cv2.bitwise_and(shadow_mask, obj_mask)

        shadow_mask = cv2.medianBlur(shadow_mask, 5)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        cnts, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = 0
        for c in cnts:
            if cv2.contourArea(c) < args.min_shadow_area:
                continue
            blobs += 1
            c_shift = c + np.array([[[x1, y1]]], dtype=np.int32)
            cv2.drawContours(overlay, [c_shift], -1, (0, 0, 255), 2)

        status = "CALIBRATING..." if not ready else f"auto={thr_auto:.3f}"
        cv2.putText(
            overlay,
            f"VIEW={view_primary} thr={thr:.3f} {status} blobs={blobs} det={kept_det}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("NetraAI Live", overlay)
        cv2.imshow("Shadow Mask (ROI)", shadow_mask)
        if motion_mask is not None:
            cv2.imshow("Motion Mask (ROI)", motion_mask)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
