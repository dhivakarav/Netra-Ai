import cv2
import numpy as np

def hue_circ_diff(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 180 - d)

class RunningBG:
    def __init__(self, alpha=0.02):
        self.alpha = float(alpha)
        self.bg = None  # float32 BGR

    def update(self, bgr):
        f = bgr.astype(np.float32)
        if self.bg is None:
            self.bg = f
        else:
            cv2.accumulateWeighted(f, self.bg, self.alpha)

    def get(self):
        if self.bg is None:
            return None
        return np.clip(self.bg, 0, 255).astype(np.uint8)

def shadow_prob_hsv(curr_bgr, bg_bgr, v_low=0.55, v_high=0.98, s_diff_max=35.0, h_diff_max=12.0):
    curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    Hc, Sc, Vc = cv2.split(curr)
    Hb, Sb, Vb = cv2.split(bg)

    eps = 1e-6
    v_ratio = Vc / (Vb + eps)
    dh = hue_circ_diff(Hc, Hb)
    ds = np.abs(Sc - Sb)

    # soft probability (not hard mask)
    pv = np.clip((v_high - v_ratio) / max(1e-6, (v_high - v_low)), 0, 1)  # higher when darker
    ps = np.clip(1 - (ds / max(1e-6, s_diff_max)), 0, 1)
    ph = np.clip(1 - (dh / max(1e-6, h_diff_max)), 0, 1)

    prob = (0.55 * pv + 0.25 * ps + 0.20 * ph)
    return prob.astype(np.float32)

def motion_mask_frame_diff(gray, prev_gray, motion_thresh=18):
    if prev_gray is None:
        return np.zeros_like(gray, dtype=np.uint8), gray
    diff = cv2.absdiff(gray, prev_gray)
    mm = (diff >= motion_thresh).astype(np.uint8) * 255
    mm = cv2.dilate(mm, np.ones((3,3), np.uint8), iterations=2)
    return mm, gray

def prob_to_mask(prob, thr=0.65):
    m = (prob >= thr).astype(np.uint8) * 255
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    return m
