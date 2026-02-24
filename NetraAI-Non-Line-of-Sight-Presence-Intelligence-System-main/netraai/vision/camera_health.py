import cv2
import numpy as np

def laplacian_blur_score(frame_bgr):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def is_dark(frame_bgr, thr=18.0):
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(g)) < thr

def is_blocked(frame_bgr, blur_thr=35.0, dark_thr=18.0):
    # practical “blocked/covered/too blurry/too dark” heuristic
    blur = laplacian_blur_score(frame_bgr)
    dark = is_dark(frame_bgr, dark_thr)
    return (blur < blur_thr) or dark, {"blur_score": blur, "dark": dark}
