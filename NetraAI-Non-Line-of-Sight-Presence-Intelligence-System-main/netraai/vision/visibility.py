import cv2
import numpy as np

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
    return float(np.mean(edges > 0))  # 0..1

def dark_channel_mean(bgr, ksize=15) -> float:
    """
    Fog/haze tends to increase the dark channel (it becomes less "dark").
    This is a common haze indicator.
    """
    b, g, r = cv2.split(bgr)
    m = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dark = cv2.erode(m, kernel)
    return float(np.mean(dark) / 255.0)  # 0..1

def glare_score(bgr) -> float:
    """
    Glare/headlight bloom: many pixels saturated (very bright).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    sat_bright = np.mean(v >= 245)  # fraction 0..1
    return float(sat_bright)

def rain_streak_score(bgr) -> float:
    """
    Simple heuristic: rain adds lots of tiny bright streak edges.
    We approximate by high edge density + moderate brightness.
    """
    ed = edge_density(bgr)
    br = brightness_mean(bgr)
    return float(ed * (1.0 if br > 40 else 0.5))

def classify_view(bgr) -> dict:
    """
    Returns a robust view/visibility diagnosis.
    Thresholds are heuristic; tune per camera.
    """
    blur = blur_score_laplacian(bgr)
    br = brightness_mean(bgr)
    ctr = contrast_std(bgr)
    ed = edge_density(bgr)
    dc = dark_channel_mean(bgr)
    gl = glare_score(bgr)
    rn = rain_streak_score(bgr)

    # Hard failures first
    blocked = (ctr < 8 and ed < 0.01)  # almost no texture
    dark = (br < 18)
    blurry = (blur < 45)

    # Fog/haze tends: low contrast + low edges + higher dark-channel mean
    foggy = (ctr < 22 and ed < 0.03 and dc > 0.22 and not dark)

    # Glare: lots of saturated pixels
    glare = (gl > 0.03)

    # Rain: lots of small edges + not too dark
    rainy = (rn > 0.06 and not foggy and not blocked)

    # Decide primary label
    labels = []
    if blocked: labels.append("blocked")
    if dark: labels.append("dark")
    if glare: labels.append("glare")
    if foggy: labels.append("fog")
    if rainy: labels.append("rain")
    if blurry and not blocked: labels.append("blur")

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
            "rain_score": rn
        }
    }
