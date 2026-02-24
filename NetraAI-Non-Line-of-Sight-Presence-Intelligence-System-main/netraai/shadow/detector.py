from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterator, Tuple

import cv2
import numpy as np

from netraai.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShadowDetectorConfig:
    background_history: int = 350
    min_area: int = 350
    max_frames: int = 900

    anomaly_threshold: float = 0.65
    ema_alpha: float = 0.25 

    brightness_jump_threshold: float = 18.0  

    warmup_frames: int = 80            # collect samples before scoring
    window_keep: int = 400             # keep last N values for stats
    high_percentile: float = 95.0      # percentile used for normalization
    min_motion_samples: int = 30       

    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass
class ShadowDetectionResult:
    suspicious: bool
    anomaly_score: float
    debug: Optional[Dict[str, Any]] = None


def frames_from_video(path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _apply_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


class ShadowDetector:
    """
    Stage 2.1 Stable Shadow-like Motion Anomaly Detector

    Key improvements over Stage 2:
    - Skips frames with global brightness jumps (flicker / headlights / auto-exposure)
    - EMA smoothing of area & centroid speed time series
    - Robust percentile-based normalization => avoids instant saturation to 1.0
    - Optional ROI for analyzing only blind-zone boundary region
    """

    def __init__(self, cfg: ShadowDetectorConfig = ShadowDetectorConfig()):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.background_history,
            detectShadows=True
        )

    def analyze(self, frames: Iterator[np.ndarray]) -> ShadowDetectionResult:
        areas_raw: list[float] = []
        speeds_raw: list[float] = []

        areas_ema: list[float] = []
        speeds_ema: list[float] = []

        prev_centroid: Optional[Tuple[float, float]] = None
        prev_brightness: Optional[float] = None

        used = 0
        skipped_brightness = 0
        motion_frames = 0

        # helper for EMA
        def ema(prev: Optional[float], x: float, a: float) -> float:
            return x if prev is None else (a * x + (1.0 - a) * prev)

        ema_area: Optional[float] = None
        ema_speed: Optional[float] = None

        for frame in frames:
            used += 1
            if used > self.cfg.max_frames:
                break

            frame = _apply_roi(frame, self.cfg.roi)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            brightness = float(np.mean(gray))
            if prev_brightness is not None:
                if abs(brightness - prev_brightness) >= self.cfg.brightness_jump_threshold:
                    skipped_brightness += 1
                    prev_brightness = brightness
                    continue
            prev_brightness = brightness

            fg = self.bg.apply(gray)

            mask = cv2.threshold(fg, 40, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= self.cfg.min_area]
            if not cnts:
                continue

            motion_frames += 1

            c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            areas_raw.append(area)

            # centroid speed
            speed = 0.0
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                if prev_centroid is not None:
                    dx = cx - prev_centroid[0]
                    dy = cy - prev_centroid[1]
                    speed = float(np.hypot(dx, dy))
                prev_centroid = (cx, cy)

            speeds_raw.append(speed)

            ema_area = ema(ema_area, area, self.cfg.ema_alpha)
            ema_speed = ema(ema_speed, speed, self.cfg.ema_alpha)
            areas_ema.append(float(ema_area))
            speeds_ema.append(float(ema_speed))

            if len(areas_ema) > self.cfg.window_keep:
                areas_ema = areas_ema[-self.cfg.window_keep:]
                speeds_ema = speeds_ema[-self.cfg.window_keep:]

        if motion_frames < self.cfg.min_motion_samples or len(areas_ema) < self.cfg.min_motion_samples:
            return ShadowDetectionResult(
                suspicious=False,
                anomaly_score=0.0,
                debug={
                    "frames_used": used,
                    "motion_frames": motion_frames,
                    "skipped_brightness": skipped_brightness,
                    "reason": "insufficient_motion",
                },
            )

        a = areas_ema[self.cfg.warmup_frames:] if len(areas_ema) > self.cfg.warmup_frames else areas_ema
        s = speeds_ema[self.cfg.warmup_frames:] if len(speeds_ema) > self.cfg.warmup_frames else speeds_ema

        area_var = float(np.var(a))
        speed_var = float(np.var(s))

        a_p = float(np.percentile(a, self.cfg.high_percentile))
        s_p = float(np.percentile(s, self.cfg.high_percentile))

        a_p = max(a_p, 1.0)
        s_p = max(s_p, 0.5)

        area_std = float(np.sqrt(max(area_var, 0.0)))
        speed_std = float(np.sqrt(max(speed_var, 0.0)))

        area_score = min(1.0, area_std / (0.75 * a_p))
        speed_score = min(1.0, speed_std / (0.75 * s_p))

        anomaly = float(0.55 * area_score + 0.45 * speed_score)
        suspicious = anomaly >= self.cfg.anomaly_threshold

        return ShadowDetectionResult(
            suspicious=bool(suspicious),
            anomaly_score=anomaly,
            debug={
                "frames_used": used,
                "motion_frames": motion_frames,
                "skipped_brightness": skipped_brightness,
                "area_std": area_std,
                "speed_std": speed_std,
                "area_pctl": a_p,
                "speed_pctl": s_p,
                "area_score": area_score,
                "speed_score": speed_score,
                "threshold": self.cfg.anomaly_threshold,
                "min_area": self.cfg.min_area,
                "roi": self.cfg.roi,
                "brightness_jump_threshold": self.cfg.brightness_jump_threshold,
                "ema_alpha": self.cfg.ema_alpha,
            },
        )
