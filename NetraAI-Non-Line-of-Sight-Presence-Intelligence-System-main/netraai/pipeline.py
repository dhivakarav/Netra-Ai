from __future__ import annotations

import os
import pandas as pd

from netraai.common.schemas import RiskFeatures, RiskResult, ShadowResult
from netraai.risk.model import RiskModel
from netraai.shadow.detector import ShadowDetector, ShadowDetectorConfig, frames_from_video
from netraai.decision.engine import DecisionEngine
from netraai.io.alerts import save_alert
from netraai.common.logging import get_logger
from netraai.shadow.detector import ShadowDetectorConfig

shadow_cfg = ShadowDetectorConfig(
    anomaly_threshold=0.70,
    brightness_jump_threshold=20.0,
    min_area=450,
    roi=None,  # or (x1,y1,x2,y2)
)

logger = get_logger(__name__)


def run_stage2_once(
    *,
    zone_id: str,
    features: RiskFeatures,
    risk_model_path: str,
    video_path: str | None,
    alerts_dir: str = "outputs/alerts",
    shadow_cfg: ShadowDetectorConfig | None = None,
):
    # ---- 1) Risk prediction ----
    rm = RiskModel.load(risk_model_path)
    df = pd.DataFrame([features.model_dump()])
    labels, scores = rm.predict(df)

    risk = RiskResult(risk_label=str(labels[0]), risk_score=float(scores[0]))
    logger.info("Risk: %s (%.3f)", risk.risk_label, risk.risk_score)

    # ---- 2) Shadow analysis (gated) ----
    if risk.risk_label != "High":
        shadow = ShadowResult(suspicious=False, anomaly_score=0.0, debug={"gated": True, "why": "risk_not_high"})
    elif not video_path:
        shadow = ShadowResult(suspicious=False, anomaly_score=0.0, debug={"gated": True, "why": "no_video"})
    else:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        det = ShadowDetector(shadow_cfg or ShadowDetectorConfig())
        res = det.analyze(frames_from_video(video_path))
        shadow = ShadowResult(suspicious=bool(res.suspicious), anomaly_score=float(res.anomaly_score), debug=res.debug)

    logger.info("Shadow suspicious=%s score=%.3f", shadow.suspicious, shadow.anomaly_score)

    # ---- 3) Decision engine ----
    de = DecisionEngine()
    if de.should_alert(risk, shadow):
        alert = de.build_alert(zone_id, features.timestamp, risk, shadow)
        path = save_alert(alert, alerts_dir)
        logger.info("ALERT SAVED: %s", path)
        return {"alert_path": path, "alert": alert.model_dump()}

    return {"alert_path": None, "risk": risk.model_dump(), "shadow": shadow.model_dump()}
