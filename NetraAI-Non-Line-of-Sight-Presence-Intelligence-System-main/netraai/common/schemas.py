from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

RiskLabel = Literal["Low", "Medium", "High"]
CameraStatus = Literal["on", "off", "blocked"]


class RiskFeatures(BaseModel):
    timestamp: str = Field(..., description="ISO8601 time string, e.g. 2026-02-06T10:00:00Z")
    weather: str = Field(..., description="e.g. clear/cloudy/rain/fog/storm")
    camera_status: CameraStatus
    patrol_frequency_per_hr: float


class RiskResult(BaseModel):
    risk_label: RiskLabel
    risk_score: float


class ShadowResult(BaseModel):
    suspicious: bool
    anomaly_score: float
    debug: Optional[Dict[str, Any]] = None


class Alert(BaseModel):
    zone_id: str
    timestamp: str
    risk: RiskResult
    shadow: ShadowResult
    reason: str
