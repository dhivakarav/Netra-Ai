from dataclasses import dataclass
from netraai.common.schemas import RiskResult, ShadowResult, Alert


@dataclass
class DecisionEngine:
    """
    Stage 2 decision rule:
      alert ONLY if (risk is High) AND (shadow suspicious == True)
    """

    def should_alert(self, risk: RiskResult, shadow: ShadowResult) -> bool:
        return (risk.risk_label == "High") and bool(shadow.suspicious)

    def build_alert(self, zone_id: str, timestamp: str, risk: RiskResult, shadow: ShadowResult) -> Alert:
        return Alert(
            zone_id=zone_id,
            timestamp=timestamp,
            risk=risk,
            shadow=shadow,
            reason="High risk + suspicious shadow-like motion",
        )
