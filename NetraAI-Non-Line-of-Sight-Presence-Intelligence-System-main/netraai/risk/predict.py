import argparse
import pandas as pd

from netraai.common.schemas import RiskFeatures
from netraai.risk.model import RiskModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to risk_model.joblib")
    ap.add_argument("--timestamp", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--camera_status", required=True, choices=["on", "off", "blocked"])
    ap.add_argument("--patrol_frequency_per_hr", required=True, type=float)
    args = ap.parse_args()

    features = RiskFeatures(
        timestamp=args.timestamp,
        weather=args.weather,
        camera_status=args.camera_status,
        patrol_frequency_per_hr=args.patrol_frequency_per_hr,
    )

    rm = RiskModel.load(args.model)
    df = pd.DataFrame([features.model_dump()])
    labels, scores = rm.predict(df)

    print(
        {
            "risk_label": labels[0],
            "risk_score": float(scores[0]),
            "input": features.model_dump(),
        }
    )


if __name__ == "__main__":
    main()
