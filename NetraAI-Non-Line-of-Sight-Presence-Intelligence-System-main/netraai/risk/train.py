import argparse
import pandas as pd

from netraai.common.logging import get_logger
from .model import RiskModel

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path with columns + risk_label")
    ap.add_argument("--out", required=True, help="Output model path (joblib)")
    ap.add_argument("--threshold_high", type=float, default=0.70)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "risk_label" not in df.columns:
        raise ValueError("CSV must contain 'risk_label' column (Low/Medium/High)")

    y = df["risk_label"].astype(str)
    Xraw = df.drop(columns=["risk_label"])

    model = RiskModel.fresh(threshold_high=args.threshold_high)
    model.fit(Xraw, y)
    model.save(args.out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
