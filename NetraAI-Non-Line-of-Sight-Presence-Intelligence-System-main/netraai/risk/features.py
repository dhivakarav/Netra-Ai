import pandas as pd

WEATHER_CATS = ["clear", "cloudy", "rain", "fog", "storm"]
CAMERA_CATS = ["on", "off", "blocked"]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input expected columns:
      - timestamp (ISO8601 string)
      - weather (string)
      - camera_status (on/off/blocked)
      - patrol_frequency_per_hr (float)
    Output: numeric feature frame for ML
    """
    out = pd.DataFrame()

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    out["hour"] = ts.dt.hour.fillna(0).astype(int)
    out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)

    out["patrol_frequency_per_hr"] = df["patrol_frequency_per_hr"].astype(float)

    weather = df["weather"].fillna("unknown").astype(str).str.lower().str.strip()
    cam = df["camera_status"].fillna("unknown").astype(str).str.lower().str.strip()

    for c in WEATHER_CATS:
        out[f"weather__{c}"] = (weather == c).astype(int)
    out["weather__other"] = (~weather.isin(WEATHER_CATS)).astype(int)

    for c in CAMERA_CATS:
        out[f"camera__{c}"] = (cam == c).astype(int)
    out["camera__other"] = (~cam.isin(CAMERA_CATS)).astype(int)

    return out
