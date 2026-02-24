import os, json
from datetime import datetime

def save_alert(payload: dict, out_dir: str = "outputs/alerts") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = payload.get("timestamp", datetime.utcnow().isoformat()).replace(":", "").replace("-", "")
    path = os.path.join(out_dir, f"alert_{payload.get('zone_id','Z')}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
