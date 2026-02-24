import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class TrackState:
    last_seen_ts: float
    last_xy: tuple
    last_box: tuple
    label: str
    conf: float
    missing_frames: int = 0
    shadow_persist_frames: int = 0

def point_in_poly(pt, poly):
    # poly: list of (x,y)
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0

class BlindSpotReasoner:
    def __init__(self, blind_polys, disappear_frames=8, shadow_need=6):
        self.blind_polys = blind_polys
        self.disappear_frames = int(disappear_frames)
        self.shadow_need = int(shadow_need)
        self.tracks = {}

    def update_seen(self, tid, xy, box, label, conf, ts):
        st = self.tracks.get(tid)
        if st is None:
            st = TrackState(ts, xy, box, label, conf)
        st.last_seen_ts = ts
        st.last_xy = xy
        st.last_box = box
        st.label = label
        st.conf = conf
        st.missing_frames = 0
        st.shadow_persist_frames = 0
        self.tracks[tid] = st

    def mark_missing(self, tid):
        if tid in self.tracks:
            self.tracks[tid].missing_frames += 1

    def update_shadow_persist(self, tid, has_shadow):
        if tid in self.tracks:
            if has_shadow:
                self.tracks[tid].shadow_persist_frames += 1

    def is_in_blind(self, xy):
        for p in self.blind_polys:
            if point_in_poly(xy, p):
                return True
        return False

    def evaluate_hidden(self, tid):
        st = self.tracks.get(tid)
        if not st:
            return False, {}
        in_blind = self.is_in_blind(st.last_xy)
        hidden = (in_blind and st.missing_frames >= self.disappear_frames and st.shadow_persist_frames >= self.shadow_need)
        return hidden, {
            "in_blind": in_blind,
            "missing_frames": st.missing_frames,
            "shadow_persist_frames": st.shadow_persist_frames
        }
