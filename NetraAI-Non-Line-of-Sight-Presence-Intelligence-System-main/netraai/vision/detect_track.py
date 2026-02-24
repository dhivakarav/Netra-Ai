from ultralytics import YOLO

COCO_ANIMALS = {"bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"}

class DetectorTracker:
    def __init__(self, weights="yolov8s.pt", conf=0.35, imgsz=960, device=None):
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device

    def track(self, frame_bgr):
        # persist=True keeps tracker state across frames
        res = self.model.track(
            frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device
        )[0]
        names = res.names
        out = []
        if res.boxes is None:
            return out

        for b in res.boxes:
            cls = int(b.cls.item())
            label = str(names.get(cls, cls))
            if label != "person" and label not in COCO_ANIMALS:
                continue
            tid = int(b.id.item()) if b.id is not None else -1
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            out.append({"id": tid, "label": label, "conf": conf, "xyxy": (x1,y1,x2,y2)})
        return out
