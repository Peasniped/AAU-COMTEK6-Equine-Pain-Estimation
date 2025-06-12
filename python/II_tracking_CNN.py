# II_tracking_CNN.py
import logging
import numpy as np
from ultralytics import YOLO

from _utility import get_center_from_bbox
from YOLO_inference import model

log = logging.getLogger(__name__)

class Feature:
    def __init__(self, bbox, label, track_id):
        x, y, w, h = bbox
        self.bbox       = list(bbox)
        self.label: str = label
        self.track_id   = track_id
        cx, cy = get_center_from_bbox(bbox)
        self.point      = get_center_from_bbox(bbox).reshape(1, 2)  # center (1,2)
        self.trail      = [(int(cx), int(cy))]

    def update(self, new_bbox):
        # update bbox
        self.bbox = list(new_bbox)
        # compute new center, store in point & trail
        cx, cy = get_center_from_bbox(new_bbox)
        self.point      = get_center_from_bbox(new_bbox).reshape(1, 2)  # center (1,2)
        self.trail.append((int(cx), int(cy)))
        if len(self.trail) > 15:
            self.trail.pop(0)

class Tracker:
    def __init__(self, features=None, conf=0.3, iou=0.5):
        """
        features parameter is ignored (kept for drop-in compatibility).
        """
        self.model = model
        self.conf  = conf
        self.iou   = iou
        # track_id → Feature
        self.features: dict[int, Feature] = {}

    def track_frame(self, frame, frame_i=None):
        """
        Runs YOLO.track on the single frame, persists tracks,
        and returns a list of {"feature","from","to","status"} dicts.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            persist=True,
            stream=False
        )[0]

        motions = []
        for box in results.boxes:
            if box.id == None:
                continue

            # extract xyxy, id, cls
            xyxy     = box.xyxy.cpu().numpy().flatten()
            track_id = int(box.id.cpu().numpy())
            cls      = int(box.cls.cpu().numpy())
            label    = self.model.names[cls]

            # convert to (x,y,w,h)
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            bbox = [int(x1), int(y1), int(w), int(h)]

            # compute center
            cx, cy      = get_center_from_bbox(bbox)
            new_center  = (cx, cy)

            if track_id in self.features:
                feat        = self.features[track_id]
                old_center  = tuple(feat.point.tolist())
                feat.update(bbox)
                from_pt, to_pt = old_center, new_center
                status = True
            else:
                feat        = Feature(bbox, label, track_id)
                self.features[track_id] = feat
                from_pt = to_pt = new_center
                status = True

            motions.append({
                "feature": feat,
                "from":    from_pt,
                "to":      to_pt,
                "status":  status
            })

        return motions
