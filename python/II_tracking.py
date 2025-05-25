import  cv2
import  numpy as np

from _utility   import get_center_from_bbox

lk_params = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)



class Feature:
    def __init__(self, bbox, label):
        x, y, w, h      = bbox
        self.bbox       = list(bbox)
        self.label:str  = label
        self.point      = get_center_from_bbox(bbox).reshape(1, 2)  # center (1,2)
        self.size       = (w, h)
        self.trail      = []
        self.was_reset  = False

    def update_point(self, new_point):
        # new_point shape: (1, 2)
        self.point = new_point
        cx, cy = new_point[0]
        w, h = self.size
        # Update bbox so its center follows the tracked point
        self.bbox[0] = int(cx - w // 2)
        self.bbox[1] = int(cy - h // 2)
        self.trail.append((int(cx), int(cy)))
        if len(self.trail) > 15:
            self.trail.pop(0)
        
class Tracker:
    def __init__(self, features):
        self.features = features  # List[Feature]
        self.prev_gray = None

    def get_points_array(self):
        points = np.array([f.point for f in self.features], dtype=np.float32)
        return points.reshape(-1, 1, 2)

    def track_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return []

        points = self.get_points_array()
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, points, None, **lk_params)

        motion_list = []
        for feature, old_pt, new_pt, _status in zip(self.features, points, next_points, status):
            feature.update_point(new_pt)
            from_xy = tuple(old_pt[0])
            to_xy = tuple(new_pt[0])
            motion_list.append({
                "feature": feature,
                "from": from_xy,
                "to": to_xy,
                "status": bool(_status)})
            
            if _status:  # Only update if tracking succeeded
                feature.point = new_pt

        self.prev_gray = gray
        return motion_list
