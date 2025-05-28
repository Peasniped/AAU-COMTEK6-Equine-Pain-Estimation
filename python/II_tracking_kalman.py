import  cv2
import  logging
import  numpy as np

from _utility   import get_center_from_bbox

log = logging.getLogger(__name__)

lk_params = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def create_kalman_filter() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    return kf

class Feature:
    def __init__(self, bbox, label):
        x, y, w, h      = bbox
        self.bbox       = list(bbox)
        self.label:str  = label
        self.point      = get_center_from_bbox(bbox).reshape(-1)  # shape (2,)
        self.size       = (w, h)
        self.trail      = []
        self.was_reset  = False
        self.kf         = create_kalman_filter()
        log.debug(f"Initializing Feature {label} with bbox {bbox}, initial point: {self.point}, shape: {self.point.shape}")


    def update_point(self, new_point):
            new_point = np.array(new_point).flatten()
            if new_point.shape[0] != 2:
                raise ValueError(f"update_point expected 2D input, got {new_point.shape}: {new_point}")
            cx, cy = new_point
            w, h = self.size
            # Update bbox so its center follows the tracked point
            self.bbox[0] = int(cx - w // 2)
            self.bbox[1] = int(cy - h // 2)
            self.kf.correct(new_point.reshape(2,1))
            point_filtered = self.kf.predict()[:2].flatten()

            px, py = int(point_filtered[0]), int(point_filtered[1])
            self.point = np.array(point_filtered).reshape(-1) # Force flatten
            self.trail.append((int(px), int(py)))
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
            old_pt = np.array(old_pt).flatten()
            new_pt = np.array(new_pt).flatten()
            if new_pt.shape[0] != 2:
                raise ValueError(f"Optical flow new_pt bad shape: {new_pt.shape}, value: {new_pt}")
            feature.update_point(new_pt)
            from_xy = tuple(old_pt)  # (x, y)
            to_xy = tuple(new_pt)    # (x, y)
            motion_list.append({
                "feature": feature,
                "from": from_xy,
                "to": to_xy,
                "status": bool(_status)})
            
            if _status:  # Only update if tracking succeeded
                feature.point = np.array(new_pt).reshape(-1)

        self.prev_gray = gray
        return motion_list
