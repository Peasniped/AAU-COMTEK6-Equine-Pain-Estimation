import cv2
import csv
import glob
import logging
import numpy as np
import os
from datetime import datetime
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

def setup_root_logger(log_filename="logs/ear_rotations.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger_format = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"

    # Prevent duplicate logs
    if not logger.hasHandlers():
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(logger_format)
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(logger_format)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger


def extract_frames(video_path) -> list[np.ndarray]:
    log.info("Extracting frames...")
    capture = cv2.VideoCapture(video_path)
    frames = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames

def get_center_from_bbox(bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    return np.array([center_x, center_y], dtype=np.float32)  # shape (2,)
    return np.array([[center_x, center_y]], dtype=np.float32)


def visualize_frame(
        frames,
        frame_i,
        features,
        segmenter=None,
        show_frame_i=True,
        show_trails=True,
        show_bbox=True,
        show_labels=True,
        show_active_segments=True,
        show_completed_segments=False,
        show_relative_trails=True,
        video_name=False
    ):
    image = frames[frame_i].copy()
    WINDOW_NAME = "Ear-Rotation Detector"

    if video_name:
        (_, text_height), _ = cv2.getTextSize(str(video_name), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(image, video_name, (5, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2)

    if show_frame_i:
        displacement = text_height + 10 if video_name else 0
        (_, text_height), _ = cv2.getTextSize(str(frame_i), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(image, f"{frame_i} of {len(frames)}", (5, text_height + 5 + displacement), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 255, 30), 2)


    for _, feature in features.items():
        # Draw the current point
        point = feature.point[0]
        x, y = int(point[0]), int(point[1])
        color = (30, 235, 30) if feature.label == "ear" else (30, 30, 235)
        cv2.circle(image, (x, y), 4, color, -1)
        
        # Draw the trail (motion history, absolute)
        if show_trails and hasattr(feature, "trail"):
            for i in range(1, len(feature.trail)):
                cv2.line(image, feature.trail[i-1], feature.trail[i], color, 1)

        # --- DRAW RELATIVE TRAIL (head-centric) ---
        if show_relative_trails and hasattr(feature, "rel_trail"):
            # Try to get the head position for this frame (for offset)
            if segmenter is not None and hasattr(segmenter, "last_head_to") and segmenter.last_head_to is not None:
                head_point = np.array(segmenter.last_head_to)
            else:
                head_point = np.array([0, 0])  # fallback: origin

            rel_color = (235, 30, 30)
            rel_points = [tuple((np.array(pt) + head_point).astype(int)) for pt in feature.rel_trail]
            for i in range(1, len(rel_points)):
                cv2.line(image, rel_points[i - 1], rel_points[i], rel_color, 2)
        
        # Draw the bounding box
        if show_bbox and hasattr(feature, "bbox"):
            x0, y0, w, h = feature.bbox
            top_left = (int(x0), int(y0))
            bottom_right = (int(x0 + w), int(y0 + h))
            cv2.rectangle(image, top_left, bottom_right, color, 2)
        
        # Optionally, draw label
        if show_labels:
            cv2.putText(image, feature.label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- Draw live (active) segment trajectories ---
    if show_active_segments and segmenter is not None:
        for seg in segmenter.active_segments.values():
            traj = seg["trajectory"]
            for i in range(1, len(traj)):
                pt1 = tuple(map(int, traj[i-1]))
                pt2 = tuple(map(int, traj[i]))
                cv2.line(image, pt1, pt2, (0,0,255), 2)  # Red line for active segment

    # Optional: Draw completed segments (e.g. in green)
    if show_completed_segments and segmenter is not None:
        for seg in segmenter.completed_segments:
            traj = seg["trajectory"]
            for i in range(1, len(traj)):
                pt1 = tuple(map(int, traj[i-1]))
                pt2 = tuple(map(int, traj[i]))
                cv2.line(image, pt1, pt2, (0,200,0), 2)  # Green for completed

    cv2.imshow(WINDOW_NAME, image)


def get_head_centers_from_motions(motions):
    """Returns head center positions as (from, to) from motions, or (None, None) if not found."""
    head_motion = next((m for m in motions if "head" in m["feature"].label), None)
    if head_motion is not None:
        head_from = np.array(head_motion["from"])
        head_to = np.array(head_motion["to"])
    else:
        head_from = head_to = None
    return head_from, head_to

def _get_csv_filename() -> str:
    date_str = datetime.now(tz=ZoneInfo("Europe/Copenhagen")).strftime("%Y-%m-%d")
    DIRECTORY = "test_and_validation"
    pattern = os.path.join(DIRECTORY, f"results_{date_str}_*.csv")
    existing_files = glob.glob(pattern)
    increments = []
    for f in existing_files:
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) >= 3:
            inc_str = parts[-1].replace('.csv', '')
            if inc_str.isdigit():
                increments.append(int(inc_str))
    next_inc = (max(increments) if increments else 0) + 1
    filename = f"results_{date_str}_{next_inc:02d}.csv"
    return os.path.join(DIRECTORY, filename)

def export_labelled_segments(labelled_segments:list, video_name: str, _print: bool = True, to_csv: bool = False, csv_path: str = None) -> None:
    counter = 1
    events = []
    for segment in labelled_segments:
        if not segment:
            continue
        frame_start = segment["start"]
        frame_end   = segment["end"]
        code        = segment["equifacs_code"]
        message     = f"{video_name} - Event {counter}: {code} from {frame_start} to {frame_end}"
        counter     += 1
        events.append({"message": message, "csv_row": (video_name, code, frame_start, frame_end)})
    
    if _print:
        log.info(f"\n-- Labelled a total of {counter} movement segments:\n")
        for event in events:
            log.info(event["message"])
    
    if to_csv:
        csv_path = csv_path if csv_path else _get_csv_filename()
        HEADERS = ["video_name","code","start_frame","end_frame"]
        
        # Check if file exists and if it"s empty
        write_headers = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_headers:
                writer.writerow(HEADERS)
            for event in events:
                writer.writerow((event["csv_row"]))