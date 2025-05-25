import cv2
import numpy as np

def extract_frames(video_path) -> list[np.ndarray]:
    print("Extracting frames...")
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
    return np.array([[center_x, center_y]], dtype=np.float32)

def visualize_frame(
        frame,
        frame_i,
        features,
        segmenter=None,
        show_frame_i=True,
        show_trails=True,
        show_bbox=True,
        show_labels=True,
        show_active_segments=True,
        show_completed_segments=False,
        show_relative_trails=True,  # <-- NEW ARG
        window_name="Tracking"
    ):
    image = frame.copy()

    if show_frame_i:
        (_, text_height), _ = cv2.getTextSize(str(frame_i), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(image, str(frame_i), (5, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 235, 0), 2)

    for feature in features:
        # Draw the current point
        x, y = map(int, feature.point[0])
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
            traj = seg['trajectory']
            for i in range(1, len(traj)):
                pt1 = tuple(map(int, traj[i-1]))
                pt2 = tuple(map(int, traj[i]))
                cv2.line(image, pt1, pt2, (0,0,255), 2)  # Red line for active segment

    # Optional: Draw completed segments (e.g. in green)
    if show_completed_segments and segmenter is not None:
        for seg in segmenter.completed_segments:
            traj = seg['trajectory']
            for i in range(1, len(traj)):
                pt1 = tuple(map(int, traj[i-1]))
                pt2 = tuple(map(int, traj[i]))
                cv2.line(image, pt1, pt2, (0,200,0), 2)  # Green for completed

    cv2.imshow(window_name, image)


def get_head_centers_from_motions(motions):
    """Returns head center positions as (from, to) from motions, or (None, None) if not found."""
    head_motion = next((m for m in motions if "head" in m['feature'].label), None)
    if head_motion is not None:
        head_from = np.array(head_motion['from'])
        head_to = np.array(head_motion['to'])
    else:
        head_from = head_to = None
    return head_from, head_to