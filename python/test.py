import  cv2
import  numpy as np
from    collections import deque

from YOLO_inference import detect_horse_features, model

def extract_frames(video_path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    frames = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames

def detect_head_and_ears(frames: list, show_initial_detections=False, model_confidence=0.5, retries: int = 50):
    for attempt in range(retries):
        img, detected_objects = detect_horse_features(model, frames[attempt], model_confidence=model_confidence)
        ear_positions = [obj["bbox"] for obj in detected_objects if obj["label"] == "ear"]
        head_position = [obj["bbox"] for obj in detected_objects if obj["label"] == "head"]
        cv2.imshow(f'Detections on image', img)
        cv2.waitKey(100)
        if len(ear_positions) == 0 or len(head_position) == 0:
            print(f"Attempt {attempt + 1}: No ears or head found in video frame, trying again")
            continue
        else:
            if show_initial_detections and img.any():
                cv2.imshow(f'Detections on image', img)
                #cv2.waitKey(0)
                _input = input("Press enter key to continue or n + enter to go to the next frame or x + enter to exit:\n > ")
                if _input == "x":
                    cv2.destroyAllWindows()
                    exit()
                elif _input == "n":
                    continue
            cv2.destroyAllWindows()
            break
    
    if len(ear_positions) == 0 or len(head_position) == 0:
        print(f"Head or ears not found in {retries} attempts. Exiting")
        exit()
    return ear_positions, head_position

def get_points_from_bbox(bbox):
    x, y, w, h = bbox
    center_x = x + w // 2
    center_y = y + h // 2
    return np.array([[center_x, center_y]], dtype=np.float32)

def create_kalman_filter():
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

def track_and_visualize(frames, ear_bboxes, head_bbox, movement_threshold=0.03, box_thickness=1):
    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    ear_points = np.array([get_points_from_bbox(bbox) for bbox in ear_bboxes], dtype=np.float32).reshape(-1, 1, 2)
    head_point = get_points_from_bbox(head_bbox).reshape(1, 1, 2)

    ear_box_sizes = [(w, h) for (_, _, w, h) in ear_bboxes]
    head_box_size = (head_bbox[2], head_bbox[3])

    ear_trails = [[] for _ in range(len(ear_points))]
    rel_movement_buffers = [deque(maxlen=5) for _ in range(len(ear_points))]
    max_trail_length = 15

    # Rotation state tracking
    rotation_states = ["idle"] * len(ear_points)
    below_threshold_counters = [0] * len(ear_points)
    stop_grace_period = 5

    stabilization_frames = 10  # Skip detection in first few frames

    # Kalman filters
    ear_kalman_filters = [create_kalman_filter() for _ in range(len(ear_points))]
    head_kf = create_kalman_filter()

    head_init = head_point[0][0].reshape(2, 1)
    head_kf.statePre[:2] = head_init
    head_kf.statePost[:2] = head_init
    head_kf.statePre[2:] = 0
    head_kf.statePost[2:] = 0

    for idx, kf in enumerate(ear_kalman_filters):
        ear_init = ear_points[idx][0].reshape(2, 1)
        kf.statePre[:2] = ear_init
        kf.statePost[:2] = ear_init
        kf.statePre[2:] = 0
        kf.statePost[2:] = 0

    # Saturate Kalman filters with repeated initial positions
    for _ in range(5):
        head_kf.correct(head_init)
        head_kf.predict()
        for idx, kf in enumerate(ear_kalman_filters):
            kf.correct(ear_points[idx][0].reshape(2, 1))
            kf.predict()

    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_ear_points, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, ear_points, None, **lk_params)
        next_head_point, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, head_point, None, **lk_params)

        head_kf.correct(next_head_point[0][0].reshape(2, 1))
        head_filtered = head_kf.predict()[:2].flatten()
        head_displacement = head_filtered - head_point[0][0]

        # Draw head box
        hx, hy = int(head_filtered[0]), int(head_filtered[1])
        hw, hh = head_box_size
        cv2.rectangle(frame, (hx - hw // 2, hy - hh // 2), (hx + hw // 2, hy + hh // 2), (200, 200, 0), 1)

        for idx, (old_ear, new_ear) in enumerate(zip(ear_points, next_ear_points)):
            ear_kf = ear_kalman_filters[idx]
            ear_kf.correct(new_ear[0].reshape(2, 1))
            ear_filtered = ear_kf.predict()[:2].flatten()

            ear_displacement = ear_filtered - old_ear[0]
            relative_displacement = ear_displacement - head_displacement
            rel_movement = np.linalg.norm(relative_displacement)

            rel_movement_buffers[idx].append(rel_movement)
            avg_rel_movement = np.mean(rel_movement_buffers[idx])

            cx, cy = int(ear_filtered[0]), int(ear_filtered[1])
            box_w, box_h = ear_box_sizes[idx]
            top_left = (cx - box_w // 2, cy - box_h // 2)
            bottom_right = (cx + box_w // 2, cy + box_h // 2)

            movement_threshold_pixels = (box_w + box_h) / 2 * movement_threshold
            HEAD_FAST_THRESHOLD = 10.0
            RELATIVE_FACTOR = 1.5
            head_speed = np.linalg.norm(head_displacement)

            is_moving = avg_rel_movement > movement_threshold_pixels and (
                head_speed < HEAD_FAST_THRESHOLD or avg_rel_movement > head_speed * RELATIVE_FACTOR
            )

            # Draw ear center
            cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

            # Draw flow vector (optical flow) for the ear
            old_pos = tuple(np.int32(old_ear[0]))
            new_pos = tuple(np.int32(new_ear[0]))
            cv2.arrowedLine(frame, old_pos, new_pos, (0,128,255), 2, tipLength=0.4)

            # Draw head center and vector
            cv2.circle(frame, (hx, hy), 5, (255,0,255), -1)
            cv2.arrowedLine(frame, tuple(np.int32(head_point[0][0])), (hx, hy), (255,0,0), 2, tipLength=0.4)

            state = rotation_states[idx]

            if i >= stabilization_frames:
                if state == "idle":
                    if is_moving:
                        rotation_states[idx] = "rotating"
                        below_threshold_counters[idx] = 0
                        print(f"Frame {i}: Ear {idx+1} started rotating")
                elif state == "rotating":
                    if is_moving:
                        below_threshold_counters[idx] = 0
                    else:
                        below_threshold_counters[idx] += 1
                        if below_threshold_counters[idx] >= stop_grace_period:
                            rotation_states[idx] = "idle"
                            print(f"Frame {i}: Ear {idx+1} stopped rotating")

            # Draw ear box
            color = (0, 255, 0)
            if i < stabilization_frames:
                color = (128, 128, 128)
            elif rotation_states[idx] == "rotating":
                color = (0, 0, 255)
                cv2.putText(frame, f"Ear {idx+1} rotating", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.rectangle(frame, top_left, bottom_right, color, thickness=box_thickness)

            # Draw ear trail
            ear_trails[idx].append((cx, cy))
            if len(ear_trails[idx]) > max_trail_length:
                ear_trails[idx].pop(0)
            for j in range(1, len(ear_trails[idx])):
                cv2.line(frame, ear_trails[idx][j - 1], ear_trails[idx][j], (255, 255, 0), 1)



        if i < stabilization_frames:
            cv2.putText(frame, f"Stabilizing... ({stabilization_frames - i})", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        ear_points = next_ear_points
        head_point = next_head_point
        prev_gray = gray

        cv2.imshow("Ear Rotation Tracker", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



def main(video_path, show=True):
    print("Extracting frames...")
    frames = extract_frames(video_path)
    print("Detecting ears in the first frame...")
    ear_positions, head_position = detect_head_and_ears(frames, show_initial_detections=True)
    print("Tracking and visualizing ear movement...")
    track_and_visualize(frames, ear_positions, head_position[0])
    print("Done.")

if __name__ == "__main__":
    main("video/S3_Video.mp4")
