import cv2
import numpy as np
from  YOLO_inference import detect_horse_features, model

def extract_frames(video_path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    return frames

def detect_head_and_ears(frame):
    # Placeholder for real detection
    height, width, _ = frame.shape
    ear1 = (100, 155, 25, 30)
    ear2 = (130, 165, 20, 38)
    return [ear1, ear2]

def get_points_from_bbox(bbox):
    x, y, w, h = bbox
    return np.array([[x + w//2, y + h//2]], dtype=np.float32)

def track_and_visualize(frames, initial_positions, movement_threshold=0.3, box_thickness:int = 1):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points = np.array([get_points_from_bbox(pos) for pos in initial_positions]).reshape(-1, 1, 2)
    box_sizes = [(w, h) for (_, _, w, h) in initial_positions]

    for i in range(1, len(frames)):
        frame = frames[i].copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)

        for idx, (old, new) in enumerate(zip(points, next_points)):
            dx, dy = new[0] - old[0]
            movement = np.sqrt(dx**2 + dy**2)

            # Get current point and original box size
            cx, cy = int(new[0][0]), int(new[0][1])
            box_w, box_h = box_sizes[idx]
            top_left = (cx - box_w // 2, cy - box_h // 2)
            bottom_right = (cx + box_w // 2, cy + box_h // 2)

            movement_threshold_pixels = box_h + box_w / 2 * movement_threshold
            color = (0, 255, 0) if movement < movement_threshold_pixels else (0, 0, 255)
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=box_thickness)

            

            if movement >= movement_threshold_pixels:
                text = f"Ear {idx+1} rotation detected!"
                cv2.putText(frame, text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Frame {i}: {text} (Movement: {movement:.2f})")

        points = next_points
        prev_gray = gray

        # Show the frame
        cv2.imshow("Ear Tracking", frame)
        if cv2.waitKey(75) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main(video_path):
    print("Extracting frames...")
    frames = extract_frames(video_path)
    print("Detecting ears in the first frame...")
    _, detected_objects = detect_horse_features(model, frames[0], show=True)
    ear_detections = [obj for obj in detected_objects if obj["label"] == "mouth"]
        
    print(detected_objects)
    print(ear_detections)

    exit()

    #initial_positions = detect_head_and_ears(frames[0])
    print("Tracking and visualizing ear movement...")
    track_and_visualize(frames, initial_positions)
    print("Done.")

if __name__ == "__main__":
    main("video/horse.mp4")