import cv2
from collections import Counter
from ultralytics import YOLO

# ENV YOLO_CONFIG_DIR 

# Load your best model
model = YOLO('runs\\detect\\horse_features\\weights\\best.pt')

# Define monitor size
max_width = 1600
max_height = 800

for i in range(1, 7):

    file_path = f'training\\test\\horse_test_00{i}.jpg'

    # Run model on image
    results = model(file_path, imgsz=640, conf=0.56)

    # Load image to draw bbox
    img = cv2.imread(file_path)

    # Counter to track detected objects
    detected_objects = Counter()

    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            label = model.names[int(cls_id)]
            if label in ["ear", "eye", "nostril", "head", "mouth"]:
                detected_objects[label] += 1  # Count the label
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                text = f'{label} {conf:.2f}'
                cv2.putText(img, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Print the detection counts
    print(f'Detections in {file_path}:')
    for label, count in detected_objects.items():
        print(f'  {label}: {count}')
    print('-' * 30)

    # Resize if needed
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        new_w = int(width * scale)
        new_h = int(height * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Show image
    cv2.imshow(f'Detections on image {file_path}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
