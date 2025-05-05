import torch
import torchvision
from ultralytics import YOLO

if __name__ == "__main__":
    print("Torchvision test:", torchvision.ops.nms)  # Should not error
    print("Torch cuda us available:", torch.cuda.is_available())        # should be True
    print("Torch cuda device name", torch.cuda.get_device_name(0))    # “NVIDIA GeForce RTX 2080 SUPER”
    print("Torch cuda device VRAM", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    # Load a pre-trained YOLOv8 model (nano, small, medium, etc.)
    #model = YOLO('yolov8n.pt') #  YOLO v8 Nano (3.2m)
    #model = YOLO('yolov8s.pt') #  YOLO v8 Small (11.2m)
    model = YOLO('yolov8m.pt')  #  YOLO v8 Medium (25.9m)

    model.train(
        data='training/horse_data.yaml',
        epochs=100,
        patience=10,
        imgsz=640,
        name='horse_features',
        batch=16,
        workers=8,
        device='cuda',
        seed=0,
        deterministic=True
    )