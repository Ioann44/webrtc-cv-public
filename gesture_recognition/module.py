from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = "YOLOv10x_gestures.pt"
full_model_path = Path(__file__).with_name(MODEL_PATH)
model = YOLO(full_model_path)


def process_frame(frame: cv2.typing.MatLike):
    cv2.flip(frame, 1, frame)
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{model.names[int(class_id)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
