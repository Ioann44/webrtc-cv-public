from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = "YOLOv10n_gestures.pt"
full_model_path = Path(__file__).with_name(MODEL_PATH)
model = YOLO(full_model_path)


def detect(image: cv2.typing.MatLike, threshold=0.5) -> list[tuple[int, int, int, int, str]]:
    results = model(image)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > threshold:
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{model.names[int(class_id)]} {conf:.2f}"
                detections.append((x1, y1, x2, y2, label))
    return detections


def draw(image: cv2.typing.MatLike, detections: list[tuple[int, int, int, int, str]]):
    for x1, y1, x2, y2, label in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_frame(frame: cv2.typing.MatLike):
    detections = detect(frame)
    draw(frame, detections)
    return frame
