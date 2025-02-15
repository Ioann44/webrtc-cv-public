from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = "YOLOv10n_gestures.pt"
full_model_path = Path(__file__).with_name(MODEL_PATH)
model = YOLO(full_model_path)


def detect(image: cv2.typing.MatLike, threshold=0.25) -> list[tuple[int, int, int, int, str]]:
    results = model(image, conf=threshold, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"{model.names[int(class_id)]} {conf:.2f}"
            detections.append((x1, y1, x2, y2, label))
    return detections


def draw_detections(image: cv2.typing.MatLike, detections: list[tuple[int, int, int, int, str]]):
    for x1, y1, x2, y2, label in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Словарь для хранения траекторий объектов
trajectories = {}


def track_objects(image: cv2.typing.MatLike, threshold=0.25) -> dict:
    results = model.track(image, persist=True, conf=threshold, verbose=False)
    tracks = {}

    for track in results:
        boxes = track.boxes.xyxy  # type: ignore
        track_ids = track.boxes.id  # type: ignore
        if track_ids is None:
            track_ids = []
        confidences = track.boxes.conf  # type: ignore
        class_ids = track.boxes.cls  # type: ignore

        for box, track_id, conf, class_id in zip(boxes, track_ids, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            track_id = int(track_id)
            label = f"{model.names[int(class_id)]} {conf:.2f}"

            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))

            tracks[track_id] = (x1, y1, x2, y2, label)

    return tracks


def draw_tracks(image: cv2.typing.MatLike, tracks: dict):
    for track_id, (x1, y1, x2, y2, label) in tracks.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"ID {track_id}: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if track_id in trajectories:
            for i in range(1, len(trajectories[track_id])):
                if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                    continue
                cv2.line(image, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 0, 255), 2)
