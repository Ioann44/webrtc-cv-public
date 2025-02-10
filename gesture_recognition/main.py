from ultralytics import YOLO
import cv2

# Укажите путь к модели YOLO в формате .pt
MODEL_PATH = "YOLOv10x_gestures.pt"

# Загрузка модели YOLO
model = YOLO(MODEL_PATH)
model.to("cuda")  # Явно переносим модель на GPU

# Настройки камеры
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.flip(frame, 1, frame)  # Зеркальное отражение кадра

    # Выполнение предсказания (обработка на GPU)
    results = model(frame, device="cuda")

    # Обработка результатов
    for result in results:
        boxes = result.boxes.xyxy  # Оставляем данные на GPU
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  # Порог уверенности
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{model.names[int(class_id)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение видео
    cv2.imshow("YOLO Gesture Recognition", frame)

    # Нажмите 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
