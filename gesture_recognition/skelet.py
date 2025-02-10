import cv2
import mediapipe as mp

# Инициализация mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Захват видео с веб-камеры
cap = cv2.VideoCapture(1)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # Преобразование изображения в формат RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Обработка изображения с помощью mediapipe
        results = pose.process(image)

        # Обратное преобразование изображения в BGR для отображения
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Отрисовка скелета
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

        # Отображение изображения
        cv2.imshow('Skeleton Detection', image)

        # Выход из программы при нажатии клавиши 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
