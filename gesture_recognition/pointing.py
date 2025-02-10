import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# Инициализация Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Размеры экрана
screen_width, screen_height = pyautogui.size()

# Переменные для калибровки
calibration_points = []  # Хранение пар: [(3D-координаты пальца, экранные координаты)]
calibration_complete = False
transform_matrix = None


def calibrate_camera(frame, tip_3d):
    """Проводит калибровку: собирает 3D-координаты и экранные точки"""
    global calibration_points, calibration_complete, transform_matrix

    # Список экранных точек для калибровки (углы и центр)
    screen_targets = [
        (screen_width // 2, screen_height // 2),  # Центр
        (0, 0),  # Левый верхний угол
        (screen_width - 1, 0),  # Правый верхний угол
        (0, screen_height - 1),  # Левый нижний угол
        (screen_width - 1, screen_height - 1),  # Правый нижний угол
    ]

    # Текущая экранная точка для калибровки
    current_target = len(calibration_points)

    # Инструкция на экране
    cv2.putText(
        frame,
        f"Calibrate {current_target + 1}/{len(screen_targets)}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame, f"Point at: {screen_targets[current_target]}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
    )

    # Если палец находится в кадре
    if tip_3d is not None:
        # Нажмите "C", чтобы сохранить текущую точку
        if cv2.waitKey(1) & 0xFF == ord("c"):
            calibration_points.append((tip_3d, screen_targets[current_target]))
            print(f"Point {current_target + 1} saved: {tip_3d} -> {screen_targets[current_target]}")

            # Проверяем, завершена ли калибровка
            if len(calibration_points) == len(screen_targets):
                calibration_complete = True
                # Вычисляем матрицу преобразования
                transform_matrix = compute_transformation_matrix(calibration_points)
                print("Completed!")


def compute_transformation_matrix(points):
    """Вычисляет матрицу преобразования (3D -> экран)"""
    finger_coords = np.array([point[0] for point in points])  # 3D-координаты пальца
    screen_coords = np.array([point[1] for point in points])  # Экранные координаты

    # Добавляем столбец единиц для линейного уравнения (A @ X = B)
    A = np.column_stack((finger_coords, np.ones(len(finger_coords))))
    B = screen_coords

    # Решаем уравнение Ax = B для нахождения матрицы трансформации
    transform_matrix, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Преобразуем в матрицу 4x2
    return transform_matrix.T


def apply_transformation(finger_3d, matrix):
    """Применяет матрицу преобразования к 3D-координатам"""
    # Добавляем единицу для совместимости с матрицей 4x2
    point = np.array([*finger_3d, 1.0])  # (x, y, z, 1)

    # Умножаем матрицу на вектор
    screen_coords = np.dot(matrix, point)
    return screen_coords[0], screen_coords[1]


# Захват видео
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Не удалось открыть камеру.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр.")
        break

    # Переворачиваем изображение
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    tip_3d = None  # 3D-координаты кончика указательного пальца

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Отрисовка скелета кисти
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Получение 3D-координат
            tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            base_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Сохраняем нормализованные 3D-координаты
            tip_3d = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])

            # Рисуем точки на видео
            tip_x, tip_y = int(tip_landmark.x * frame_width), int(tip_landmark.y * frame_height)
            base_x, base_y = int(base_landmark.x * frame_width), int(base_landmark.y * frame_height)
            cv2.circle(frame, (tip_x, tip_y), 10, (255, 0, 0), -1)  # Кончик пальца
            cv2.circle(frame, (base_x, base_y), 10, (0, 255, 0), -1)  # Основание пальца

    # Если калибровка не завершена, запускаем процесс калибровки
    if not calibration_complete:
        calibrate_camera(frame, tip_3d)
    else:
        # Преобразуем 3D-координаты пальца в экранные
        try:
            screen_coords = apply_transformation(tip_3d, transform_matrix)
            pyautogui.moveTo(screen_coords[0], screen_coords[1])
      # Отображаем видео
    cv2.imshow("Hand Tracking with Calibration", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()
