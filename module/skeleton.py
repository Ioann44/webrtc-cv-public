import json
from typing import Any, Protocol, cast

import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose

MODEL_COMPLEXITY = 0

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY)
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY, max_num_hands=2
)


class PoseResult(Protocol):
    pose_landmarks: Any
    pose_world_landmarks: Any
    segmentation_mask: Any


class HandsResult(Protocol):
    multi_hand_landmarks: Any
    multi_hand_world_landmarks: Any
    multi_handedness: Any


def draw_skeleton(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = cast(PoseResult, pose.process(image_rgb))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )


def draw_hands(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = cast(HandsResult, hands.process(image_rgb))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            )


def get_avatar_coordinates(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Получаем результаты для тела
    pose_results = cast(PoseResult, pose.process(image_rgb))
    hands_results = cast(HandsResult, hands.process(image_rgb))

    data = {}
    torso_center = {"x": 0.0, "y": 0.0, "z": 0.0}

    if pose_results.pose_world_landmarks:
        # Получаем 3D координаты точек тела
        pose_landmarks = pose_results.pose_world_landmarks.landmark
        data["body"] = [{"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in pose_landmarks]

        # Определяем центр туловища (например, средняя точка между плечами)
        left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        torso_center = {
            "x": (left_shoulder.x + right_shoulder.x) / 2,
            "y": (left_shoulder.y + right_shoulder.y) / 2,
            "z": (left_shoulder.z + right_shoulder.z) / 2,
        }
        data["torso_center"] = torso_center

    if hands_results.multi_hand_world_landmarks:
        # Получаем 3D координаты точек рук
        data["hands"] = []
        for hand_landmarks in hands_results.multi_hand_world_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(
                    {
                        "x": landmark.x - torso_center["x"],
                        "y": landmark.y - torso_center["y"],
                        "z": landmark.z - torso_center["z"],
                    }
                )
            data["hands"].append(hand_data)

    return json.dumps(data)
