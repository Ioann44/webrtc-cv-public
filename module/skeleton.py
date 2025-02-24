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

    # Получаем результаты для тела и рук
    pose_results = cast(PoseResult, pose.process(image_rgb))
    hands_results = cast(HandsResult, hands.process(image_rgb))

    data = {"body": {}, "hands": []}  # <-- Теперь `hands` всегда массив
    pelvis_position = [0.0, 0.0, 0.0]  # Для нормализации координат

    pose_map = {
        "Bip01_Pelvis": mp_pose.PoseLandmark.LEFT_HIP,
        "Bip01_Spine": mp_pose.PoseLandmark.RIGHT_HIP,
        "Bip01_Spine1": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "Bip01_Spine2": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "Bip01_Neck1": mp_pose.PoseLandmark.NOSE,
        "Bip01_Head1": mp_pose.PoseLandmark.NOSE,
        "Bip01_L_UpperArm": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "Bip01_L_Forearm": mp_pose.PoseLandmark.LEFT_ELBOW,
        "Bip01_L_Hand": mp_pose.PoseLandmark.LEFT_WRIST,
        "Bip01_R_UpperArm": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "Bip01_R_Forearm": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "Bip01_R_Hand": mp_pose.PoseLandmark.RIGHT_WRIST,
    }

    hand_map = {
        "Bip01_L_Finger0": 4,
        "Bip01_L_Finger1": 8,
        "Bip01_L_Finger2": 12,
        "Bip01_L_Finger3": 16,
        "Bip01_L_Finger4": 20,
        "Bip01_R_Finger0": 4,
        "Bip01_R_Finger1": 8,
        "Bip01_R_Finger2": 12,
        "Bip01_R_Finger3": 16,
        "Bip01_R_Finger4": 20,
    }

    if pose_results.pose_world_landmarks:
        pose_landmarks = pose_results.pose_world_landmarks.landmark

        pelvis_landmark = pose_landmarks[pose_map["Bip01_Pelvis"]]
        pelvis_position = [pelvis_landmark.x, pelvis_landmark.y, pelvis_landmark.z]

        for bone, landmark_id in pose_map.items():
            landmark = pose_landmarks[landmark_id]
            data["body"][bone] = [
                landmark.x - pelvis_position[0],
                landmark.y - pelvis_position[1],
                landmark.z - pelvis_position[2],
            ]

    if hands_results.multi_hand_world_landmarks and hands_results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(hands_results.multi_hand_world_landmarks, hands_results.multi_handedness)):
            hand_side = "L" if handedness.classification[0].label == "Left" else "R"
            hand_data = {}

            for bone, landmark_id in hand_map.items():
                if bone.startswith(f"Bip01_{hand_side}_"):
                    landmark = hand_landmarks.landmark[landmark_id]
                    hand_data[bone] = [
                        landmark.x - pelvis_position[0],
                        landmark.y - pelvis_position[1],
                        landmark.z - pelvis_position[2],
                    ]

            data["hands"].append(hand_data)

    return json.dumps(data, indent=2)
