import json
from collections import namedtuple
from typing import Any, Literal, Protocol, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose

MODEL_COMPLEXITY = 1

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


type HeadLandmarks = dict[Literal["head", "left_ear", "right_ear", "head_center"], Landmark]


class Landmark(Protocol):
    x: float
    y: float
    z: float


def draw_skeleton(image: cv2.typing.MatLike, draw_on_copy=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if draw_on_copy:
        image_rgb = image.copy()
    results = cast(PoseResult, pose.process(image_rgb))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
    return image_rgb


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


def get_landmarks(image: cv2.typing.MatLike, is_rgb=False) -> HeadLandmarks:
    if not is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = cast(PoseResult, pose.process(image))

    pose_landmarks = pose_results.pose_landmarks.landmark
    assert pose_landmarks

    head = pose_landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

    def get_avg(first: Landmark, second: Landmark) -> Landmark:
        return namedtuple("Averages", "x y z")(
            (first.x + second.x) / 2, (first.y + second.y) / 2, (first.z + second.z) / 2
        )

    head_center = get_avg(right_ear, left_ear)

    return {"head": head, "left_ear": left_ear, "right_ear": right_ear, "head_center": head_center}


def get_avatar_coordinates(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_diffs(first: Landmark, second: Landmark) -> Landmark:
        dx = second.x - first.x
        dy = first.y - second.y
        dz = second.z - first.z
        return namedtuple("Diffs", "x y z")(dx, dy, dz)

    data = {"body": {}}  # Оставляем только голову
    # Получаем 3D координаты головы
    head_landmarks = get_landmarks(image_rgb, True)
    diffs = get_diffs(head_landmarks["right_ear"], head_landmarks["left_ear"])
    head = head_landmarks["head"]
    image_width = image.shape[1]

    yaw = np.arctan2(diffs.z, diffs.x)
    roll = np.arctan2(diffs.y, diffs.x)

    avg_to_head_diff = get_diffs(head_landmarks["head_center"], head_landmarks["head"])
    pitch = np.arctan2(avg_to_head_diff.y, -avg_to_head_diff.z)

    # Передаём координаты и угол поворота
    data["body"]["Bip01_Head1"] = {
        "position": [head.x * image_width, head.y * image_width, head.z * image_width],
        "rotation": [pitch, yaw, roll],
    }

    return json.dumps(data)
