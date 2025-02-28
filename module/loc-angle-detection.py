# %%
import json
from typing import Any, Protocol, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose

# %%
MODEL_COMPLEXITY = 0

pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY
)
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

# %%
from typing import Literal

type HeadLandmarks = dict[Literal["head", "left_ear", "right_ear"], Landmark]


class Landmark(Protocol):
    x: float
    y: float
    z: float
    # visibility: float


def get_landmarks(image: cv2.typing.MatLike) -> HeadLandmarks:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = cast(PoseResult, pose.process(image_rgb))

    pose_landmarks = pose_results.pose_landmarks.landmark
    assert pose_landmarks

    head = pose_landmarks[mp_pose.PoseLandmark.NOSE]
    left_ear = pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

    return {"head": head, "left_ear": left_ear, "right_ear": right_ear}


def draw_skeleton(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    copied_image = image.copy()
    results = cast(PoseResult, pose.process(image_rgb))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            copied_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
    return copied_image

# %%
from pathlib import Path

RESOURCE_DIR = Path("/home/projects/webrtc-cv/resources")
FRONT = RESOURCE_DIR.joinpath("front.jpg")
SIDE = RESOURCE_DIR.joinpath("side.jpg")

front_img = cv2.imread(FRONT.as_posix())
side_img = cv2.imread(SIDE.as_posix())

# %%
def show(*image: cv2.typing.MatLike):
    """Draw several images in a row in full size"""
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(16, 16))
    for i, img in enumerate(image):
        plt.subplot(1, len(image), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


show(front_img, side_img)
show(draw_skeleton(front_img), draw_skeleton(side_img))

# %%
from collections import namedtuple


def get_diffs(first: Landmark, second: Landmark) -> Landmark:
    # return namedtuple("Landmark", "x y z")(first.x - second.x, first.y - second.y, first.z - second.z)
    dx = second.x - first.x
    dy = first.y - second.y
    dz = second.z - first.z
    return namedtuple("Diffs", "x y z")(dx, dy, dz)


front_diffs = get_diffs(get_landmarks(front_img)["right_ear"], get_landmarks(front_img)["left_ear"])
side_diffs = get_diffs(get_landmarks(side_img)["right_ear"], get_landmarks(side_img)["left_ear"])
print(front_diffs, side_diffs, sep="\n")

# %%
import math

def print_atan2(y, x):
    print((math.degrees(math.atan2(y, x))) % 360)

print_atan2(front_diffs.z, front_diffs.x)
print_atan2(side_diffs.z, side_diffs.x)

# %%
print(np.arctan2(front_diffs.z, front_diffs.x))
print(np.arctan2(side_diffs.z, side_diffs.x))

# %%
def get_avatar_coordinates(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_diffs(first: Landmark, second: Landmark) -> Landmark:
        dx = second.x - first.x
        dy = first.y - second.y
        dz = second.z - first.z
        return namedtuple("Diffs", "x y z")(dx, dy, dz)

    data = {"body": {}}  # Оставляем только голову
    # Получаем 3D координаты головы
    head_landmarks = get_landmarks(image)
    diffs = get_diffs(head_landmarks["right_ear"], head_landmarks["left_ear"])
    head = head_landmarks["head"]
    image_width = image.shape[1]

    yaw = np.arctan2(diffs.z, diffs.x)
    # Передаём координаты и угол поворота
    data["body"]["Bip01_Head1"] = {
        "position": [head.x * image_width, head.y * image_width, head.z * image_width],
        "rotation": [0, yaw, 0],  # Оставляем только поворот вокруг Y
    }

    return json.dumps(data)

get_avatar_coordinates(front_img)


