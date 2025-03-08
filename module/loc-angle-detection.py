# %%
from dataclasses import dataclass
import json
import math
from collections import namedtuple
from collections.abc import Callable, Sequence
from itertools import islice
from typing import Any, Literal, Protocol, cast

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.pose import PoseLandmark

MODEL_COMPLEXITY = 1

pose = mp_pose.Pose(True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY)
hands = mp_hands.Hands(
    True, min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY, max_num_hands=2
)


class Landmark(Protocol):
    x: float
    y: float
    z: float


class PoseLandmarks(Protocol):
    __getitem__: Callable[[PoseLandmark], Landmark]


class HandLandmarks(Protocol):
    __getitem__: Callable[[HandLandmark], Landmark]


class LandmarksContainer[T](Protocol):
    landmark: T
    __iter__: Callable[[], Any]


class PoseResult(Protocol):
    pose_landmarks: LandmarksContainer[PoseLandmarks]
    pose_world_landmarks: LandmarksContainer[PoseLandmarks]
    segmentation_mask: Any


class HandsResult(Protocol):
    multi_hand_landmarks: LandmarksContainer[PoseLandmarks]
    multi_hand_world_landmarks: LandmarksContainer[PoseLandmarks]
    multi_handedness: Any


@dataclass(slots=True)
class Point:
    x: float
    y: float
    z: float

# %%
def draw_skeleton(image: cv2.typing.MatLike, draw_on_copy=True, is_bgr=False):
    if is_bgr:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
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


def get_avg(*landmarks: Landmark) -> Landmark:
    x = y = z = 0
    for landmark in landmarks:
        x += landmark.x
        y += landmark.y
        z += landmark.z
    return Point(x / len(landmarks), y / len(landmarks), z / len(landmarks))


def get_pos_list(landmark: Landmark) -> list[float]:
    return [landmark.x, landmark.y, landmark.z]


def get_diffs(first: Landmark, second: Landmark) -> Landmark:
    dx = second.x - first.x
    dy = first.y - second.y
    dz = second.z - first.z
    return Point(dx, dy, dz)

# %%
def get_landmarks(image: cv2.typing.MatLike):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = cast(PoseResult, pose.process(image_rgb))
    return results.pose_landmarks.landmark

# %%
from pathlib import Path

RESOURCE_DIR = Path("/home/projects/webrtc-cv/resources")
FRONT = RESOURCE_DIR.joinpath("front.jpg")
SIDE = RESOURCE_DIR.joinpath("side.jpg")
ROLL = RESOURCE_DIR.joinpath("roll.jpg")
PITCH = RESOURCE_DIR.joinpath("pitch.jpg")
POSED = RESOURCE_DIR.joinpath("posed.jpg")

front_img = cv2.imread(FRONT.as_posix())
side_img = cv2.imread(SIDE.as_posix())
roll_img = cv2.imread(ROLL.as_posix())
pitch_img = cv2.imread(PITCH.as_posix())
posed_img = cv2.imread(POSED.as_posix())

titled_images = {"front": front_img, "side": side_img, "roll": roll_img, "pitch": pitch_img, "posed": posed_img}
landmarks = {title: get_landmarks(img) for title, img in titled_images.items()}

# %%
def show(*image: cv2.typing.MatLike):
    """Draw several images in a row in full size"""
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(16, 16))
    for i, img in enumerate(image):
        plt.subplot(1, len(image), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


show(*islice(titled_images.values(), 4))
show(*islice((draw_skeleton(img) for img in titled_images.values()), 4))

# %%
show(posed_img, draw_skeleton(posed_img))

# %% [markdown]
# <details>
# <summary>Deprecated pose landmarks</summary>
# ```py
# def get_diffs(first: Landmark, second: Landmark) -> Landmark:
#     # return namedtuple("Landmark", "x y z")(first.x - second.x, first.y - second.y, first.z - second.z)
#     dx = second.x - first.x
#     dy = first.y - second.y
#     dz = second.z - first.z
#     return namedtuple("Diffs", "x y z")(dx, dy, dz)
# 
# 
# def get_avg(first: Landmark, second: Landmark) -> Landmark:
#     return namedtuple("Averages", "x y z")((first.x + second.x) / 2, (first.y + second.y) / 2, (first.z + second.z) / 2)
# 
# 
# def get_head_info(pose_results: PoseResult, image_shape: Sequence[int]):
#     pose_landmarks = pose_results.pose_landmarks.landmark
#     assert pose_landmarks
# 
#     head = pose_landmarks[mp_pose.PoseLandmark.NOSE]
#     left_ear = pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
#     right_ear = pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
# 
#     head_center = get_avg(right_ear, left_ear)
#     head_landmarks = {"head": head, "left_ear": left_ear, "right_ear": right_ear, "head_center": head_center}
# 
#     diffs = get_diffs(head_landmarks["right_ear"], head_landmarks["left_ear"])
#     head = head_landmarks["head"]
#     image_width = image_shape[1]
# 
#     yaw = np.arctan2(diffs.z, diffs.x)
#     roll = np.arctan2(diffs.y, diffs.x)
# 
#     avg_to_head_diff = get_diffs(head_landmarks["head_center"], head_landmarks["head"])
#     pitch = np.arctan2(avg_to_head_diff.y, -avg_to_head_diff.z)
# 
#     # Передаём координаты и угол поворота
#     return {
#         "position": [head.x * image_width, head.y * image_width, head.z * image_width],
#         "rotation": [pitch, yaw, roll],
#     }
# 
# 
# landmarks = {title: get_landmarks(img) for title, img in titled_images.items()}
# 
# diffs = {title: get_diffs(landmark["right_ear"], landmark["left_ear"]) for title, landmark in landmarks.items()}
# print("Ears diffs", *diffs.items(), sep="\n")
# avg_ears = {title: get_avg(landmark["right_ear"], landmark["left_ear"]) for title, landmark in landmarks.items()}
# print("Ears avgs", *avg_ears.items(), sep="\n")
# avg_noses_diffs = {title: get_diffs(avg_ears[title], landmarks[title]["head"]) for title in landmarks}
# print("Center-To-Nose diffs", *avg_noses_diffs.items(), sep="\n")
# 
# def print_atan2(y, x):
#     print((math.degrees(math.atan2(y, x))) % 360)
# 
# print("YAW")
# for title, diff in diffs.items():
#     print(title, end="\t")
#     print_atan2(diff.z, diff.x)
# print("ROLL")
# for title, diff in diffs.items():
#     print(title, end="\t")
#     print_atan2(diff.y, diff.x)
# print("PITCH")
# for title, diff in avg_noses_diffs.items():
#     print(title, end="\t")
#     print_atan2(diff.y, -diff.z)
# ```
# </details>

# %%
from collections.abc import Iterable


def show(*image: cv2.typing.MatLike, annotations: Iterable[tuple[str, tuple[float, float, float]]] | None = None):
    """Draw several images in a row in full size with optional annotations."""
    plt.figure(figsize=(16, 16))

    for i, img in enumerate(image):
        plt.subplot(1, len(image), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if annotations and i + 1 == len(image):
            for title, (x, y, _) in annotations:
                plt.text(x, y, title, rotation=30, color="green", fontsize=10)
                plt.plot(x, y, "ro")  # Рисуем точку

    plt.show()

# show(posed_img, annotations=[("abc", (100,100,100))])

# %%
def get_head_info(pose_landmarks: PoseLandmarks):
    head = pose_landmarks[PoseLandmark.NOSE]
    left_ear = pose_landmarks[PoseLandmark.LEFT_EAR]
    right_ear = pose_landmarks[PoseLandmark.RIGHT_EAR]

    diffs = get_diffs(right_ear, left_ear)
    yaw = np.arctan2(diffs.z, diffs.x)
    roll = np.arctan2(diffs.y, diffs.x)

    head_center = get_avg(right_ear, left_ear)
    avg_to_head_diff = get_diffs(head_center, head)
    pitch = np.arctan2(avg_to_head_diff.y, -avg_to_head_diff.z)

    # Передаём координаты и угол поворота
    return {
        "position": get_pos_list(
            get_avg(
                pose_landmarks[PoseLandmark.LEFT_SHOULDER],
                pose_landmarks[PoseLandmark.RIGHT_SHOULDER],
                Point(head_center.x, head_center.y * (-0.5), head_center.z),
            )
        ),
        "rotation": [yaw, roll, pitch],
    }


type NodeDescription = dict[str, dict[Literal["position", "rotation"], Sequence[float]]]
type ResultData = dict[Literal["body"], NodeDescription]


def get_avatar_coordinates(image: cv2.typing.MatLike) -> ResultData:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = cast(PoseResult, pose.process(image_rgb))
    pose_landmarks = pose_results.pose_landmarks.landmark
    # pose_landmarks = pose_results.pose_world_landmarks.landmark

    data = {"body": {}}
    head_landmarks = get_head_info(pose_landmarks)

    data["body"]["Bip01_Head1"] = head_landmarks
    for model_bone, mp_landmark in (
        # ("Bip01_Spine", PoseLandmark.LEFT_HIP),
        ("Bip01_L_UpperArm", PoseLandmark.LEFT_SHOULDER),
        ("Bip01_R_UpperArm", PoseLandmark.RIGHT_SHOULDER),
        ("Bip01_L_Forearm", PoseLandmark.LEFT_ELBOW),
        ("Bip01_R_Forearm", PoseLandmark.RIGHT_ELBOW),
        ("Bip01_L_Thigh", PoseLandmark.LEFT_HIP),
        ("Bip01_R_Thigh", PoseLandmark.RIGHT_HIP),
        ("Bip01_L_Calf", PoseLandmark.LEFT_KNEE),
        ("Bip01_R_Calf", PoseLandmark.RIGHT_KNEE),
        ("Bip01_L_Foot", PoseLandmark.LEFT_ANKLE),
        ("Bip01_R_Foot", PoseLandmark.RIGHT_ANKLE),
        ("Bip01_L_Hand", PoseLandmark.LEFT_WRIST),
        ("Bip01_R_Hand", PoseLandmark.RIGHT_WRIST),
    ):
        data["body"][model_bone] = {"position": get_pos_list(pose_landmarks[mp_landmark])}
    data["body"]["Bip01_Spine4"] = {
        "position": get_pos_list(
            get_avg(pose_landmarks[PoseLandmark.LEFT_SHOULDER], pose_landmarks[PoseLandmark.RIGHT_SHOULDER])
        )
    }
    data["body"]["Bip01_Spine2"] = {
        "position": get_pos_list(
            get_avg(
                pose_landmarks[PoseLandmark.LEFT_HIP],
                pose_landmarks[PoseLandmark.RIGHT_HIP],
                pose_landmarks[PoseLandmark.LEFT_SHOULDER],
                pose_landmarks[PoseLandmark.RIGHT_SHOULDER],
            )
        )
    }

    shoulders_diff = get_diffs(pose_landmarks[PoseLandmark.RIGHT_SHOULDER], pose_landmarks[PoseLandmark.LEFT_SHOULDER])
    spine_yaw = np.arctan2(shoulders_diff.z, shoulders_diff.x)
    top = get_avg(pose_landmarks[PoseLandmark.LEFT_SHOULDER], pose_landmarks[PoseLandmark.RIGHT_SHOULDER])
    bottom = get_avg(pose_landmarks[PoseLandmark.LEFT_HIP], pose_landmarks[PoseLandmark.RIGHT_HIP])
    up_down_diff = get_diffs(top, bottom)
    spine_pitch = np.arctan2(up_down_diff.z, -up_down_diff.y)
    spine_roll = np.arctan2(up_down_diff.x, -up_down_diff.y)
    #  yaw, roll, pitch
    data["body"]["Bip01_Spine2"]["rotation"] = [spine_yaw, spine_roll, spine_pitch]

    hips_diff = get_diffs(pose_landmarks[PoseLandmark.RIGHT_HIP], pose_landmarks[PoseLandmark.LEFT_HIP])
    hip_yaw = np.arctan2(hips_diff.z, hips_diff.x)
    hip_roll = np.arctan2(hips_diff.y, hips_diff.x)
    r_hip_knee_diff = get_diffs(pose_landmarks[PoseLandmark.RIGHT_HIP], pose_landmarks[PoseLandmark.RIGHT_KNEE])
    r_hip_pitch = np.arctan2(r_hip_knee_diff.z, -r_hip_knee_diff.y)
    l_hip_knee_diff = get_diffs(pose_landmarks[PoseLandmark.LEFT_HIP], pose_landmarks[PoseLandmark.LEFT_KNEE])
    l_hip_pitch = np.arctan2(l_hip_knee_diff.z, -l_hip_knee_diff.y)
    hip_pitch = (r_hip_pitch + l_hip_pitch) / 2
    #  yaw, roll, pitch
    data["body"]["Bip01_Pelvis"] = {"rotation": [hip_yaw, hip_roll, hip_pitch]}

    return data  # type: ignore


def multiply(array: Sequence[float], mul: float):
    return [i * mul for i in array]


data = get_avatar_coordinates(posed_img)["body"]
print("yaw roll pitch")
print(*map(lambda x: x * 57, data["Bip01_Head1"]["rotation"]))
print(*map(lambda x: x * 57, data["Bip01_Spine2"]["rotation"]))
print(*map(lambda x: x * 57, data["Bip01_Pelvis"]["rotation"]))

annotaions = (
    (title, ((pos := landmark["position"])[0] * posed_img.shape[1], pos[1] * posed_img.shape[0], 0))
    for title, landmark in data.items()
)
# show(draw_skeleton(posed_img), posed_img, annotations=annotaions)


