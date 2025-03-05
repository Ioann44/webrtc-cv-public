from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence, cast

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark

MODEL_COMPLEXITY = 1

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY)
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=MODEL_COMPLEXITY, max_num_hands=2
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


def get_avg(
    *landmarks: Landmark | PoseLandmark | HandLandmark,
    landmarks_container: PoseLandmarks | HandLandmarks | None = None,
) -> Landmark:
    if landmarks_container is not None:
        assert isinstance(landmarks[0], PoseLandmark | HandLandmark)
        landmarks = tuple(landmarks_container[landmark] for landmark in landmarks)
    assert landmarks
    x = y = z = 0
    for landmark in landmarks:
        x += landmark.x
        y += landmark.y
        z += landmark.z
    return namedtuple("Averages", "x y z")(x / len(landmarks), y / len(landmarks), z / len(landmarks))


def get_pos_list(landmark: Landmark) -> list[float]:
    return [landmark.x, landmark.y, landmark.z]


def get_diffs(first: Landmark, second: Landmark) -> Landmark:
    dx = second.x - first.x
    dy = first.y - second.y
    dz = second.z - first.z
    return namedtuple("Diffs", "x y z")(dx, dy, dz)


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
    # pose_landmarks = pose_results.pose_landmarks.landmark
    pose_landmarks = pose_results.pose_world_landmarks.landmark

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

    return data  # type: ignore
