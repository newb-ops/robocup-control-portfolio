from __future__ import annotations
"""Project RealSense detections onto robot-relative floor coordinates.

Uses detector boxes and FramePacket metadata from BallFollower;
camera capture and object detection remain with the caller."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import normalize_class_name
from .localization import Observation
from .sources import CameraIntrinsics, FramePacket


BALL_RADIUS = 0.11

DETECTABLE_OBJECT_TYPES = {
    "corner_arc",
    "goal",
    "goal_bar",
    "goalpost",
    "line_l",
    "line_t",
    "line_x",
    "penalty_mark",
    "center_mark",
    "robot",
}

PF_TYPE_BY_OBJECT = {
    "line_l": "L",
    "line_t": "T",
    "line_x": "X",
    "goal": "goalpost",
    "goal_bar": "goalpost",
    "goalpost": "goalpost",
    "center_mark": "center_mark",
    "penalty_mark": "penalty_mark",
    "corner_arc": "corner_arc",
}

BALL_CLASS_ALIASES = {
    "ball",
    "soccer_ball",
    "soccerball",
    "football",
    "sports_ball",
    "sportsball",
    "orange_ball",
    "orangeball",
    "ballhz60",
    "ballhz6o",
}


@dataclass
class LandmarkDetection:
    obs_type: str
    class_name: str
    conf: float
    bbox: Tuple[float, float, float, float]
    u: float
    v: float
    depth_m: Optional[float] = None
    robot_xy: Optional[np.ndarray] = None
    method: str = "none"
    reject_reason: str = ""
    used_for_pf: bool = False
    pf_type: Optional[str] = None


@dataclass
class RealPositionSnap:
    pitch_down_deg: float = 0.0
    pitch_source: str = "manual"
    n_landmarks: int = 0
    n_used_for_pf: int = 0
    ball_xy: Optional[Tuple[float, float]] = None
    ball_method: str = "none"


def class_to_object_type(class_name: object) -> Optional[str]:
    raw = str(class_name).strip()
    norm = normalize_class_name(raw)
    direct = {
        "cornerarc": "corner_arc",
        "goal": "goal",
        "goalbar": "goal_bar",
        "bar": "goalpost",
        "crossbar": "goalpost",
        "goalpost": "goalpost",
        "goalposts": "goalpost",
        "post": "goalpost",
        "posts": "goalpost",
        "linel": "line_l",
        "linecorner": "line_l",
        "lline": "line_l",
        "lshape": "line_l",
        "l": "line_l",
        "twoangle": "line_l",
        "2angle": "line_l",
        "linet": "line_t",
        "tline": "line_t",
        "tshape": "line_t",
        "t": "line_t",
        "threeangle": "line_t",
        "3angle": "line_t",
        "linex": "line_x",
        "xline": "line_x",
        "cross": "line_x",
        "intersection": "line_x",
        "x": "line_x",
        "centermark": "center_mark",
        "center": "center_mark",
        "penaltymark": "penalty_mark",
        "penalty": "penalty_mark",
        "robot": "robot",
    }
    low = raw.lower()
    if low in DETECTABLE_OBJECT_TYPES:
        return low
    return direct.get(norm)


def object_type_to_pf_type(object_type: Optional[str]) -> Optional[str]:
    if object_type is None:
        return None
    return PF_TYPE_BY_OBJECT.get(object_type)


def is_ball_class(class_name: object) -> bool:
    return normalize_class_name(class_name) in {normalize_class_name(x) for x in BALL_CLASS_ALIASES}


def robust_depth_at(
    depth_m: Optional[np.ndarray],
    u: float,
    v: float,
    window: int,
    min_depth: float,
    max_depth: float,
) -> Optional[float]:
    if depth_m is None:
        return None
    h, w = depth_m.shape[:2]
    cx = int(np.clip(round(float(u)), 0, w - 1))
    cy = int(np.clip(round(float(v)), 0, h - 1))
    r = max(1, int(window))
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch)]
    valid = valid[(valid >= float(min_depth)) & (valid <= float(max_depth))]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class RealCameraGeometry:
    """Convert D435/D435i camera points into robot forward/left floor coords."""

    def __init__(
        self,
        intr: CameraIntrinsics,
        camera_height: float,
        pitch_down_deg: float,
        camera_x: float,
        camera_y: float,
    ):
        self.intr = intr
        self.t_robot = np.array([float(camera_x), float(camera_y), float(camera_height)], dtype=float)
        self.pitch_down_deg = float(pitch_down_deg)
        self.r_robot_cam = np.eye(3, dtype=float)
        self.set_pitch_down_deg(self.pitch_down_deg)

    def set_intrinsics(self, intr: CameraIntrinsics) -> None:
        self.intr = intr

    def set_pitch_down_deg(self, pitch_down_deg: float) -> None:
        self.pitch_down_deg = float(pitch_down_deg)
        pitch = math.radians(self.pitch_down_deg)
        r0 = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        ry = np.array(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ],
            dtype=float,
        )
        self.r_robot_cam = ry @ r0

    def deproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        z = float(depth_m)
        x = (float(u) - self.intr.ppx) * z / max(self.intr.fx, 1e-8)
        y = (float(v) - self.intr.ppy) * z / max(self.intr.fy, 1e-8)
        return np.array([x, y, z], dtype=float)

    def camera_ray(self, u: float, v: float) -> np.ndarray:
        ray = np.array(
            [
                (float(u) - self.intr.ppx) / max(self.intr.fx, 1e-8),
                (float(v) - self.intr.ppy) / max(self.intr.fy, 1e-8),
                1.0,
            ],
            dtype=float,
        )
        return ray / max(float(np.linalg.norm(ray)), 1e-8)

    def camera_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        return self.t_robot + self.r_robot_cam @ np.asarray(point_cam, dtype=float)

    def robot_to_camera(self, point_robot: np.ndarray) -> np.ndarray:
        return self.r_robot_cam.T @ (np.asarray(point_robot, dtype=float) - self.t_robot)

    def pixel_to_robot_floor(
        self,
        u: float,
        v: float,
        depth_m: Optional[float],
        use_depth: bool,
        allow_floor_ray: bool,
    ) -> Tuple[Optional[np.ndarray], str]:
        if use_depth and depth_m is not None:
            point_robot = self.camera_to_robot(self.deproject(u, v, depth_m))
            xy = np.array([float(point_robot[0]), float(point_robot[1])], dtype=float)
            if np.isfinite(xy).all():
                return xy, "depth"

        if allow_floor_ray:
            ray_robot = self.r_robot_cam @ self.camera_ray(u, v)
            if ray_robot[2] < -1e-6:
                scale = -self.t_robot[2] / ray_robot[2]
                if scale > 0:
                    hit = self.t_robot + scale * ray_robot
                    xy = np.array([float(hit[0]), float(hit[1])], dtype=float)
                    if np.isfinite(xy).all():
                        return xy, "floor_ray"
        return None, "none"


def _box_values(box) -> Tuple[int, str, float, Tuple[float, float, float, float]]:
    cid = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)
    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)
    xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, "cpu") else np.asarray(box.xyxy)[0]
    cname = str(getattr(box, "cls_name", cid))
    return cid, cname, conf, tuple(float(v) for v in xyxy)


class RealPositionEstimator:
    def __init__(self, cfg, names: Optional[Dict[int, str]] = None):
        self.cfg = cfg
        self.names = names or {}
        self.geometry: Optional[RealCameraGeometry] = None
        self.last_landmarks: List[LandmarkDetection] = []
        self.last_snap = RealPositionSnap(pitch_down_deg=float(getattr(cfg, "camera_pitch_down_deg", 20.0)))

    def update_packet(self, pkt: FramePacket) -> None:
        intr = pkt.intrinsics
        if intr is None:
            intr = CameraIntrinsics(
                fx=float(getattr(self.cfg, "camera_focal_length", 600.0)),
                fy=float(getattr(self.cfg, "camera_focal_length", 600.0)),
                ppx=float(getattr(self.cfg, "img_width", 640)) / 2.0,
                ppy=float(getattr(self.cfg, "img_height", 480)) / 2.0,
                width=int(getattr(self.cfg, "img_width", 640)),
                height=int(getattr(self.cfg, "img_height", 480)),
            )
        pitch = float(pkt.imu_pitch_deg if pkt.imu_pitch_deg is not None
                      else getattr(self.cfg, "camera_pitch_down_deg", 20.0))
        if self.geometry is None:
            self.geometry = RealCameraGeometry(
                intr,
                camera_height=getattr(self.cfg, "camera_height", 1.23),
                pitch_down_deg=pitch,
                camera_x=getattr(self.cfg, "camera_x", 0.0),
                camera_y=getattr(self.cfg, "camera_y", 0.0),
            )
        else:
            self.geometry.set_intrinsics(intr)
            self.geometry.set_pitch_down_deg(pitch)
        self.last_snap.pitch_down_deg = pitch
        self.last_snap.pitch_source = pkt.pitch_source

    def landmark_observations(self, boxes: Sequence, depth: Optional[np.ndarray]) -> List[Observation]:
        if self.geometry is None:
            self.last_landmarks = []
            return []
        # PF에 사용할 랜드마크 타입은 설정에서 제한한다. 오검출이 잦은 타입은 제외한다.
        allow = getattr(self.cfg, "loc_landmark_classes", ("L", "T", "X", "corner_arc", "penalty_mark", "goalpost"))
        allow = set(allow) if allow else None   # None/빈값 = 전부 허용
        landmarks: List[LandmarkDetection] = []
        obs: List[Observation] = []
        for box in boxes:
            cid, cname, conf, xyxy = _box_values(box)
            if cid in self.names:
                cname = str(self.names[cid])
            object_type = class_to_object_type(cname)
            pf_type = object_type_to_pf_type(object_type)
            if object_type is None:
                continue
            x1, y1, x2, y2 = xyxy
            if object_type in ("goal", "goal_bar", "goalpost", "robot"):
                u = 0.5 * (x1 + x2)
                v = y2
            else:
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
            det = LandmarkDetection(object_type, str(cname), conf, xyxy, u, v, pf_type=pf_type)
            det.depth_m = robust_depth_at(
                depth, u, v, getattr(self.cfg, "depth_sample_radius", 5),
                getattr(self.cfg, "real_depth_min", 0.08),
                getattr(self.cfg, "real_depth_max", 10.0),
            )
            det.robot_xy, det.method = self.geometry.pixel_to_robot_floor(
                u, v, det.depth_m, use_depth=det.depth_m is not None,
                allow_floor_ray=getattr(self.cfg, "real_floor_ray_fallback", True),
            )
            if det.robot_xy is None:
                det.reject_reason = "no_3d"
            elif pf_type is None:
                det.reject_reason = "non_pf_type"
            elif allow is not None and pf_type not in allow:
                det.reject_reason = "not_whitelisted"   # center/bar 등 제외
            else:
                det.used_for_pf = True
                obs.append(Observation(pf_type, float(det.robot_xy[0]), float(det.robot_xy[1])))
            landmarks.append(det)
        self.last_landmarks = landmarks
        self.last_snap.n_landmarks = len(landmarks)
        self.last_snap.n_used_for_pf = len(obs)
        return obs[:6]

    def ball_robot_xy(self, boxes: Sequence, depth: Optional[np.ndarray], ball_ids) -> Tuple[Optional[Tuple[float, float]], str]:
        if self.geometry is None:
            return None, "none"
        best = None
        ball_id_set = {int(x) for x in (ball_ids or [])}
        for box in boxes:
            cid, cname, conf, xyxy = _box_values(box)
            if cid in self.names:
                cname = str(self.names[cid])
            if not (cid in ball_id_set or is_ball_class(cname)):
                continue
            if best is None or conf > best[0]:
                best = (conf, xyxy)
        if best is None:
            self.last_snap.ball_xy = None
            self.last_snap.ball_method = "none"
            return None, "none"
        _conf, (x1, y1, x2, y2) = best
        u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        depth_m = robust_depth_at(
            depth, u, v, getattr(self.cfg, "depth_sample_radius", 5),
            getattr(self.cfg, "real_depth_min", 0.08),
            getattr(self.cfg, "real_depth_max", 10.0),
        )
        if depth_m is not None:
            surface = self.geometry.deproject(u, v, depth_m)
            ray = self.geometry.camera_ray(u, v)
            center_robot = self.geometry.camera_to_robot(surface + ray * BALL_RADIUS)
            xy = (float(center_robot[0]), float(center_robot[1]))
            self.last_snap.ball_xy = xy
            self.last_snap.ball_method = "depth"
            return xy, "depth"
        if getattr(self.cfg, "real_ball_bbox_fallback", True):
            radius_px = max(1.0, 0.25 * ((x2 - x1) + (y2 - y1)))
            z = self.geometry.intr.fx * BALL_RADIUS / radius_px
            center_robot = self.geometry.camera_to_robot(self.geometry.deproject(u, v, z))
            xy = (float(center_robot[0]), float(center_robot[1]))
            self.last_snap.ball_xy = xy
            self.last_snap.ball_method = "bbox"
            return xy, "bbox"
        self.last_snap.ball_xy = None
        self.last_snap.ball_method = "none"
        return None, "none"
