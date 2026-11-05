from __future__ import annotations
"""검출 박스에서 공을 추적하고 거리와 골 방향을 추정한다.

공 후보를 검증한 뒤 Kalman 필터와 깊이·면적 기반 거리 추정을 적용한다.
골 방향은 IMU 기준 각도로 GoalMemory에 저장한다."""

import math
import threading
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from .common import (BallCandidate, BallObservation, RobotConfig, TuningParams,
                     clamp, now_sec)


# 공 Kalman (adaptive Q + gating)
class BallKalmanFilter:
    def __init__(self):
        self.x = np.zeros(4)         # [px, py, vx, vy]
        self.P = np.eye(4) * 500.0
        self.initialized = False
        self.last_t = None
        self.R_px = np.diag([8.0, 8.0])
        self.miss = 0

    def _Q(self, dt):
        q = 50.0
        return np.diag([q * dt, q * dt, q, q])

    def predict(self, dt: float = None):
        if not self.initialized:
            return self.get_position()
        if dt is None:
            t = now_sec(); dt = (t - self.last_t) if self.last_t else 0.033; self.last_t = t
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.get_position()

    def update(self, z: np.ndarray, use_depth: bool = True) -> bool:
        if not self.initialized:
            self.x[:2] = z; self.initialized = True; self.last_t = now_sec(); return True
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_px
        try:
            maha = float(y.T @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            maha = 0.0
        if maha > 30.0:                # innovation gating
            self.miss += 1
            if self.miss > 8:          # 오래 reject → 강제 수용 (재포착)
                self.x[:2] = z; self.miss = 0; return True
            return False
        self.miss = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return True

    def get_position(self): return float(self.x[0]), float(self.x[1])
    def get_velocity(self): return float(self.x[2]), float(self.x[3])
    def get_predicted_position(self, dt): return float(self.x[0] + self.x[2] * dt), float(self.x[1] + self.x[3] * dt)
    def reset(self):
        self.x = np.zeros(4); self.P = np.eye(4) * 500.0; self.initialized = False; self.last_t = None; self.miss = 0


# 골 기억 (IMU world-angle)
class GoalMemory:
    def __init__(self, tuning: TuningParams):
        self.t = tuning
        self.goal_world_angle = 0.0
        self.confidence = 0.0
        self.initialized = False
        self._last_imu = 0.0

    def update_from_detection(self, goal_screen_x, head_yaw, imu_yaw, area, conf):
        # 화면상 골 방향 → world angle = imu_yaw + head_yaw + 화면각
        screen_ang = -goal_screen_x * 0.5
        self.goal_world_angle = imu_yaw + head_yaw + screen_ang
        self.confidence = min(1.0, self.confidence + 0.5 * conf)
        self.initialized = True
        self._last_imu = imu_yaw

    def update_confidence_decay(self, current_imu_yaw: float = None):
        self.confidence = max(0.0, self.confidence - self.t.goal_memory_decay)
        if current_imu_yaw is not None:
            dyaw = abs(current_imu_yaw - self._last_imu)
            self.confidence = max(0.0, self.confidence - dyaw * self.t.goal_yaw_rate_decay)
            self._last_imu = current_imu_yaw

    def get_goal_direction(self, imu_yaw: float) -> float:
        return (self.goal_world_angle - imu_yaw + math.pi) % (2 * math.pi) - math.pi

    def is_valid(self) -> bool:
        return self.initialized and self.confidence >= self.t.goal_conf_threshold

    def reset(self):
        self.initialized = False; self.confidence = 0.0; self.goal_world_angle = 0.0


class HeadStabilizer:
    def __init__(self, gain=0.6): self.gain = gain
    def update(self, body_roll, body_pitch):
        return -self.gain * body_roll, -self.gain * body_pitch
    def reset(self): pass


class AdaptiveHeadController:
    def __init__(self, max_speed=0.02, smoothing=0.15):
        self.max_speed = max_speed; self.smoothing = smoothing
        self.target_yaw = 0.0; self.target_pitch = 0.0
        self.filtered_yaw = 0.0; self.filtered_pitch = 0.0
        self.approach_mode = False; self.search_mode = False
    def set_approach_mode(self, e): self.approach_mode = e; self.search_mode = False if e else self.search_mode
    def set_search_mode(self, e): self.search_mode = e; self.approach_mode = False if e else self.approach_mode
    def set_target(self, yaw, pitch): self.target_yaw = float(yaw); self.target_pitch = float(pitch)
    def set_target_direct(self, yaw, pitch): self.target_yaw = float(yaw); self.target_pitch = float(pitch)
    def get_command(self):
        a = self.smoothing
        self.filtered_yaw += a * (self.target_yaw - self.filtered_yaw)
        self.filtered_pitch += a * (self.target_pitch - self.filtered_pitch)
        return self.filtered_yaw, self.filtered_pitch
    def reset(self, yaw=0.0, pitch=0.0):
        self.target_yaw = self.filtered_yaw = yaw
        self.target_pitch = self.filtered_pitch = pitch


class GoalDetector:
    def __init__(self):
        self.detected = False; self.screen_x = 0.0; self.consecutive_detect = 0
    def update(self, detected, screen_x=0.0, screen_y=0.0, area=0.0, width=0.0, conf=0.0):
        self.detected = detected
        if detected:
            self.screen_x = screen_x; self.consecutive_detect += 1
        else:
            self.consecutive_detect = 0
    def is_visible(self): return self.detected and self.consecutive_detect >= 5
    def get_screen_x(self): return self.screen_x if self.detected else 0.0


class BallTracker:
    def __init__(self, buffer_size=12):
        self.buffer = deque(maxlen=buffer_size)
        self.consecutive_detect = 0; self.consecutive_miss = 0
        self.last_valid_area = None; self.last_valid_pos = None

    def validate_ball(self, c: BallCandidate, t: TuningParams, cfg: RobotConfig) -> Tuple[bool, str]:
        if c.conf < t.ball_conf_threshold: return False, "conf"
        if c.area < t.ball_min_area or c.area > t.ball_max_area: return False, "size"
        if c.aspect_ratio < cfg.min_aspect_ratio or c.aspect_ratio > cfg.max_aspect_ratio: return False, "AR"
        if self.last_valid_area is not None:
            close = self.last_valid_area > cfg.close_ball_area_threshold
            mx = cfg.close_ball_max_area_change if close else cfg.max_area_change_ratio
            r = c.area / self.last_valid_area
            if r > mx or r < 1 / mx: return False, "area_jump"
        return True, "OK"

    def update(self, c: Optional[BallCandidate], t: TuningParams, cfg: RobotConfig) -> Tuple[bool, str]:
        if c is None:
            self.buffer.append(None); self.consecutive_miss += 1; self.consecutive_detect = 0
            return False, "none"
        ok, why = self.validate_ball(c, t, cfg)
        if ok:
            self.buffer.append(c); self.consecutive_detect += 1; self.consecutive_miss = 0
            self.last_valid_area = c.area; self.last_valid_pos = (c.x, c.y)
        else:
            self.buffer.append(None); self.consecutive_miss += 1; self.consecutive_detect = 0
        return ok, why

    def reset(self):
        self.buffer.clear(); self.consecutive_detect = 0; self.consecutive_miss = 0
        self.last_valid_area = None; self.last_valid_pos = None


# VisionCore
class VisionCore:
    """검출 결과에서 공과 골 관측값을 만든다. MuJoCo와 ROS에 의존하지 않는다."""
    def __init__(self, cfg: RobotConfig, tuning: TuningParams, class_id_map: Optional[dict] = None):
        self.cfg = cfg; self.t = tuning
        self.kalman = BallKalmanFilter()
        self.tracker = BallTracker(cfg.detection_buffer_size)
        self.goal_memory = GoalMemory(tuning)
        self.goal_detector = GoalDetector()
        # 이름 기반 해석 결과 (없으면 cfg id 폴백)
        m = class_id_map or {}
        self.ball_ids = set(m.get("ball") or {cfg.ball_class_id})
        self.goal_ids = set(m.get("goal") or set(cfg.goal_class_ids))
        self._fid = 0
        self._obs = BallObservation()
        self._lock = threading.Lock()
        self.last_valid_dist = 2.0

    # 거리/각
    def get_depth(self, depth, cx, cy) -> Optional[float]:
        if depth is None: return None
        r = self.cfg.depth_sample_radius
        h, w = depth.shape[:2]
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        patch = depth[y0:y1, x0:x1]
        patch = patch[(patch > self.cfg.depth_min_valid) & (patch < self.cfg.depth_max_valid + 5.0)]
        return float(np.median(patch)) if patch.size else None

    def estimate_distance_from_area(self, area: float) -> float:
        if area <= 1.0: return 9.99
        px_d = math.sqrt(area)
        return float(self.cfg.ball_real_diameter * self.cfg.camera_focal_length / max(1.0, px_d))

    def _box_cls(self, box) -> int:
        return int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)

    def _box_conf(self, box) -> float:
        return float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)

    def _box_xywh(self, box):
        v = box.xywh.cpu().numpy()[0] if hasattr(box.xywh, "cpu") else np.asarray(box.xywh)[0]
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])

    def find_ball_candidates(self, boxes) -> List[BallCandidate]:
        out = []
        for b in boxes:
            if self.cfg.use_class_filter and self._box_cls(b) not in self.ball_ids:
                continue
            bx, by, bw, bh = self._box_xywh(b)
            ar = bw / bh if bh > 0 else 0.0
            out.append(BallCandidate(bx, by, bw, bh, bw * bh, self._box_conf(b), ar, 1.0 - abs(1.0 - ar)))
        out.sort(key=lambda c: c.conf * c.circularity, reverse=True)
        return out

    def find_goal(self, boxes, img_w, img_h) -> Optional[dict]:
        for b in boxes:
            if self._box_cls(b) not in self.goal_ids:
                continue
            conf = self._box_conf(b)
            if conf < self.cfg.goal_conf_weak:
                continue
            bx, by, bw, bh = self._box_xywh(b)
            if bw * bh < self.cfg.goal_min_area:
                continue
            return {"norm_x": (bx - img_w / 2) / (img_w / 2), "norm_y": (by - img_h / 2) / (img_h / 2),
                    "area": bw * bh, "width": bw, "height": bh, "conf": conf, "cx": int(bx), "cy": int(by)}
        return None

    def screen_to_angle(self, sx, sy):
        ax = sx * (self.cfg.camera_fov_h / 2); ay = sy * (self.cfg.camera_fov_v / 2)
        yaw = ax if self.cfg.head_yaw_inverted else -ax
        pitch = ay if self.cfg.head_pitch_inverted else -ay
        return float(yaw), float(pitch)

    # 공유 관측
    def get_observation(self) -> BallObservation:
        with self._lock:
            return self._obs

    def _write(self, obs: BallObservation):
        with self._lock:
            self._obs = obs

    def process(self, boxes, depth, img_w, img_h, imu_yaw=0.0, head_yaw=0.0):
        """단일 프레임 처리 → (obs, goal_data, candidates)."""
        self._fid += 1
        self.goal_memory.update_confidence_decay(current_imu_yaw=imu_yaw)

        # 골대
        goal_data = self.find_goal(boxes, img_w, img_h) if boxes is not None else None
        if goal_data:
            self.goal_detector.update(True, goal_data["norm_x"], area=goal_data["area"],
                                      width=goal_data["width"], conf=goal_data["conf"])
            if self.goal_detector.consecutive_detect >= 2:
                self.goal_memory.update_from_detection(goal_data["norm_x"], head_yaw, imu_yaw,
                                                       goal_data["area"], goal_data["conf"])
        else:
            self.goal_detector.update(False)

        # 공
        candidates = self.find_ball_candidates(boxes) if boxes is not None else []
        best = None
        for c in candidates:
            ok, _ = self.tracker.validate_ball(c, self.t, self.cfg)
            if ok:
                best = c; break
        detected, _ = self.tracker.update(best, self.t, self.cfg)

        if detected and best is not None:
            depth_d = self.get_depth(depth, best.x, best.y)
            has_depth = depth_d is not None and self.cfg.depth_min_valid < depth_d < self.cfg.depth_max_valid
            self.kalman.update(np.array([best.x, best.y]), use_depth=has_depth)
            px, py = self.kalman.get_predicted_position(self.cfg.kalman_predict_time)
            nx = (px - img_w / 2) / (img_w / 2)
            ny = (py - img_h / 2) / (img_h / 2)
            dist_area = self.estimate_distance_from_area(best.area)
            dist = depth_d if has_depth else dist_area
            if dist < 2.5:
                self.last_valid_dist = dist
            obs = BallObservation(
                area=best.area, dist=dist, dist_from_area=dist_area,
                screen_x=float(nx), screen_y=float(ny),
                last_dir=float(np.sign(nx)) if abs(nx) > 0.02 else 0.0,
                lost_frames=0, last_valid_dist=self.last_valid_dist, conf=best.conf,
                timestamp=now_sec(), detected=True,
                consecutive_detect=self.tracker.consecutive_detect,
                kalman_accepted=has_depth, frame_id=self._fid)
        else:
            prev = self.get_observation()
            obs = BallObservation(
                area=0.0, dist=9.99, dist_from_area=9.99,
                screen_x=prev.screen_x, screen_y=prev.screen_y, last_dir=prev.last_dir,
                lost_frames=prev.lost_frames + 1, last_valid_dist=self.last_valid_dist,
                conf=0.0, timestamp=now_sec(), detected=False, consecutive_detect=0,
                kalman_accepted=False, frame_id=self._fid)

        self._write(obs)
        return obs, goal_data, candidates

    def reset(self):
        self.kalman.reset(); self.tracker.reset(); self.goal_memory.reset()
