#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 카메라 영상의 YOLO 검출과 TRON1 IMU로 공·골대 관측 및 위치 추정값을 만든다.
RealRobotBrain이 관측과 선택적 GameController·setplay 입력으로 상태를 결정하고 WebSocket 보행 명령과 ROS 상체 모터 명령을 출력한다.
비전과 제어는 별도 스레드로 실행하며 피드백, watchdog, 자세 한계를 감시한다.
실기 실행 전 모터 매핑, 카메라 보정값, 검출 클래스와 depth 스케일을 확인해야 하며 ROS 통합 실행은 별도 검증이 필요하다."""

import rospy
import cv2
import numpy as np
import json
import time
import threading
import math
import websocket
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum, auto
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from tronb_vision.real_position import RealPositionEstimator, RealCameraGeometry
from tronb_vision.sources import CameraIntrinsics
from tronb_vision.localization import LocalizationEngine
from tronb_vision.features import BallStopEstimator
from tronb_vision.setplay import SetPlayVisionAdapter, SetPlayVisionConfig
from tronb_vision.tronb_brain import RealRobotBrain
from tronb_vision.realrobot_glue import make_ball_obs, make_goal_obs, build_robot_state
from tronb_vision.vision_state import VisionStateConfig
try:
    from tronb_vision.gc_state import GcReceiver
except Exception:
    GcReceiver = None

try:
    from tron_upper_body.msg import MotorCommand, MotorFeedback
except ImportError:
    class MotorCommand: pass
    class MotorFeedback: pass

# 선택적 위치 추정 입력은 std_msgs/String의 JSON 형식이다.
try:
    from std_msgs.msg import String as RosString
except ImportError:
    RosString = None


# Vision은 데이터만 쓰고, Control이 상태 전이를 결정
@dataclass
class BallObservation:
    """Vision loop에서 한 번에 쓰고, Control loop에서 한 번에 읽는 단위"""
    area: float = 0.0
    dist: float = 9.99
    dist_from_area: float = 9.99
    screen_x: float = 0.0
    screen_y: float = 0.0
    last_dir: float = 0.0
    lost_frames: int = 0
    last_valid_dist: float = 2.0
    timestamp: float = 0.0

    detected: bool = False
    consecutive_detect: int = 0
    kalman_accepted: bool = False         # 칼만 필터가 이번 관측 수용했는지

    # 프레임 동기화
    frame_id: int = 0                     # Vision이 쓸 때마다 증가, Control이 중복 처리 방지


class RobotState(Enum):
    SEARCH = auto()
    TRACK = auto()
    APPROACH = auto()
    AIM = auto()        # 정지, 골대 정렬, 킥 판정
    KICK = auto()
    COOLDOWN = auto()
    RECOVER = auto()


@dataclass
class TuningParams:
    """현장 로봇과 환경에 맞춰 조정하는 제어 파라미터."""

    # 골대 정렬 / 킥
    align_good_angle: float = 0.10      # 이 각도(rad) 이하면 정렬 완료로 보고 킥 (~5.7도)
    align_max_angle: float = 0.16       # 참고용 상한 (~9도)
    align_turn_speed: float = 0.08      # AIM에서 제자리 정렬 회전 속도 (rad/s)
    kick_ball_center_threshold: float = 0.15
    kick_settle_time: float = 0.3       # 킥 전 안정화 시간 (초)
    kick_area_threshold: int = 45000
    kick_dist_threshold: float = 0.45
    kick_speed: float = 0.80

    # AIM은 골대 기억을 우선 사용하고 짧게 스캔한 뒤 직진 킥으로 폴백한다.
    aim_settle_time: float = 0.25       # 진입 직후 안정화 (초)
    aim_glance_time: float = 1.2        # 골대 기억 없을 때만 고개 들어 스캔하는 최대 시간 (초)
    aim_timeout: float = 2.5            # 제한 시간 뒤에는 정렬 여부와 관계없이 킥
    aim_look_pitch: float = 0.2        # 글랜스 시 head pitch (rad)

    # 지연 보상 조기 정지 (공이 발밑 오기 전에 멈춤)
    perception_latency: float = 0.15    # 비전 파이프라인 지연(초), 실측값으로 조정
    stop_dist: float = 0.50             # 보상 거리가 이 값 이하면 AIM 진입 (m)
    stop_confirm_frames: int = 2        # AIM 확정 프레임 수

    # 곡선 주행 (공 근처에서 골대쪽으로 미리 휘어 접근 → AIM 회전량 최소화)
    curved_approach_enabled: bool = True
    curved_approach_dist: float = 0.6
    curved_approach_gain: float = 0.15

    ball_conf_threshold: float = 0.40
    ball_min_area: int = 300
    ball_max_area: int = 90000
    lost_frames_threshold: int = 35

    # 골대 기억 (IMU world-angle 기반)
    goal_memory_decay: float = 0.02     # 초당 신뢰도 감소
    goal_memory_timeout: float = 30.0   # 기억 유효 시간 (초)
    goal_conf_threshold: float = 0.25
    goal_yaw_rate_decay: float = 0.4    # yaw 회전량당 추가 decay

    approach_speed_far: float = 0.060
    approach_speed_near: float = 0.025
    search_turn_speed: float = 0.10     # SEARCH 좌우 회전 최대 (rad/s)

    search_drift_compensation: float = -0.015   # 전진 drift 보상 (음수=후진)
    search_turn_period: float = 6.0
    search_turn_bias: float = 0.03
    search_head_scan_speed: float = 0.030
    search_head_pitch_center: float = -0.15

    straight_drift_correction: float = -0.05
    approach_drift_correction: float = -0.05

    def print_summary(self):
        print("\n" + "="*46)
        print("TuningParams v9")
        print("="*46)
        print(f"align_good_angle  = {self.align_good_angle:.2f} rad ({math.degrees(self.align_good_angle):.1f} deg)")
        print(f"aim_glance/timeout= {self.aim_glance_time:.1f} / {self.aim_timeout:.1f} s")
        print(f"perception_latency= {self.perception_latency:.3f} s")
        print(f"stop_dist         = {self.stop_dist:.2f} m  (confirm {self.stop_confirm_frames})")
        print(f"curved_approach   = {self.curved_approach_enabled} (gain {self.curved_approach_gain:.2f})")
        print(f"goal_yaw_decay    = {self.goal_yaw_rate_decay:.2f}")
        print(f"search drift/bias = {self.search_drift_compensation:.3f} / {self.search_turn_bias:.3f}")
        print("="*46 + "\n")


# 공 칼만 필터 (Adaptive Q + Mahalanobis Gating)
class BallKalmanFilter:
    def __init__(self):
        self.x = np.zeros(4)       # [px, py, vx, vy]
        self.P = np.eye(4) * 100
        self.Q_base = np.diag([1.0, 1.0, 5.0, 5.0])
        self.R_default = np.diag([10.0, 10.0])
        self.R_area_only = np.diag([25.0, 25.0])  # area 기반 추정 시 더 높은 불확실성
        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.last_update = time.time()
        self.initialized = False
        # Mahalanobis gate (chi-squared, 2 DOF, 99.9% → 13.82, 실전 여유 → 16.0)
        # 상대 로봇 킥 등 급격한 공 이동에도 tracking 유지
        self.mahalanobis_gate = 16.0
        self.consecutive_reject = 0
        self.max_consecutive_reject = 5

    def _get_adaptive_Q(self, dt: float) -> np.ndarray:
        """공 속도에 따라 프로세스 노이즈 조정"""
        speed = math.sqrt(self.x[2]**2 + self.x[3]**2)
        speed_factor = 1.0 + min(speed / 50.0, 5.0)
        Q = self.Q_base.copy()
        Q[2, 2] *= speed_factor
        Q[3, 3] *= speed_factor
        return Q * max(dt, 0.01)

    def predict(self, dt: float = None):
        if dt is None:
            dt = time.time() - self.last_update
        dt = max(dt, 1e-4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        Q = self._get_adaptive_Q(dt)
        self.P = self.F @ self.P @ self.F.T + Q
        return self.x[:2]

    def update(self, z: np.ndarray, use_depth: bool = True):
        """
        z: [px, py] measurement
        use_depth: depth 카메라로 거리 확인된 경우 True, area 추정만이면 False
        Returns: True if measurement accepted, False if rejected by gate
        """
        if not self.initialized:
            self.x[:2] = z
            self.initialized = True
            self.last_update = time.time()
            self.consecutive_reject = 0
            return True

        self.predict()

        R = self.R_default if use_depth else self.R_area_only

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        # Mahalanobis distance (NIS: Normalized Innovation Squared)
        # NIS = y^T * S^-1 * y  (2 DOF chi-squared 분포)
        try:
            S_inv = np.linalg.inv(S)
            nis = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            # S가 특이행렬이면 유클리드 거리로 폴백한다.
            nis = (y[0]**2 + y[1]**2) / max(S[0, 0], 1.0)

        if nis > self.mahalanobis_gate:
            self.consecutive_reject += 1
            rospy.loginfo_throttle(2.0,
                f"[Kalman] reject: NIS={nis:.1f} > gate={self.mahalanobis_gate:.1f} "
                f"(#{self.consecutive_reject})")
            if self.consecutive_reject >= self.max_consecutive_reject:
                rospy.logwarn(f"[Kalman] {self.consecutive_reject}회 연속 reject → 리셋")
                self.reset()
                self.x[:2] = z
                self.initialized = True
                self.last_update = time.time()
                self.consecutive_reject = 0
                return True
            return False

        self.consecutive_reject = 0
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_update = time.time()
        return True

    def get_position(self) -> Tuple[float, float]:
        return self.x[0], self.x[1]

    def get_velocity(self) -> Tuple[float, float]:
        """velocity vector (blind approach에서 사용)"""
        return self.x[2], self.x[3]

    def get_predicted_position(self, dt: float) -> Tuple[float, float]:
        return self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt

    def get_speed(self) -> float:
        """pixel/s 단위 공 속도"""
        return math.sqrt(self.x[2]**2 + self.x[3]**2)

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False
        self.consecutive_reject = 0


# IMU 기반 골대 기억 (Yaw-rate decay)
class GoalMemory:
    def __init__(self, tuning: TuningParams):
        self.tuning = tuning
        self.goal_world_angle = 0.0
        self.initialized = False
        self.last_update_time = 0.0
        self.confidence = 0.0
        self.last_screen_x = 0.0
        self.last_area = 0.0
        self._prev_imu_yaw = 0.0
        self._accumulated_yaw_delta = 0.0  # 마지막 관측 이후 총 yaw 변화량

    def update_from_detection(self, goal_screen_x: float, head_yaw: float,
                               imu_yaw: float, area: float, conf: float):
        screen_angle = goal_screen_x * 0.52
        goal_rel_angle = head_yaw + screen_angle
        new_world_angle = imu_yaw + goal_rel_angle

        if self.initialized:
            alpha = 0.3
            self.goal_world_angle = alpha * new_world_angle + (1 - alpha) * self.goal_world_angle
        else:
            self.goal_world_angle = new_world_angle

        self.initialized = True
        self.last_update_time = time.time()
        self.confidence = min(1.0, self.confidence + conf * 0.5)
        self.last_screen_x = goal_screen_x
        self.last_area = area
        self._prev_imu_yaw = imu_yaw
        self._accumulated_yaw_delta = 0.0

    def update_confidence_decay(self, current_imu_yaw: float = None):
        """
        시간 기반 decay + IMU yaw 변화량 비례 decay
        로봇이 많이 회전할수록 골대 위치에 대한 확신이 빠르게 줄어듦
        """
        if not self.initialized:
            return

        elapsed = time.time() - self.last_update_time
        time_decay = elapsed * self.tuning.goal_memory_decay * 0.1

        yaw_decay = 0.0
        if current_imu_yaw is not None:
            yaw_delta = abs(current_imu_yaw - self._prev_imu_yaw)
            # wrap-around 처리
            if yaw_delta > math.pi:
                yaw_delta = 2 * math.pi - yaw_delta
            self._accumulated_yaw_delta += yaw_delta
            self._prev_imu_yaw = current_imu_yaw

            yaw_decay = self._accumulated_yaw_delta * self.tuning.goal_yaw_rate_decay * 0.01

        total_decay = time_decay + yaw_decay
        self.confidence = max(0.0, self.confidence - total_decay)

    def get_goal_direction(self, imu_yaw: float) -> float:
        if not self.initialized:
            return 0.0
        rel_dir = self.goal_world_angle - imu_yaw
        while rel_dir > math.pi:
            rel_dir -= 2 * math.pi
        while rel_dir < -math.pi:
            rel_dir += 2 * math.pi
        return rel_dir

    def get_confidence(self) -> float:
        return self.confidence

    def is_valid(self) -> bool:
        if not self.initialized:
            return False
        if self.confidence < 0.2:
            return False
        if time.time() - self.last_update_time > self.tuning.goal_memory_timeout:
            return False
        return True

    def get_status_string(self, imu_yaw: float) -> str:
        if not self.initialized:
            return "NO_MEMORY"
        rel_dir = self.get_goal_direction(imu_yaw)
        age = time.time() - self.last_update_time
        if self.is_valid():
            return f"MEM dir:{rel_dir:.2f}rad ({math.degrees(rel_dir):.0f}°) conf:{self.confidence:.2f}"
        else:
            return f"STALE"

    def reset(self):
        self.goal_world_angle = 0.0
        self.initialized = False
        self.confidence = 0.0


class HeadStabilizer:
    def __init__(self, compensation_gain: float = 0.6):
        self.compensation_gain = compensation_gain
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.alpha = 0.3
        self.prev_pitch = 0.0
        self.prev_time = time.time()

    def update(self, body_roll: float, body_pitch: float) -> Tuple[float, float]:
        now = time.time()
        dt = max(now - self.prev_time, 0.001)
        self.filtered_roll = self.alpha * body_roll + (1 - self.alpha) * self.filtered_roll
        self.filtered_pitch = self.alpha * body_pitch + (1 - self.alpha) * self.filtered_pitch
        pitch_rate = (self.filtered_pitch - self.prev_pitch) / dt
        self.prev_pitch = self.filtered_pitch
        self.prev_time = now
        yaw_comp = 0.0
        pitch_comp = -self.filtered_pitch * self.compensation_gain
        pitch_comp += -pitch_rate * 0.05 * self.compensation_gain * 0.5
        return yaw_comp, pitch_comp

    def reset(self):
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.prev_pitch = 0.0


class AdaptiveHeadController:
    def __init__(self, max_speed: float = 0.02, smoothing: float = 0.15):
        self.max_speed = max_speed
        self.smoothing = smoothing
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0
        self.approach_mode = False
        self.search_mode = False           # SEARCH용 모드
        self.approach_smoothing = 0.25
        self.search_smoothing = 0.4        # SEARCH에서는 단일 EMA로 추종 지연을 줄인다.

    def set_approach_mode(self, enabled: bool):
        self.approach_mode = enabled
        if enabled:
            self.search_mode = False

    def set_search_mode(self, enabled: bool):
        """SEARCH에서는 빠른 추종을 위해 단일 EMA를 사용한다."""
        self.search_mode = enabled
        if enabled:
            self.approach_mode = False

    def set_target(self, yaw: float, pitch: float):
        dyaw = np.clip(yaw - self.target_yaw, -self.max_speed, self.max_speed)
        dpitch = np.clip(pitch - self.target_pitch, -self.max_speed, self.max_speed)
        self.target_yaw += dyaw
        self.target_pitch += dpitch

    def set_target_direct(self, yaw: float, pitch: float):
        self.target_yaw = yaw
        self.target_pitch = pitch

    def get_command(self) -> Tuple[float, float]:
        if self.search_mode:
            # 단일 EMA로 목표를 추종한다.
            alpha = self.search_smoothing
            self.current_yaw = alpha * self.target_yaw + (1 - alpha) * self.current_yaw
            self.current_pitch = alpha * self.target_pitch + (1 - alpha) * self.current_pitch
            self.filtered_yaw = self.current_yaw
            self.filtered_pitch = self.current_pitch
        elif self.approach_mode:
            alpha = self.approach_smoothing
            self.current_yaw = alpha * self.target_yaw + (1 - alpha) * self.current_yaw
            self.current_pitch = alpha * self.target_pitch + (1 - alpha) * self.current_pitch
            self.filtered_yaw = self.current_yaw
            self.filtered_pitch = self.current_pitch
        else:
            self.current_yaw = self.smoothing * self.target_yaw + (1 - self.smoothing) * self.current_yaw
            self.current_pitch = self.smoothing * self.target_pitch + (1 - self.smoothing) * self.current_pitch
            self.filtered_yaw = self.smoothing * self.current_yaw + (1 - self.smoothing) * self.filtered_yaw
            self.filtered_pitch = self.smoothing * self.current_pitch + (1 - self.smoothing) * self.filtered_pitch
        return self.filtered_yaw, self.filtered_pitch

    def reset(self, yaw: float = 0.0, pitch: float = 0.0):
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.target_yaw = yaw
        self.target_pitch = pitch
        self.filtered_yaw = yaw
        self.filtered_pitch = pitch
        self.approach_mode = False
        self.search_mode = False


class GoalDetector:
    def __init__(self):
        self.detected = False
        self.screen_x = 0.0
        self.screen_y = 0.0
        self.area = 0.0
        self.width = 0.0
        self.conf = 0.0
        self.consecutive_detect = 0
        self.consecutive_miss = 0

    def update(self, detected: bool, screen_x: float = 0.0, screen_y: float = 0.0,
               area: float = 0.0, width: float = 0.0, conf: float = 0.0):
        if detected:
            self.detected = True
            self.screen_x = screen_x
            self.screen_y = screen_y
            self.area = area
            self.width = width
            self.conf = conf
            self.consecutive_detect += 1
            self.consecutive_miss = 0
        else:
            self.consecutive_miss += 1
            self.consecutive_detect = 0
            if self.consecutive_miss > 5:
                self.detected = False

    def is_visible(self) -> bool:
        return self.detected and self.consecutive_detect >= 5

    def get_screen_x(self) -> float:
        return self.screen_x if self.detected else 0.0


@dataclass
class BallCandidate:
    x: float
    y: float
    w: float
    h: float
    area: float
    conf: float
    aspect_ratio: float
    circularity: float


@dataclass
class RobotConfig:
    """하드웨어, 통신, 모터 제한과 인지 설정."""
    robot_sn: str = "PF_TRON1A_359"
    ws_url: str = "ws://192.0.2.11:5000"
    model_path: str = "weights/best8.pt"
    debug: bool = True
    verbose_log: bool = True

    # Motor IDs
    head_yaw_id: int = 5
    head_pitch_id: int = 6
    l_roll_id: int = 1
    l_pitch_id: int = 3
    r_roll_id: int = 2
    r_pitch_id: int = 4

    ball_class_id: int = 0
    goal_class_id: int = 1
    use_class_filter: bool = True

    depth_scale: float = 0.001
    depth_sample_radius: int = 5
    depth_min_valid: float = 0.1
    depth_max_valid: float = 3.0

    conf_threshold_strict: float = 0.55
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.50
    max_area_change_ratio: float = 3.5
    min_consecutive_detect: int = 3
    close_ball_area_threshold: int = 15000
    close_ball_max_area_change: float = 5.0

    ball_real_diameter: float = 0.22
    camera_focal_length: float = 600

    detection_buffer_size: int = 12

    head_yaw_inverted: bool = True
    head_pitch_inverted: bool = False
    camera_fov_h: float = 1.047
    camera_fov_v: float = 0.785
    head_max_speed: float = 0.015
    head_smoothing: float = 0.12
    head_approach_smoothing: float = 0.25
    head_pitch_offset_target: float = 1.3   # 골대를 볼 수 있도록 머리를 올리는 오프셋
    head_init_steps: int = 40
    head_init_kp: float = 8.0
    head_yaw_limit: Tuple[float, float] = (-0.6, 0.6)
    head_pitch_limit: Tuple[float, float] = (-0.70, 0.30)

    imu_compensation_gain: float = 0.5

    kalman_predict_time: float = 0.1

    scan_pitch_center: float = -0.05
    scan_pitch_amplitude: float = 0.8
    scan_yaw_amplitude: float = 0.0
    scan_speed: float = 0.010

    approach_pitch_far: float = -0.30
    approach_pitch_near: float = -0.65
    approach_pitch_dist_far: float = 1.5
    approach_pitch_dist_near: float = 0.4

    # 몸 회전
    body_turn_deadzone: float = 0.03
    body_turn_gain: float = 0.90
    body_turn_inverted: bool = False
    body_slew_rate: float = 0.040
    max_linear_speed: float = 0.08
    max_angular_speed: float = 0.35
    acc_limit: float = 0.025
    dec_limit: float = 0.040
    kick_max_speed: float = 0.85
    kick_acc_limit: float = 0.5
    head_body_compensation: float = 0.50

    # 역관절 결합에서 카메라 전방은 TRON1의 -x 방향이다.
    # 내부 계산은 공 접근을 양수로 유지한다.
    # 지연 보상의 좌표계를 유지하기 위해 송신 단계에서만 twist x를 반전한다.
    # TRON1 후진 속도 한계는 미확인이다. 작은 속도에서 시작해 실기 한계를 확인한다.
    reverse_gait: bool = True
    reverse_max_speed: float = 0.03    # 후진 접근·탐색 속도 상한 [m/s]
    reverse_kick_speed: float = 0.06   # 후진 킥 속도 상한 [m/s]

    track_to_approach_yaw: float = 0.15
    track_min_frames: int = 5
    track_min_dist: float = 2.0
    approach_to_track_yaw: float = 0.35
    lost_ball_turn_speed: float = 0.08
    lost_ball_search_frames: int = 15

    # 킥 (기본값, 튜닝파람으로 override됨)
    kick_area_very_close: int = 55000
    kick_dist_strict: float = 0.25
    kick_align_threshold: float = 0.15
    kick_confirm_required: int = 5
    kick_duration: float = 1.5
    blind_kick_wait: float = 0.1
    normal_kick_wait: float = 0.25

    # 골대
    goal_conf_weak: float = 0.15
    goal_min_area: int = 300
    goal_align_max_time: float = 3.0
    goal_align_max_frames: int = 60
    goal_real_width: float = 2.6

    approach_dist_far: float = 1.4
    approach_dist_near: float = 0.5

    roll_warning_limit: float = 0.25
    roll_fall_limit: float = 0.35
    kick_zmp_factor: float = 0.20

    cooldown_time: float = 2.5
    recover_time: float = 1.5

    arm_rate_limit_normal: float = 0.03
    arm_rate_limit_kick: float = 0.025
    arm_base_roll: float = 0.20
    arm_swing_base: float = 0.10
    arm_swing_speed_gain: float = 0.20
    arm_swing_freq: float = 4.0
    arm_pitch_inverted: bool = True

    vision_timeout: float = 3.0
    control_timeout: float = 1.0
    ws_reconnect_interval: float = 3.0

    motor_fb_timeout: float = 0.50        # 이 시간(s) 넘게 모터 피드백 없으면 해당 관절 compliant.
                                          # 실측 피드백 주기와 최대 지연을 기준으로 조정한다.
    follow_err_limit: float = 0.70        # |지령-실제|(rad) 넘으면 잼/런어웨이로 보고 토크 백오프.
                                          # 자유 스윙과 킥 동작이 오검출되지 않도록 실기에서 확인한다.
    follow_err_kp_backoff: float = 2.0    # 트립/스테일/정지 시 낮출 kp (토크 완화)
    pitch_fall_limit: float = 0.40        # 전/후 기울기 fall 한계 (rad). roll_fall_limit과 함께 사용
    # 관절별 offset-상대 지령 절대한계 (오프셋이 틀려도 gross runaway 차단)
    rel_limit_head_yaw: float = 0.70
    rel_limit_head_pitch: float = 0.90
    rel_limit_arm_roll: float = 0.50
    rel_limit_arm_pitch: float = 0.90

    # 외부 위치 추정 입력 (기본 비활성)
    localization_enabled: bool = False
    localization_pose_topic: str = "/localization/pose"   # vision 노드가 JSON publish
    localization_odom_topic: str = "/localization/odom"   # behavior -> localization predict 입력
    localization_pose_max_age: float = 0.50               # 이보다 오래된 pose는 무시
    attack_positive_x: bool = True        # 공격 골대 방향 (전반/후반 교대 시 토글)


class BallTracker:
    def __init__(self, buffer_size=12):
        self.buffer = deque(maxlen=buffer_size)
        self.consecutive_detect = 0
        self.consecutive_miss = 0
        self.last_valid_area = None
        self.last_valid_pos = None
        self.confidence_history = deque(maxlen=5)

    def validate_ball(self, candidate: BallCandidate, tuning: TuningParams, cfg: RobotConfig) -> Tuple[bool, str]:
        if candidate.conf < tuning.ball_conf_threshold:
            return False, f"conf low"
        if candidate.area < tuning.ball_min_area or candidate.area > tuning.ball_max_area:
            return False, f"size"
        ar = candidate.aspect_ratio
        if ar < cfg.min_aspect_ratio or ar > cfg.max_aspect_ratio:
            return False, f"AR={ar:.2f}"

        if self.last_valid_area is not None:
            is_close = self.last_valid_area > cfg.close_ball_area_threshold
            max_change = cfg.close_ball_max_area_change if is_close else cfg.max_area_change_ratio
            area_ratio = candidate.area / self.last_valid_area
            if area_ratio > max_change or area_ratio < 1/max_change:
                return False, f"area jump"

        if self.last_valid_pos is not None:
            dx = abs(candidate.x - self.last_valid_pos[0])
            dy = abs(candidate.y - self.last_valid_pos[1])
            max_jump = 400 if (self.last_valid_area and self.last_valid_area > cfg.close_ball_area_threshold) else 300
            if dx > max_jump or dy > max_jump:
                return False, f"pos jump"
        return True, "OK"

    def update(self, candidate: Optional[BallCandidate], tuning: TuningParams, cfg: RobotConfig) -> Tuple[bool, str]:
        if candidate is None:
            self.buffer.append(None)
            self.consecutive_miss += 1
            self.consecutive_detect = 0
            return False, "no detection"
        valid, reason = self.validate_ball(candidate, tuning, cfg)
        if valid:
            self.buffer.append(candidate)
            self.consecutive_detect += 1
            self.consecutive_miss = 0
            self.last_valid_area = candidate.area
            self.last_valid_pos = (candidate.x, candidate.y)
            self.confidence_history.append(candidate.conf)
            return True, reason
        else:
            self.buffer.append(None)
            self.consecutive_miss += 1
            self.consecutive_detect = 0
            return False, reason

    def reset(self):
        self.buffer.clear()
        self.consecutive_detect = 0
        self.consecutive_miss = 0
        self.last_valid_area = None
        self.last_valid_pos = None
        self.confidence_history.clear()


class Watchdog:
    def __init__(self, timeout, name, on_timeout=None):
        self.timeout, self.name = timeout, name
        self.last_feed = time.time()
        self.lock = threading.Lock()
        self.running = True
        self.armed = False  # 부팅 완료 전에는 모니터링 안 함
        self.on_timeout = on_timeout   # stuck 시 호출 (SAFE STOP)
        self.tripped = False           # 중복 콜백 방지
        threading.Thread(target=self._monitor, daemon=True).start()
    def feed(self):
        with self.lock: self.last_feed = time.time()
    def arm(self):
        """부팅 완료 후 호출 — 이때부터 stuck 감시 시작"""
        with self.lock:
            self.armed = True
            self.last_feed = time.time()
    def stop(self): self.running = False
    def _monitor(self):
        while self.running:
            time.sleep(self.timeout / 2)
            with self.lock:
                stuck = self.armed and (time.time() - self.last_feed > self.timeout)
            # 타임아웃당 정지 콜백은 한 번만 호출한다.
            if stuck and not self.tripped:
                self.tripped = True
                print(f"[WATCHDOG] {self.name} stuck -> SAFE STOP")
                if self.on_timeout:
                    try:
                        self.on_timeout(self.name)
                    except Exception as e:
                        print(f"[WATCHDOG] callback error: {e}")


class LocalizationClient:
    """std_msgs/String의 JSON pose와 로봇 기준 골대 방향을 구독한다.

    입력: {"x":m,"y":m,"theta":rad,"trust":bool,"goal_dir":rad,
           "n_landmarks":int,"spread_xy":m}
    trust는 위치 분산과 랜드마크 수로 판단한다. 미신뢰 또는 오래된 관측은
    None을 반환해 AIM이 IMU 골대 기억을 사용하도록 한다. 기본값은 비활성이다."""
    def __init__(self, topic="/localization/pose", enabled=False, max_age=0.5):
        self.enabled = enabled and (RosString is not None)
        self.max_age = max_age
        self.pose = None
        self.trust = False
        self.goal_dir = 0.0
        self.stamp = 0.0
        self.lock = threading.Lock()
        if self.enabled:
            rospy.Subscriber(topic, RosString, self._cb, queue_size=1)
            print(f"[Localization] 구독 시작: {topic} (trust 안되면 IMU 기억으로 폴백)")

    def _cb(self, msg):
        try:
            d = json.loads(msg.data)
            with self.lock:
                self.pose = (float(d["x"]), float(d["y"]), float(d["theta"]))
                self.trust = bool(d.get("trust", False))
                self.goal_dir = float(d.get("goal_dir", 0.0))
                self.stamp = time.time()
        except Exception:
            pass  # 잘못된 패킷은 조용히 무시 (폴백 동작)

    def goal_direction(self):
        """신뢰 가능 & 신선하면 로봇기준 골대 각(rad), 아니면 None."""
        if not self.enabled:
            return None
        with self.lock:
            if self.pose is None or not self.trust:
                return None
            if time.time() - self.stamp > self.max_age:
                return None
            return self.goal_dir


class BallFollower:
    def __init__(self, config=None, tuning=None):
        rospy.init_node("ball_follower", anonymous=True)
        self.cfg = config if config else RobotConfig()
        self.tuning = tuning if tuning else TuningParams()

        print("\n" + "="*60)
        print("Ball Follower v10 (safety-hardened)")
        print("- 상태: SEARCH > TRACK > APPROACH > AIM > KICK")
        print("- 지연 보상 조기 정지(perception_latency)")
        print("- 골대: IMU world-angle 기억 상시 유지 + 빠른 AIM")
        print("="*60)

        self.tuning.print_summary()

        self.ARM_IDS = [self.cfg.l_roll_id, self.cfg.r_roll_id,
                       self.cfg.l_pitch_id, self.cfg.r_pitch_id]
        self.HEAD_IDS = [self.cfg.head_yaw_id, self.cfg.head_pitch_id]
        self.ALL_MOTOR_IDS = self.ARM_IDS + self.HEAD_IDS

        self.motor_lock = threading.Lock()
        self.motor_pos = {}
        self.motor_vel = {}            # follow-error/속도 감시용
        self.motor_fb_time = {}        # 관절별 마지막 피드백 시각 (신선도 게이팅)
        self.motor_offsets = {}
        self.prev_arm_cmd = {}
        self.motors_ready = False
        self.vision_ready = False
        self._safe_stop_active = False  # watchdog/예외/종료 시 latch

        self.ball_kalman = BallKalmanFilter()
        self.head_stabilizer = HeadStabilizer(self.cfg.imu_compensation_gain)
        self.head_controller = AdaptiveHeadController(
            self.cfg.head_max_speed, self.cfg.head_smoothing)

        self.goal_memory = GoalMemory(self.tuning)
        self.goal_detector = GoalDetector()

        self.prev_cmd_x = 0.0
        self.prev_cmd_wz = 0.0

        self.prev_roll = 0.0
        self.prev_roll_time = time.time()
        self.filtered_roll_rate = 0.0

        self.scan_timer = 0.0
        self.search_body_timer = 0.0    # body 좌우 회전용 타이머
        self.stop_confirm_count = 0
        self.track_frames = 0
        self.lost_frames = 0
        self.last_valid_dist = 2.0

        self.aim_start_time = 0.0
        self.kick_start_time = 0.0
        self.kick_settling = False          # 킥 전 안정화 중
        self.kick_settle_start = 0.0

        self.ball_tracker = BallTracker(self.cfg.detection_buffer_size)

        # Atomic observation (vision→control 데이터 전달)
        self._ball_obs = BallObservation()
        self._obs_lock = threading.Lock()
        self._vision_frame_counter = 0     # Vision이 obs 갱신할 때마다 증가
        self._last_processed_frame = -1    # Control이 마지막으로 처리한 frame_id

        # 개별 변수 유지 (하위 호환, 내부 로직용)
        self.ball_area = 0.0
        self.ball_dist = 9.99
        self.ball_dist_from_area = 9.99
        self.ball_screen_x = 0.0
        self.ball_screen_y = 0.0
        self.last_ball_dir = 0.0
        self.last_valid_dist = 2.0

        self.lock = threading.Lock()
        self.img_lock = threading.Lock()
        self.ws_lock = threading.Lock()

        self.bridge = CvBridge()
        self.model = YOLO(self.cfg.model_path)

        print(f"YOLO Model: {self.cfg.model_path}")

        self.latest_rgb = None
        self.latest_depth = None
        self.new_frame = False
        self.img_width = 640
        self.img_height = 480

        # 기본 설정은 brain 활성, GameController와 setplay 비활성이다.
        self.use_tronb_brain = bool(getattr(self.cfg, "use_tronb_brain", True))
        self.use_gc = bool(getattr(self.cfg, "use_gc", False))
        self.use_setplay = bool(getattr(self.cfg, "use_setplay", False))

        # 카메라 intrinsics, 높이, pitch는 실측 보정값을 사용한다.
        intr = CameraIntrinsics(
            float(getattr(self.cfg, "cam_fx", 415.69)), float(getattr(self.cfg, "cam_fy", 415.69)),
            float(getattr(self.cfg, "cam_cx", 320.0)), float(getattr(self.cfg, "cam_cy", 240.0)),
            self.img_width, self.img_height)
        geom = RealCameraGeometry(
            intr,
            camera_height=float(getattr(self.cfg, "cam_height_m", 1.27)),
            pitch_down_deg=float(getattr(self.cfg, "cam_pitch_down_deg", 0.0)),
            camera_x=float(getattr(self.cfg, "cam_x_m", 0.0)),
            camera_y=float(getattr(self.cfg, "cam_y_m", 0.0)))
        # YOLO class id↔name (ultralytics 모델 메타) → 랜드마크 타입 매핑에 사용
        names = getattr(self.model, "names", {}) or {}
        self.tronb_realpos = RealPositionEstimator(self.cfg, names=names)
        self.tronb_realpos.geometry = geom
        self.tronb_loc = LocalizationEngine(self.cfg)
        self.tronb_loc.enabled = True
        self.tronb_ball_stop = BallStopEstimator(fov_h=float(getattr(self.cfg, "cam_fov_h", 1.2)))
        self.tronb_setplay = SetPlayVisionAdapter(SetPlayVisionConfig()) if self.use_setplay else None
        self.tronb_brain = RealRobotBrain(setplay=self.tronb_setplay)
        self.tronb_vision_cfg = VisionStateConfig()
        self.tronb_robot_state = {}
        self._tronb_force_stop = False
        self._tronb_suppress_kick = False
        self._tronb_prev_odom_t = None
        # GameController(UDP) 수신 (옵션)
        self.tronb_gc = None
        if self.use_gc and GcReceiver is not None:
            self.tronb_gc = GcReceiver(
                int(getattr(self.cfg, "gc_team_number", 359)),
                int(getattr(self.cfg, "gc_player_number", 1)),
                verbose=False)
            try:
                self.tronb_gc.start()
                print("[TRONB] GameController 수신 시작 (UDP 3838)")
            except Exception as e:
                print(f"[TRONB] GC 수신 시작 실패: {e}")
                self.tronb_gc = None

        self.ws = None
        self.is_connected = False
        self.ws_should_reconnect = True
        self.is_robot_ready = False
        self.stand_done = threading.Event()
        self.walk_done = threading.Event()

        self.state = RobotState.SEARCH
        self.cooldown_start_time = 0.0
        self.recover_start_time = 0.0
        self.current_roll = 0.0
        self.current_pitch_imu = 0.0
        self.current_yaw_imu = 0.0

        self.motor_pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=20)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self.feedback_cb)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_cb, queue_size=1)
        self.debug_pub = rospy.Publisher('/ball_detection/debug_image', Image, queue_size=1)

        self.vision_wd = Watchdog(self.cfg.vision_timeout, "Vision", on_timeout=self._on_watchdog)
        self.control_wd = Watchdog(self.cfg.control_timeout, "Control", on_timeout=self._on_watchdog)

        # 비전 통합 (기본 OFF)
        self.loc_client = LocalizationClient(
            topic=self.cfg.localization_pose_topic,
            enabled=self.cfg.localization_enabled,
            max_age=self.cfg.localization_pose_max_age)
        self.odom_pub = None
        if self.cfg.localization_enabled and RosString is not None:
            self.odom_pub = rospy.Publisher(self.cfg.localization_odom_topic, RosString, queue_size=5)
        self._prev_imu_yaw_odom = 0.0
        self._prev_odom_time = time.time()

        self._start_ws()
        threading.Thread(target=self.control_loop, daemon=True).start()
        threading.Thread(target=self.vision_loop, daemon=True).start()

    def log(self, msg):
        if self.cfg.verbose_log:
            print(f"[LOG] {msg}")

    def get_motor_snapshot(self):
        with self.motor_lock:
            return dict(self.motor_pos)

    def sync_offsets(self, motor_ids=None):
        if motor_ids is None:
            motor_ids = self.ALL_MOTOR_IDS
        snap = self.get_motor_snapshot()
        with self.lock:
            for mid in motor_ids:
                # 피드백 없는 관절의 영점을 0으로 추정하지 않는다.
                # 보정 영점과 하드웨어 영점이 어긋날 수 있으므로 해당 관절은 건너뛴다.
                if mid not in snap:
                    print(f"[sync_offsets] WARN id {mid} 피드백 없음 -> 오프셋 설정 보류")
                    continue
                pos = snap[mid]
                self.motor_offsets[mid] = pos
                self.prev_arm_cmd[f"m_{mid}"] = pos

    def _start_ws(self):
        def run():
            while self.ws_should_reconnect and not rospy.is_shutdown():
                try:
                    self.ws = websocket.WebSocketApp(
                        self.cfg.ws_url,
                        on_open=self.on_open, on_message=self.on_msg,
                        on_error=lambda ws,e: print(f"WS: {e}"),
                        on_close=lambda ws,c,m: self._set_connected(False))
                    self.ws.run_forever()
                except Exception as e:
                    print(f"WS: {e}")
                time.sleep(self.cfg.ws_reconnect_interval)
        threading.Thread(target=run, daemon=True).start()

    def _set_connected(self, val):
        with self.ws_lock:
            self.is_connected = val
        # 연결이 끊기면 twist 이력을 비워 재연결 시 0부터 가속한다.
        # TRON1이 마지막 twist를 유지할 수 있으므로 하체·브리지 watchdog도 확인해야 한다.
        if not val:
            self.prev_cmd_x = 0.0
            self.prev_cmd_wz = 0.0
            print("[WS] disconnected -> twist memory zeroed")

    def set_state(self, new_state):
        with self.lock:
            if self.state != new_state:
                old = self.state
                print(f"{old.name} -> {new_state.name}")
                self.state = new_state

                if new_state == RobotState.APPROACH:
                    self.head_controller.set_approach_mode(True)
                elif new_state in [RobotState.SEARCH, RobotState.AIM]:
                    # SEARCH: head oscillation, AIM: 글랜스 스캔
                    self.head_controller.set_search_mode(True)
                else:
                    self.head_controller.set_approach_mode(False)
                    self.head_controller.set_search_mode(False)

                if new_state == RobotState.TRACK:
                    self.track_frames = 0
                elif new_state == RobotState.SEARCH:
                    self.ball_tracker.reset()
                    self.ball_kalman.reset()
                    self.lost_frames = 0
                    self.scan_timer = 0.0
                    self.search_body_timer = 0.0
                elif new_state == RobotState.AIM:
                    self.aim_start_time = time.time()
                    # APPROACH에서 머리 들면서 쌓인 lost_frames 리셋
                    with self._obs_lock:
                        self._ball_obs.lost_frames = 0
                    self.lost_frames = 0
                elif new_state == RobotState.KICK:
                    self.kick_settling = True
                    self.kick_settle_start = time.time()
                    self.kick_start_time = 0.0  # settle 끝나면 설정
                    with self._obs_lock:
                        self._ball_obs.lost_frames = 0
                    self.lost_frames = 0

    def get_state(self):
        with self.lock:
            return self.state

    def feedback_cb(self, msg):
        now = time.time()
        with self.motor_lock:
            self.motor_pos[msg.id] = msg.position
            self.motor_vel[msg.id] = getattr(msg, 'velocity', 0.0)
            self.motor_fb_time[msg.id] = now                        # 신선도 게이팅용

    def rgb_cb(self, msg):
        try:
            with self.img_lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                h, w = self.latest_rgb.shape[:2]
                self.img_width, self.img_height = w, h
                self.new_frame = True
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[rgb_cb] {e}")

    def depth_cb(self, msg):
        try:
            if msg.encoding == "16UC1":
                img = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) * self.cfg.depth_scale
            elif msg.encoding == "32FC1":
                img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                img = self.bridge.imgmsg_to_cv2(msg, "passthrough").astype(np.float32) * self.cfg.depth_scale
            with self.img_lock:
                self.latest_depth = img
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[depth_cb] {e}")

    def on_open(self, ws):
        print(f"WS Connected")
        self._set_connected(True)
        self.send_ws("request_enable_imu", {"enable": True})

    def on_msg(self, ws, msg):
        try:
            data = json.loads(msg)
            title = data.get('title', '')
            result = data.get('data', {}).get('result', '')
            if title == 'notify_imu':
                imu = data.get('data', {})
                roll, pitch, yaw = 0.0, 0.0, 0.0
                if 'euler' in imu:
                    e = imu['euler']
                    if isinstance(e, list):
                        roll = e[0]
                        pitch = e[1] if len(e) > 1 else 0.0
                        yaw = e[2] if len(e) > 2 else 0.0
                    else:
                        roll = e.get('roll', 0.0)
                        pitch = e.get('pitch', 0.0)
                        yaw = e.get('yaw', 0.0)
                # NaN/inf IMU 값이 들어오면 무시 (전체 제어로 전파 차단)
                if not (math.isfinite(roll) and math.isfinite(pitch) and math.isfinite(yaw)):
                    rospy.logwarn_throttle(2.0, "[IMU] non-finite euler -> dropped")
                    return
                with self.lock:
                    self.current_roll = math.radians(roll)
                    self.current_pitch_imu = math.radians(pitch)
                    self.current_yaw_imu = math.radians(yaw)
                    if not hasattr(self, '_imu_first_logged') or not self._imu_first_logged:
                        print(f"[IMU 초기] yaw={yaw:.1f}° ({math.radians(yaw):.3f} rad)")
                        self._imu_first_logged = True
            elif title == 'notify_stand_mode' and result == 'success':
                self.stand_done.set()
            elif title == 'notify_walk_mode' and result == 'success':
                self.walk_done.set()
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[on_msg] {e}")

    def send_ws(self, title, data):
        with self.ws_lock:
            if self.is_connected and self.ws:
                try:
                    self.ws.send(json.dumps({
                        "accid": self.cfg.robot_sn, "title": title,
                        "timestamp": int(time.time()*1000),
                        "guid": str(uuid.uuid4()).replace("-", ""),
                        "data": data
                    }))
                except Exception as e:
                    rospy.logwarn_throttle(3.0, f"[send_ws] {title}: {e}")

    def get_depth(self, depth_img, cx, cy):
        if depth_img is None: return None
        h, w = depth_img.shape
        r = self.cfg.depth_sample_radius
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        if x2 <= x1 or y2 <= y1: return None
        region = depth_img[y1:y2, x1:x2]
        valid = region[(region > self.cfg.depth_min_valid) &
                       (region < self.cfg.depth_max_valid) & ~np.isnan(region)]
        return float(np.median(valid)) if len(valid) >= 3 else None

    def estimate_distance_from_area(self, area: float) -> float:
        if area <= 0: return 9.99
        ball_pixel_diameter = math.sqrt(area * 4 / math.pi)
        if ball_pixel_diameter > 0:
            dist = (self.cfg.camera_focal_length * self.cfg.ball_real_diameter) / ball_pixel_diameter
            return max(0.1, min(5.0, dist))
        return 9.99

    def find_ball_candidates(self, boxes) -> List[BallCandidate]:
        candidates = []
        for box in boxes:
            if self.cfg.use_class_filter:
                cls = int(box.cls.cpu().numpy()[0])
                if cls != self.cfg.ball_class_id: continue
            bx, by, bw, bh = box.xywh.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            area = bw * bh
            aspect_ratio = bw / bh if bh > 0 else 0
            circularity = 1.0 - abs(1.0 - aspect_ratio)
            candidates.append(BallCandidate(bx, by, bw, bh, area, conf, aspect_ratio, circularity))
        candidates.sort(key=lambda c: c.conf * c.circularity, reverse=True)
        return candidates

    def find_goal_in_results(self, boxes, img_w, img_h) -> Optional[dict]:
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            if cls != self.cfg.goal_class_id:
                continue
            conf = box.conf.cpu().numpy()[0]
            if conf < self.cfg.goal_conf_weak:
                continue
            bx, by, bw, bh = box.xywh.cpu().numpy()[0]
            area = bw * bh
            if area < self.cfg.goal_min_area:
                continue
            norm_x = (bx - img_w/2) / (img_w/2)
            norm_y = (by - img_h/2) / (img_h/2)
            return {
                'norm_x': norm_x, 'norm_y': norm_y, 'area': area,
                'width': bw, 'height': bh, 'conf': conf,
                'cx': int(bx), 'cy': int(by)
            }
        return None

    def screen_to_angle(self, screen_x, screen_y):
        angle_x = screen_x * (self.cfg.camera_fov_h / 2)
        angle_y = screen_y * (self.cfg.camera_fov_v / 2)
        yaw_delta = -angle_x if not self.cfg.head_yaw_inverted else angle_x
        pitch_delta = -angle_y if not self.cfg.head_pitch_inverted else angle_y
        return yaw_delta, pitch_delta

    # 영상 관측으로 위치 추정과 robot_state를 갱신한다.
    # ball_follower YOLO 결과(boxes)에서 랜드마크 추출 → PF → robot_state(brain 입력).
    def _tronb_perceive(self, boxes, depth, ball_obs, goal_data):
        # 검출 클래스와 투영 기하를 적용해 랜드마크 관측을 만든다.
        try:
            lm_obs = self.tronb_realpos.landmark_observations(boxes, depth)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[TRONB] landmark_observations 실패: {e}")
            lm_obs = []
        # 직전 명령 twist의 적분을 위치 추정기의 예측 입력으로 사용한다.
        now = time.time()
        dt = 0.0 if self._tronb_prev_odom_t is None else max(0.0, now - self._tronb_prev_odom_t)
        self._tronb_prev_odom_t = now
        vx = max(0.0, getattr(self, "prev_cmd_x", 0.0)); wz = getattr(self, "prev_cmd_wz", 0.0)
        odom = (vx * dt, 0.0, wz * dt)
        self.tronb_loc.update(lm_obs, odom)
        loc_trust = self.tronb_loc.trust()
        # 공의 로봇 기준 좌표가 없으면 거리와 각도로 근사한다.
        ball_xy = None
        if ball_obs.detected and ball_obs.dist < 9.0:
            yaw_d, _ = self.screen_to_angle(ball_obs.screen_x, 0.0)
            ball_xy = (ball_obs.dist * math.cos(yaw_d), -ball_obs.dist * math.sin(yaw_d))
        motion = self.tronb_ball_stop.update(ball_obs)
        gcs = restart_type = restart_owner = None
        if self.tronb_gc is not None:
            snap = self.tronb_gc.snapshot()
            gcs = snap.get("game_controller_state")
            restart_type = snap.get("restart_type")
            restart_owner = snap.get("restart_owner")
        goal_obs = make_goal_obs(
            detected=bool(goal_data is not None),
            screen_x=float(goal_data["norm_x"]) if goal_data else 0.0,
            conf=float(goal_data["conf"]) if goal_data else 0.0)
        rs = build_robot_state(
            ball=ball_obs, goal=goal_obs,
            loc_pose=self.tronb_loc.pose, loc_trust=loc_trust, ball_xy_robot=ball_xy,
            field=self.tronb_loc.fm, motion=motion, vision_cfg=self.tronb_vision_cfg,
            game_controller_state=gcs, restart_type=restart_type, restart_owner=restart_owner,
            setplay_adapter=self.tronb_setplay)
        self.tronb_robot_state = rs
        return rs

    def get_approach_speed(self, dist):
        if dist > self.cfg.approach_dist_far:
            return self.tuning.approach_speed_far
        elif dist > self.cfg.approach_dist_near:
            r = (dist - self.cfg.approach_dist_near) / (self.cfg.approach_dist_far - self.cfg.approach_dist_near)
            return self.tuning.approach_speed_near + r * (self.tuning.approach_speed_far - self.tuning.approach_speed_near)
        return self.tuning.approach_speed_near

    def get_approach_pitch(self, dist) -> float:
        if dist >= self.cfg.approach_pitch_dist_far:
            return self.cfg.approach_pitch_far
        elif dist <= self.cfg.approach_pitch_dist_near:
            return self.cfg.approach_pitch_near
        else:
            r = (dist - self.cfg.approach_pitch_dist_near) / \
                (self.cfg.approach_pitch_dist_far - self.cfg.approach_pitch_dist_near)
            return self.cfg.approach_pitch_near + r * (self.cfg.approach_pitch_far - self.cfg.approach_pitch_near)

    def vision_loop(self):
        rate = rospy.Rate(30)
        while not self.vision_ready and not rospy.is_shutdown():
            self.vision_wd.feed()  # 초기화 대기 중에도 watchdog feed
            rate.sleep()

        print("Vision Loop Active (v9)")
        self.vision_wd.arm()  # 부팅 완료 후에만 stuck 감시
        last_goal_log = 0.0

        while not rospy.is_shutdown():
            try:
                self.vision_wd.feed()
                curr = self.get_state()

                if curr in [RobotState.RECOVER, RobotState.COOLDOWN]:
                    rate.sleep()
                    continue

                with self.lock:
                    imu_yaw = self.current_yaw_imu

                self.goal_memory.update_confidence_decay(current_imu_yaw=imu_yaw)

                if self.ball_kalman.initialized:
                    self.ball_kalman.get_predicted_position(self.cfg.kalman_predict_time)

                # 상태별 head 제어 (AIM은 control_loop에서 직접 제어 -> 여기선 패스)
                if curr == RobotState.SEARCH:
                    pass

                elif curr == RobotState.APPROACH:
                    # pitch만 거리 기반 설정, yaw는 아래 ball tracking에서 처리
                    with self.lock:
                        target_pitch = self.get_approach_pitch(self.ball_dist)
                        self.head_controller.target_pitch = target_pitch

                elif curr == RobotState.AIM:
                    pass

                frame, depth = None, None
                with self.img_lock:
                    if self.new_frame:
                        frame = self.latest_rgb.copy()
                        depth = self.latest_depth.copy() if self.latest_depth is not None else None
                        self.new_frame = False

                if frame is None:
                    rate.sleep()
                    continue

                results = self.model.predict(frame, conf=self.cfg.goal_conf_weak, verbose=False)
                h, w = frame.shape[:2]
                debug = frame.copy()

                # 골대 감지 -> world-angle 기억 갱신
                # SEARCH/TRACK/AIM 어디서든 골대가 보이면 기억을 채워둔다(상시 유지).
                goal_data = self.find_goal_in_results(results[0].boxes, w, h)

                if curr in [RobotState.SEARCH, RobotState.TRACK, RobotState.AIM]:
                    if goal_data:
                        self.goal_detector.update(
                            detected=True,
                            screen_x=goal_data['norm_x'],
                            area=goal_data['area'],
                            width=goal_data['width'],
                            conf=goal_data['conf']
                        )
                        # 2프레임만 연속 잡혀도 기억 갱신(빠른 획득)
                        if self.goal_detector.consecutive_detect >= 2:
                            self.goal_memory.update_from_detection(
                                goal_screen_x=goal_data['norm_x'],
                                head_yaw=self.head_controller.filtered_yaw,
                                imu_yaw=imu_yaw,
                                area=goal_data['area'],
                                conf=goal_data['conf']
                            )
                        g = goal_data
                        half_w = int(g['width'] / 2)
                        half_h = int(g['height'] / 2)
                        cv2.rectangle(debug,
                                     (g['cx'] - half_w, g['cy'] - half_h),
                                     (g['cx'] + half_w, g['cy'] + half_h),
                                     (0, 255, 255), 2)
                    else:
                        self.goal_detector.update(detected=False)

                if time.time() - last_goal_log > 1.0:
                    mem_status = self.goal_memory.get_status_string(imu_yaw)
                    print(f"[GOAL] {curr.name}: {mem_status}")
                    last_goal_log = time.time()

                candidates = self.find_ball_candidates(results[0].boxes)
                best_candidate = None

                for candidate in candidates:
                    valid, reason = self.ball_tracker.validate_ball(candidate, self.tuning, self.cfg)
                    if valid:
                        best_candidate = candidate
                        break

                detected, detect_reason = self.ball_tracker.update(best_candidate, self.tuning, self.cfg)

                if detected and best_candidate:
                    bx, by = best_candidate.x, best_candidate.y
                    area = best_candidate.area

                    depth_dist = self.get_depth(depth, int(bx), int(by)) if depth is not None else None
                    has_depth = (depth_dist is not None and
                                self.cfg.depth_min_valid < depth_dist < self.cfg.depth_max_valid)

                    accepted = self.ball_kalman.update(np.array([bx, by]), use_depth=has_depth)
                    if not accepted:
                        rospy.loginfo_throttle(2.0,
                            f"[Vision] Kalman rejected measurement (area={area:.0f})")

                    pred_x, pred_y = self.ball_kalman.get_predicted_position(self.cfg.kalman_predict_time)

                    norm_x = (pred_x - w/2) / (w/2)
                    norm_y = (pred_y - h/2) / (h/2)

                    dist_from_area = self.estimate_distance_from_area(area)

                    if has_depth:
                        final_dist = depth_dist
                    else:
                        final_dist = dist_from_area

                    last_valid = final_dist if final_dist < 2.5 else self.last_valid_dist

                    # Atomic observation 작성 (감지 메타데이터 포함)
                    obs = BallObservation(
                        area=area,
                        dist=final_dist,
                        dist_from_area=dist_from_area,
                        screen_x=norm_x,
                        screen_y=norm_y,
                        last_dir=np.sign(norm_x) if abs(norm_x) > 0.1 else self.last_ball_dir,
                        lost_frames=0,
                        last_valid_dist=last_valid,
                        timestamp=time.time(),
                        detected=True,
                        consecutive_detect=self.ball_tracker.consecutive_detect,
                        kalman_accepted=accepted,
                    )
                    self._vision_frame_counter += 1
                    obs.frame_id = self._vision_frame_counter
                    with self._obs_lock:
                        self._ball_obs = obs

                    # 개별 변수도 업데이트 (하위 호환)
                    with self.lock:
                        self.ball_area = area
                        self.ball_screen_x = norm_x
                        self.ball_screen_y = norm_y
                        self.last_ball_dir = obs.last_dir
                        self.lost_frames = 0
                        self.ball_dist_from_area = dist_from_area
                        self.ball_dist = final_dist
                        self.last_valid_dist = last_valid

                    yaw_delta, pitch_delta = self.screen_to_angle(norm_x, norm_y)

                    with self.lock:
                        curr_yaw, curr_pitch = self.head_controller.get_command()

                        if curr == RobotState.TRACK:
                            new_yaw = np.clip(curr_yaw + yaw_delta * 0.3, *self.cfg.head_yaw_limit)
                            new_pitch = np.clip(curr_pitch + pitch_delta * 0.3, *self.cfg.head_pitch_limit)
                            self.head_controller.set_target(new_yaw, new_pitch)
                        elif curr == RobotState.APPROACH:
                            # yaw만 ball tracking, pitch는 state-based에서 거리 기반 설정
                            new_yaw = np.clip(curr_yaw + yaw_delta * 0.3, *self.cfg.head_yaw_limit)
                            self.head_controller.target_yaw = new_yaw

                    sz = int(math.sqrt(area))
                    cv2.rectangle(debug, (int(bx-sz/2), int(by-sz/2)), (int(bx+sz/2), int(by+sz/2)), (0,255,0), 3)
                    cv2.circle(debug, (int(pred_x), int(pred_y)), 10, (255,0,255), 2)
                    cv2.putText(debug, f"A:{area:.0f} D:{self.ball_dist:.2f}",
                               (int(bx-sz/2), int(by-sz/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    # Vision은 데이터만 갱신. 상태 전이 판단은 Control Loop에서.

                else:
                    if self.ball_kalman.initialized:
                        pred_x, pred_y = self.ball_kalman.predict()
                        if 0 < pred_x < w and 0 < pred_y < h:
                            norm_x = (pred_x - w/2) / (w/2)
                            with self.lock:
                                self.ball_screen_x = norm_x

                    with self.lock:
                        self.lost_frames += 1
                        lost = self.lost_frames

                    self._vision_frame_counter += 1
                    with self._obs_lock:
                        self._ball_obs.lost_frames = lost
                        self._ball_obs.detected = False
                        self._ball_obs.consecutive_detect = 0
                        self._ball_obs.frame_id = self._vision_frame_counter

                # 검출 결과로 위치 추정과 brain 입력을 갱신한다.
                if self.use_tronb_brain:
                    with self._obs_lock:
                        _bo = self._ball_obs
                    _tb_ball = make_ball_obs(
                        detected=_bo.detected, dist=_bo.dist, screen_x=_bo.screen_x, screen_y=_bo.screen_y,
                        lost_frames=_bo.lost_frames, consecutive_detect=_bo.consecutive_detect,
                        last_valid_dist=_bo.last_valid_dist, frame_id=_bo.frame_id)
                    try:
                        self._tronb_perceive(results[0].boxes, depth, _tb_ball, goal_data)
                    except Exception as e:
                        rospy.logwarn_throttle(5.0, f"[TRONB] perceive 실패: {e}")

                # Debug overlay (get_command 호출 없이 값만 읽기 - EMA 진행 방지)
                yaw_dbg = self.head_controller.filtered_yaw
                pitch_dbg = self.head_controller.filtered_pitch

                cv2.putText(debug, f"{curr.name} Y:{yaw_dbg:.2f} P:{pitch_dbg:.2f}",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(debug, f"Ball: x={self.ball_screen_x:.2f} d={self.ball_dist:.2f}m",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

                if self.goal_memory.is_valid():
                    mem_dir = self.goal_memory.get_goal_direction(imu_yaw)
                    cv2.putText(debug, f"Goal: {math.degrees(mem_dir):.0f}deg",
                               (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                try:
                    self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))
                except Exception as e:
                    rospy.loginfo_throttle(10.0, f"[debug_pub] {e}")

                rate.sleep()
            except Exception as e:
                print(f"Vision: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

    def control_loop(self):
        rate = rospy.Rate(50)
        print(f"[부팅] 모터 대기...")

        while not rospy.is_shutdown():
            self.control_wd.feed()  # 초기화 중에도 watchdog feed
            snap = self.get_motor_snapshot()
            found = [m for m in self.ALL_MOTOR_IDS if m in snap]
            if len(found) == len(self.ALL_MOTOR_IDS):
                print(f"모터 OK ({len(found)}/{len(self.ALL_MOTOR_IDS)})")
                break
            for m in self.ALL_MOTOR_IDS:
                if m not in snap:
                    self.send_motor_raw(m, 0, 0)
            time.sleep(0.1)

        print("[부팅] WS 대기...")
        while not self.is_connected and not rospy.is_shutdown():
            self.control_wd.feed()
            time.sleep(1.0)

        self.control_wd.feed()
        self.sync_offsets()

        print("Stand Mode...")
        self.send_ws("request_stand_mode", {})
        t0 = time.time()
        while not self.stand_done.is_set():
            self.control_wd.feed()
            if rospy.is_shutdown(): return
            if time.time() - t0 > 10:
                print("Stand 재요청...")
                self.send_ws("request_stand_mode", {})
                t0 = time.time()
            for mid in self.ARM_IDS:
                self.send_motor_raw(mid, 0, 0)
            rate.sleep()
        print("Stand OK")
        time.sleep(1.5)
        self.sync_offsets()

        print(f"Head 정렬...")
        start_p = self.motor_offsets.get(self.cfg.head_pitch_id, 0)
        target_p = start_p + self.cfg.head_pitch_offset_target
        for i in range(self.cfg.head_init_steps):
            self.control_wd.feed()
            t = (i + 1) / self.cfg.head_init_steps
            p = start_p + (target_p - start_p) * t
            self.send_motor_raw(self.cfg.head_pitch_id, p, self.cfg.head_init_kp)
            rate.sleep()

        time.sleep(0.8)
        self.sync_offsets(self.HEAD_IDS)

        print("Walk Mode...")
        self.send_ws("request_walk_mode", {})
        if not self.walk_done.wait(timeout=10):
            print("Walk timeout")
        else:
            print("Walk OK")
        self.control_wd.feed()
        time.sleep(1.5)
        self.sync_offsets(self.ARM_IDS)

        self.head_controller.reset(0.0, self.cfg.scan_pitch_center)
        time.sleep(0.5)

        self.is_robot_ready = True
        self.motors_ready = True
        self.vision_ready = True

        self.prev_roll = self.current_roll
        self.prev_roll_time = time.time()
        start_time = time.time()
        print("Control Loop Started (v10)")
        if self.cfg.reverse_gait:
            print(f"[REVERSE GAIT] ON  (역관절 후진)  reverse_max_speed={self.cfg.reverse_max_speed:.3f}  "
                  f"reverse_kick_speed={self.cfg.reverse_kick_speed:.3f} m/s  "
                  f"-> TRON1 로 보내는 x 부호 반전. 캡을 조금씩 올려 TRON1 후진 리밋 찾기.")
        else:
            print("[REVERSE GAIT] OFF (전진 보행)")
        self.control_wd.arm()  # 부팅 완료 후에만 stuck 감시

        while not rospy.is_shutdown():
          try:
            self.control_wd.feed()

            # SAFE STOP (watchdog/예외 latch). 정지 + 상체 현재위치 compliant 유지.
            if self._safe_stop_active:
                self.send_twist_v2(0.0, 0.0)
                snap = self.get_motor_snapshot()
                for mid in self.ALL_MOTOR_IDS:
                    if mid in snap:
                        self.send_motor_raw(mid, snap[mid], self.cfg.follow_err_kp_backoff)
                rate.sleep()
                continue

            if not self.is_robot_ready:
                rate.sleep()
                continue

            with self.lock:
                body_roll = self.current_roll
                body_pitch = self.current_pitch_imu
                imu_yaw = self.current_yaw_imu

            yaw_comp, pitch_comp = self.head_stabilizer.update(body_roll, body_pitch)
            # state machine 전에는 filtered 값만 읽기 (EMA 진행 없음)
            # get_command()는 state machine 후에 호출 → set_target_direct 반영됨
            yaw_cmd = self.head_controller.filtered_yaw
            pitch_cmd = self.head_controller.filtered_pitch

            # Atomic observation 읽기
            with self._obs_lock:
                obs = self._ball_obs

            # 새 프레임인지 확인 (50Hz control vs 30Hz vision 동기화)
            is_new_frame = (obs.frame_id != self._last_processed_frame)

            with self.lock:
                curr = self.state

            area = obs.area
            dist = obs.dist
            dist_from_area = obs.dist_from_area
            screen_x = obs.screen_x
            last_dir = obs.last_dir
            last_dist = obs.last_valid_dist
            lost = obs.lost_frames

            # 지각→제어 지연 보상
            # obs 나이(obs_age) + 파이프라인 지연만큼 로봇이 전진했다고 보고 거리를 당겨줌
            # 관측이 오래되거나 이동 속도가 빠르면 더 일찍 정지한다.
            obs_age = max(0.0, time.time() - obs.timestamp)
            lat = obs_age + self.tuning.perception_latency
            dist_comp = max(0.05, dist - max(0.0, self.prev_cmd_x) * lat)

            # 상태 전이 (새 프레임에서만, 중복 방지)
            # use_tronb_brain이 활성화되면 RealRobotBrain이 상태 전이를 결정한다.
            if is_new_frame:
                self._last_processed_frame = obs.frame_id

                if self.use_tronb_brain and self.tronb_robot_state:
                    # 넘어짐은 ball_follower fall guard(아래)가 별도 처리하므로 brain 엔 is_fallen=False.
                    dec = self.tronb_brain.decide(self.tronb_robot_state, lower={"is_fallen": False})
                    self._tronb_force_stop = dec.force_stop
                    self._tronb_suppress_kick = dec.suppress_kick
                    try:
                        mapped = RobotState[dec.state]   # brain 의 state 이름 → ball_follower RobotState
                    except KeyError:
                        mapped = RobotState.SEARCH
                    if mapped != curr and curr not in (RobotState.RECOVER, RobotState.COOLDOWN):
                        self.set_state(mapped)
                else:
                    # 원본 ball_follower 전이 (use_tronb_brain=False)
                    self._tronb_force_stop = False
                    self._tronb_suppress_kick = False
                    if obs.detected and obs.consecutive_detect >= self.cfg.min_consecutive_detect:
                        if curr == RobotState.SEARCH:
                            self.set_state(RobotState.TRACK)
                    if not obs.detected and obs.lost_frames > self.tuning.lost_frames_threshold:
                        if curr in [RobotState.TRACK, RobotState.APPROACH]:
                            if curr == RobotState.APPROACH and last_dist < self.tuning.stop_dist:
                                rospy.loginfo(f"[Control] 공 놓침 (가까움 {last_dist:.2f}m) -> AIM")
                                self.set_state(RobotState.AIM)
                            else:
                                rospy.loginfo(f"[Control] 공 놓침 (lost={obs.lost_frames}) -> SEARCH")
                                self.set_state(RobotState.SEARCH)

            # 상태가 바뀌었을 수 있으므로 다시 읽기
            with self.lock:
                curr = self.state

            # Fall guard (모든 상태 공통). roll/pitch fall 한계 넘으면 즉시 정지 + RECOVER.
            if (math.isfinite(body_roll) and math.isfinite(body_pitch) and
                    (abs(body_roll) > self.cfg.roll_fall_limit or
                     abs(body_pitch) > self.cfg.pitch_fall_limit)):
                if curr not in (RobotState.RECOVER, RobotState.COOLDOWN):
                    print(f"[FALL] roll={body_roll:.2f} pitch={body_pitch:.2f} -> RECOVER (정지)")
                    self.recover_start_time = time.time()
                    self.set_state(RobotState.RECOVER)
                    with self.lock:
                        curr = self.state

            t = time.time() - start_time
            cvx, cwz = 0.0, 0.0
            turn_dir = -1 if self.cfg.body_turn_inverted else 1

            if curr == RobotState.SEARCH:
                self.stop_confirm_count = 0

                # 전진 drift 보상 (TRON1 walk mode 하드웨어 drift)
                cvx = self.tuning.search_drift_compensation

                # body 좌우 sin 회전 (한쪽으로만 안 돌게)
                self.search_body_timer += 0.02
                phase = (2.0 * math.pi / self.tuning.search_turn_period) * self.search_body_timer
                cwz = self.tuning.search_turn_speed * math.sin(phase) + self.tuning.search_turn_bias

                # head oscillation (50Hz, goal_memory 유무 관계없이 항상)
                current_time = time.time()
                target_pitch = self.tuning.search_head_pitch_center + \
                               self.cfg.scan_pitch_amplitude * math.sin(current_time * 2.0)
                self.head_controller.set_target_direct(0.0, target_pitch)

            elif curr == RobotState.TRACK:
                self.stop_confirm_count = 0
                with self.lock:
                    self.track_frames += 1
                    tf = self.track_frames

                if lost > 0 and lost < self.cfg.lost_ball_search_frames:
                    cwz = last_dir * self.cfg.lost_ball_turn_speed * turn_dir
                else:
                    if abs(yaw_cmd) > self.cfg.body_turn_deadzone:
                        cwz = np.clip(-turn_dir * yaw_cmd * self.cfg.body_turn_gain, -0.35, 0.35)
                        with self.lock:
                            curr_yaw, curr_pitch = self.head_controller.get_command()
                            compensation = cwz * self.cfg.head_body_compensation
                            new_yaw = np.clip(curr_yaw + compensation, *self.cfg.head_yaw_limit)
                            self.head_controller.set_target(new_yaw, self.head_controller.target_pitch)

                    yaw_ok = abs(yaw_cmd) < self.cfg.track_to_approach_yaw
                    frames_ok = tf >= self.cfg.track_min_frames
                    dist_ok = dist < self.cfg.track_min_dist

                    if yaw_ok and frames_ok and dist_ok:
                        self.set_state(RobotState.APPROACH)
            elif curr == RobotState.APPROACH:
                if lost > self.cfg.lost_ball_search_frames:
                    cwz = last_dir * self.cfg.lost_ball_turn_speed * turn_dir
                else:
                    if abs(yaw_cmd) > self.cfg.approach_to_track_yaw:
                        self.set_state(RobotState.TRACK)
                    else:
                        cvx = self.get_approach_speed(dist)
                        base_turn = -turn_dir * yaw_cmd * 0.70
                        drift_correction = self.tuning.approach_drift_correction

                        # 곡선 주행
                        curve_turn = 0.0
                        if self.tuning.curved_approach_enabled and dist < self.tuning.curved_approach_dist:
                            gdir, gvalid = self._goal_direction(imu_yaw)   # loc 우선, 폴백 IMU 기억
                            if gvalid:
                                curve_turn = -np.sign(gdir) * self.tuning.curved_approach_gain

                        cwz = base_turn + drift_correction + curve_turn

                        with self.lock:
                            curr_yaw, curr_pitch = self.head_controller.get_command()
                            comp = cwz * self.cfg.head_body_compensation * 0.3
                            new_yaw = np.clip(curr_yaw + comp, *self.cfg.head_yaw_limit)
                            self.head_controller.set_target(new_yaw, self.head_controller.target_pitch)

                        # 보상 거리 기반 조기 정지 (공이 시야에 있을 때 멈춰서 AIM)
                        ball_aligned = abs(yaw_cmd) < self.cfg.kick_align_threshold
                        if dist_comp < self.tuning.stop_dist and ball_aligned:
                            self.stop_confirm_count += 1
                        else:
                            self.stop_confirm_count = max(0, self.stop_confirm_count - 1)

                        if self.stop_confirm_count >= self.tuning.stop_confirm_frames:
                            self.stop_confirm_count = 0
                            print(f"[AIM] 진입 (dist_comp={dist_comp:.2f}m, age={obs_age*1000:.0f}ms)")
                            self.set_state(RobotState.AIM)

            elif curr == RobotState.AIM:
                # 골대 기억, 제한된 스캔 시간, 타임아웃 순으로 킥을 판단한다.
                # - 골대 기억 있으면: 바로 정렬/킥
                # - 기억 없으면: aim_glance_time 동안만 고개 들어 스캔 -> 그래도 없으면 직진 킥
                # aim_timeout 뒤에는 정렬 여부와 관계없이 킥한다.
                elapsed = time.time() - self.aim_start_time
                cvx = self.tuning.search_drift_compensation   # 제자리(앞밀림만 상쇄)
                cwz = 0.0

                # localization 신뢰 시 그 골방향, 아니면 IMU 기억 (기본 OFF면 항상 IMU 기억)
                aim_goal_dir, have_goal = self._goal_direction(imu_yaw)

                if elapsed < self.tuning.aim_settle_time:
                    pass
                elif not have_goal and elapsed < self.tuning.aim_settle_time + self.tuning.aim_glance_time:
                    # 골대 기억 없음 -> 짧게 고개 들어 스캔 (vision_loop가 잡으면 기억 갱신됨)
                    scan_yaw = 0.5 * math.sin(elapsed * 3.0)
                    self.head_controller.set_target_direct(scan_yaw, self.tuning.aim_look_pitch)
                else:
                    goal_dir = aim_goal_dir if have_goal else 0.0
                    aligned = abs(goal_dir) < self.tuning.align_good_angle

                    if aligned or elapsed > self.tuning.aim_timeout:
                        src = "정렬" if aligned else "타임아웃"
                        print(f"[AIM] {src} -> KICK (goal_dir={math.degrees(goal_dir):.1f}deg, valid={have_goal})")
                        self.set_state(RobotState.KICK)
                    else:
                        # 제자리에서 골대쪽으로 소폭 회전, 머리는 공쪽 유지
                        cwz = np.clip(-turn_dir * goal_dir * (self.tuning.align_turn_speed / 0.25),
                                     -self.tuning.align_turn_speed, self.tuning.align_turn_speed)
                        self.head_controller.set_target(0.0, self.cfg.approach_pitch_near)

            elif curr == RobotState.KICK:
                if self.kick_settling:
                    settle_elapsed = time.time() - self.kick_settle_start
                    if settle_elapsed < self.tuning.kick_settle_time:
                        cvx, cwz = 0.0, 0.0
                    else:
                        self.kick_settling = False
                        self.kick_start_time = time.time()
                        print("KICK!")
                else:
                    self.head_controller.set_target(0.0, -0.30)
                    cvx = self.tuning.kick_speed
                    cwz = self.tuning.straight_drift_correction

                    if abs(self.current_roll) > self.cfg.roll_warning_limit:
                        cvx *= 0.7
                    if abs(self.current_roll) > self.cfg.roll_fall_limit:
                        self.recover_start_time = time.time()
                        self.set_state(RobotState.RECOVER)
                    elif time.time() - self.kick_start_time > self.cfg.kick_duration:
                        self.cooldown_start_time = time.time()
                        self.set_state(RobotState.COOLDOWN)

            elif curr == RobotState.COOLDOWN:
                cvx = -0.02
                self.head_controller.set_target(0.0, self.cfg.scan_pitch_center)
                if time.time() - self.cooldown_start_time > self.cfg.cooldown_time:
                    self.set_state(RobotState.SEARCH)

            elif curr == RobotState.RECOVER:
                self.head_controller.set_target(0.0, 0.0)
                if time.time() - self.recover_start_time > self.cfg.recover_time:
                    self.set_state(RobotState.SEARCH)

            # PENALIZED, 이동 불가, setplay 대기에서는 정지하고 킥 억제를 적용한다.
            # 자세 한계 초과에 따른 RECOVER가 우선한다.
            _kick = (curr == RobotState.KICK and not self.kick_settling)
            if self.use_tronb_brain:
                if self._tronb_force_stop:
                    cvx, cwz = 0.0, 0.0
                if self._tronb_suppress_kick:
                    _kick = False
            self.send_twist_v2(cvx, cwz, is_kick=_kick)

            # 상태 전이 뒤 명령을 읽어 이번 틱의 set_target_direct를 반영한다.
            yaw_cmd_final, pitch_cmd_final = self.head_controller.get_command()
            yaw_final = np.clip(yaw_cmd_final + yaw_comp, *self.cfg.head_yaw_limit)
            pitch_final = np.clip(pitch_cmd_final + pitch_comp, *self.cfg.head_pitch_limit)

            self.send_motor_offset(self.cfg.head_yaw_id, yaw_final, 10)
            self.send_motor_offset(self.cfg.head_pitch_id, pitch_final, 10)

            now = time.time()
            dt = max(now - self.prev_roll_time, 1e-3)
            raw_rate = (self.current_roll - self.prev_roll) / dt
            self.filtered_roll_rate = 0.15 * raw_rate + 0.85 * self.filtered_roll_rate
            self.prev_roll = self.current_roll
            self.prev_roll_time = now

            bal = 0.0
            if abs(self.current_roll) > 0.05:
                bal = -np.clip((0.8 * self.current_roll + 0.08 * self.filtered_roll_rate) * 0.4, -0.12, 0.12)
            if curr == RobotState.KICK:
                bal *= self.cfg.kick_zmp_factor

            swing = 0.0
            base_roll = self.cfg.arm_base_roll
            kp = 15.0

            if curr not in [RobotState.AIM, RobotState.KICK, RobotState.RECOVER,
                           RobotState.COOLDOWN]:
                total = np.clip(self.cfg.arm_swing_base + (abs(cvx) + abs(cwz)) * self.cfg.arm_swing_speed_gain, 0, 0.4)
                swing = total * math.sin(t * self.cfg.arm_swing_freq)

            if curr == RobotState.KICK and not self.kick_settling:
                swing = 0.70
                base_roll = 0.28

            left_roll = np.clip(-base_roll + bal, -0.35, 0.35)
            right_roll = np.clip(base_roll - bal, -0.35, 0.35)

            if self.cfg.arm_pitch_inverted:
                self.send_motor_offset(self.cfg.l_pitch_id, swing, kp)
                self.send_motor_offset(self.cfg.r_pitch_id, swing, kp)
            else:
                self.send_motor_offset(self.cfg.l_pitch_id, swing, kp)
                self.send_motor_offset(self.cfg.r_pitch_id, -swing, kp)

            self.send_motor_offset(self.cfg.l_roll_id, left_roll, kp)
            self.send_motor_offset(self.cfg.r_roll_id, right_roll, kp)

            self._publish_odom(cvx, cwz, imu_yaw)   # localization predict 입력 (enabled일 때만)

            rate.sleep()
          except Exception as e:
            self.send_twist_v2(0.0, 0.0)   # 예외 시 우선 정지 (0.1s 동안 마지막 twist 유지 방지)
            print(f"Control: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    def _rel_limit(self, mid):
        """오프셋 오류에 대비한 관절별 상대 지령 절대한계."""
        if mid == self.cfg.head_yaw_id:   return self.cfg.rel_limit_head_yaw
        if mid == self.cfg.head_pitch_id: return self.cfg.rel_limit_head_pitch
        if mid in (self.cfg.l_roll_id, self.cfg.r_roll_id):   return self.cfg.rel_limit_arm_roll
        if mid in (self.cfg.l_pitch_id, self.cfg.r_pitch_id): return self.cfg.rel_limit_arm_pitch
        return 0.5

    def send_motor_offset(self, mid, rel, kp=15):
        # 지령은 유한값, 관절 한계, 변화율, 피드백 신선도, 추종 오차 순으로 검사한다.
        if rel is None or not np.isfinite(rel):
            return
        rlim = self._rel_limit(mid)
        rel = float(np.clip(rel, -rlim, rlim))
        with self.lock:
            if mid not in self.motor_offsets:
                return
            key = f"m_{mid}"
            prev = self.prev_arm_cmd.get(key, self.motor_offsets[mid])
            tgt = self.motor_offsets[mid] + rel
            limit = self.cfg.arm_rate_limit_kick if self.state == RobotState.KICK else self.cfg.arm_rate_limit_normal
            safe = prev + float(np.clip(tgt - prev, -limit, limit))

        now = time.time()
        with self.motor_lock:
            fb_t = self.motor_fb_time.get(mid, 0.0)
            actual = self.motor_pos.get(mid, safe)

        # 피드백이 오래되면 현재 위치에서 낮은 Kp를 사용한다.
        if now - fb_t > self.cfg.motor_fb_timeout:
            with self.lock:
                self.prev_arm_cmd[key] = actual
            self.send_motor_raw(mid, actual, self.cfg.follow_err_kp_backoff)
            rospy.logwarn_throttle(2.0, f"[motor {mid}] feedback stale -> compliant")
            return

        # 추종 오차가 한계를 넘으면 지령 증가를 멈추고 Kp를 낮춘다.
        if abs(safe - actual) > self.cfg.follow_err_limit:
            with self.lock:
                self.prev_arm_cmd[key] = prev   # advance 안 함 (계속 밀지 않음)
            self.send_motor_raw(mid, prev, self.cfg.follow_err_kp_backoff)
            rospy.logwarn_throttle(1.0, f"[motor {mid}] follow_err {abs(safe-actual):.2f}rad -> kp backoff")
            return

        with self.lock:
            self.prev_arm_cmd[key] = safe
        self.send_motor_raw(mid, safe, kp)

    def send_motor_raw(self, mid, pos, kp):
        if pos is None or not np.isfinite(pos):   # NaN 가드
            return
        cmd = MotorCommand()
        cmd.id = mid
        cmd.p_des = float(pos)
        cmd.kp = float(kp)
        cmd.kd = 0.8
        cmd.v_des = 0.0
        cmd.t_ff = 0.0
        self.motor_pub.publish(cmd)

    def send_twist_v2(self, x, z, is_kick=False):
        if not self.is_robot_ready: return
        if x is None or not np.isfinite(x): x = 0.0   # NaN 가드
        if z is None or not np.isfinite(z): z = 0.0
        if is_kick:
            max_spd, acc, dec = self.cfg.kick_max_speed, self.cfg.kick_acc_limit, self.cfg.kick_acc_limit
        else:
            max_spd, acc, dec = self.cfg.max_linear_speed, self.cfg.acc_limit, self.cfg.dec_limit
        # 후진 전용 속도 상한을 적용하며 송신할 때만 방향을 반전한다.
        if self.cfg.reverse_gait:
            max_spd = self.cfg.reverse_kick_speed if is_kick else self.cfg.reverse_max_speed

        tvx = np.clip(x, -max_spd, max_spd)
        dx = tvx - self.prev_cmd_x
        dx = min(dx, acc) if dx > 0 else max(dx, -dec)
        fx = np.clip(self.prev_cmd_x + dx, -max_spd, max_spd)
        fz = np.clip(self.prev_cmd_wz + np.clip(z - self.prev_cmd_wz, -self.cfg.body_slew_rate, self.cfg.body_slew_rate),
                     -self.cfg.max_angular_speed, self.cfg.max_angular_speed)
        # 지연 보상에 사용하는 prev_cmd_x는 공 접근을 양수로 유지한다.
        self.prev_cmd_x, self.prev_cmd_wz = fx, fz

        # TRON1 로 나가는 선속도만 반전 (역관절: 공 접근 = TRON1 -x)
        out_x = -fx if self.cfg.reverse_gait else fx

        with self.ws_lock:
            if self.is_connected and self.ws:
                try:
                    self.ws.send(json.dumps({
                        "accid": self.cfg.robot_sn, "title": "request_twist",
                        "timestamp": int(time.time() * 1000),
                        "guid": str(uuid.uuid4()).replace("-", ""),
                        "data": {"x": float(out_x), "y": 0.0, "z": float(fz)}
                    }))
                except Exception as e:
                    rospy.logwarn_throttle(3.0, f"[twist] {e}")

    def _on_watchdog(self, name):
        """비전·제어 루프가 멈추면 정지 상태를 재시작까지 유지한다."""
        self._safe_stop_active = True
        try:
            self.send_twist_v2(0.0, 0.0)
        except Exception:
            pass

    def _goal_direction(self, imu_yaw):
        """최신 위치 추정, IMU 기억 순으로 로봇 기준 골대 각(rad)과 유효 여부를 반환한다."""
        loc = self.loc_client.goal_direction()
        if loc is not None:
            return loc, True
        return self.goal_memory.get_goal_direction(imu_yaw), self.goal_memory.is_valid()

    def _publish_odom(self, cvx, cwz, imu_yaw):
        """실측 odom 대신 명령 속도와 IMU로 위치 추정기의 예측 입력을 만든다."""
        if self.odom_pub is None:
            return
        now = time.time()
        dt = max(1e-3, now - self._prev_odom_time)
        dtheta = (imu_yaw - self._prev_imu_yaw_odom + math.pi) % (2 * math.pi) - math.pi
        self._prev_imu_yaw_odom = imu_yaw
        self._prev_odom_time = now
        try:
            self.odom_pub.publish(RosString(data=json.dumps({
                "dx": float(cvx * dt), "dy": 0.0, "dtheta": float(dtheta),
                "vx": float(cvx), "wz": float(cwz), "imu_yaw": float(imu_yaw),
            })))
        except Exception:
            pass

    def shutdown(self):
        # 정지 + 상체 토크 해제 (kp=0). 종료 시 팔이 hold 토크를 물고 있지 않게.
        print("\nShutdown...")
        self._safe_stop_active = True
        self.ws_should_reconnect = False
        if getattr(self, "tronb_gc", None) is not None:
            try: self.tronb_gc.stop()
            except Exception: pass
        self.vision_wd.stop()
        self.control_wd.stop()
        self.send_twist_v2(0.0, 0.0)
        snap = self.get_motor_snapshot()
        for _ in range(10):
            for mid in self.ALL_MOTOR_IDS:
                if mid in snap:
                    self.send_motor_raw(mid, snap[mid], 0.0)
            time.sleep(0.02)
        print("상체 토크 해제 완료")


if __name__ == "__main__":
    # 현장 튜닝 파라미터
    tuning = TuningParams(
        align_good_angle = 0.10,      # 5.7도 이하면 정렬 완료로 보고 킥
        align_turn_speed = 0.08,

        aim_settle_time = 0.25,
        aim_glance_time = 1.2,        # 기억 없을 때만 고개 들어 스캔하는 시간
        aim_timeout = 2.5,            # 이 시간 넘으면 정렬 못해도 킥
        aim_look_pitch = 0.20,

        # 지연 보상 조기 정지
        perception_latency = 0.15,    # 비전 지연(초), 실측값으로 조정
        stop_dist = 0.50,             # 보상 거리 이 값 이하면 AIM 진입
        stop_confirm_frames = 2,

        # 곡선 주행 (AIM 회전량 줄이려고 미리 골대쪽으로 휘어 접근)
        curved_approach_enabled = True,
        curved_approach_dist = 0.6,
        curved_approach_gain = 0.15,

        # 골대 메모리 (접근 중 유지되도록 decay 낮춤)
        goal_yaw_rate_decay = 0.4,

        straight_drift_correction = -0.05,
        approach_drift_correction = -0.05,

        search_drift_compensation = -0.015,
        search_turn_speed = 0.10,
        search_turn_period = 6.0,
        search_turn_bias = 0.05,
        search_head_pitch_center = -0.05,
    )

    node = None
    try:
        node = BallFollower(tuning=tuning)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"[Fatal] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.shutdown()
