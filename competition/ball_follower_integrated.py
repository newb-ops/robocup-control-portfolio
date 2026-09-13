#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 카메라 영상과 모터 피드백, TRON1 IMU, 선택적 GameController 상태를 받아 경기 동작을 제어한다.
tronb_vision의 인지 결과를 DecisionEngine에 전달하고 WebSocket 보행 명령과 ROS 상체 모터 명령을 출력한다.
비전과 제어는 별도 스레드로 실행하며 피드백, watchdog, 자세 한계를 감시한다.
실기 실행 전 모터 매핑과 카메라 보정값을 확인해야 하며 ROS 통합 실행은 별도 검증이 필요하다."""

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

try:
    from tron_upper_body.msg import MotorCommand, MotorFeedback
except ImportError:
    class MotorCommand: pass
    class MotorFeedback: pass

# ROS 통합 실행과 하드웨어 설정은 실기에서 확인해야 한다.
from tronb_vision.app import BallFollower as _VisionEngine
from tronb_vision.common import (RobotConfig as _TConfig, TuningParams as _TTuning,
                                 OperationMode as _TMode, TriState as _TTri)
from tronb_vision.drivers import RobotDriver as _TDriver
from tronb_vision.detector import create_detector as _create_detector
from tronb_vision.sources import DummySource as _DummySource, FramePacket as _FramePacket, CameraIntrinsics as _Intr
from tronb_vision.decision import DecisionEngine as _DecisionEngine
try:
    from tronb_vision.gc_state import GcReceiver as _GcReceiver
except Exception:
    _GcReceiver = None


class _YawProxyDriver(_TDriver):
    """IMU yaw를 비전 위치 추정기에 전달한다. get_base_pose()[2]가 yaw(rad)다."""
    def __init__(self):
        self._yaw = 0.0
    def start(self): pass
    def send_twist(self, x, z, is_kick=False): pass   # 실제 구동은 ball_follower 가 함
    def set_head(self, yaw, pitch): pass
    def get_base_pose(self):
        return (0.0, 0.0, float(self._yaw))

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
    robot_sn: str = "SF_TRON1A_359"
    ws_url: str = "ws://192.0.2.11:5000"
    model_path: str = "weights/real_best.onnx"   # 경기 번들 디렉터리를 기준으로 한 실카메라 모델 경로
    debug: bool = True
    verbose_log: bool = True

    # 모터 ID와 방향: 현재 하드웨어에서 확인할 것
    head_yaw_id: int = 5      # 목 yaw (좌우)
    head_pitch_id: int = 6    # 목 pitch (위아래)
    l_roll_id: int = 2        # 왼 어깨 roll (밖 -)
    l_pitch_id: int = 3       # 왼 어깨 pitch(앞 +)
    r_roll_id: int = 1        # 오른 어깨 roll (밖 +)
    r_pitch_id: int = 4       # 오 어깨 pitch(앞 -)
    l_elbow_id: int = 7       # 왼 팔꿈치 (앞 +)
    r_elbow_id: int = 8       # 오른 팔꿈치 (앞 -)

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
    # 게걸음 y와 회전 z의 방향은 실기에서 확인한다.
    reverse_strafe: bool = True        # 게걸음 y 반전. 왼쪽 의도가 오른쪽이면 False
    reverse_turn: bool = False         # 회전 z 반전. 좌회전이 오른쪽으로 돌면 True
    max_strafe_speed: float = 0.03     # 게걸음 속도 상한 [m/s]; 현재 decision은 cvy=0
    # TRON1 STAND/WALK 모드 전환
    skip_mode_switch: bool = True      # True: 조이스틱으로 WALK 세팅 후 시작(WS 모드 전환·대기 생략).
                                       # False: 코드가 WS 로 request_stand_mode/walk_mode 직접 제어.

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
    # 상체 swing 게인과 팔꿈치 추종 설정
    arm_swing_kp: float = 12.0         # 어깨 swing Kp; 낮을수록 외력에 유연하게 반응
    elbow_kp: float = 8.0              # 팔꿈치 kp
    elbow_base_bend: float = 0.30      # 팔꿈치 기본 굽힘 [rad]
    elbow_follow: float = 0.40         # 어깨 swing 추종 비율 (secondary motion)
    elbow_delay: float = 0.12          # 어깨 대비 지연 [s] (overlapping action)

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
    rel_limit_arm_elbow: float = 1.00     # 팔꿈치; 실측 가동범위에 맞춰 조정

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
                       self.cfg.l_pitch_id, self.cfg.r_pitch_id,
                       self.cfg.l_elbow_id, self.cfg.r_elbow_id]
        self.HEAD_IDS = [self.cfg.head_yaw_id, self.cfg.head_pitch_id]
        self.ALL_MOTOR_IDS = self.ARM_IDS + self.HEAD_IDS

        self.swing_hist = deque(maxlen=64)   # 팔꿈치 secondary motion 용 (어깨 swing 지연)

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
        # 검출 모델은 vision_engine에서만 로드한다.
        # self.model을 비워 중복 모델 메모리를 사용하지 않는다.
        self.model = None

        print(f"YOLO Model (vision_engine): {self.cfg.model_path}")

        self.latest_rgb = None
        self.latest_depth = None
        self.new_frame = False
        self.img_width = 640
        self.img_height = 480

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

        self._setup_integration()

        self._start_ws()
        threading.Thread(target=self.control_loop, daemon=True).start()
        threading.Thread(target=self.vision_loop, daemon=True).start()

    def _setup_integration(self):
        """비전(tronb_vision) + 결정(control_fsm+setplay) + GC 엔진 구성."""
        # D435 640×480 기본 intrinsics. 실제 카메라 보정값으로 조정한다.
        fx = float(getattr(self.cfg, "fx", 415.69)); fy = float(getattr(self.cfg, "fy", 415.69))
        cx = float(getattr(self.cfg, "cx", 320.0));  cy = float(getattr(self.cfg, "cy", 240.0))
        self._intr = _Intr(fx, fy, cx, cy, self.img_width, self.img_height)
        # 바닥 투영에 사용하는 카메라 하향 pitch. IMU 연동 시 매 프레임 갱신한다.
        self._cam_pitch_down_deg = float(getattr(self.cfg, "camera_pitch_down_deg", 0.0))

        # 실카메라 검출 모델과 위치 추정기를 사용한다.
        tcfg = _TConfig(
            detector_backend=getattr(self.cfg, "detector_backend", "onnx"),
            localization_enabled=True,
        )
        tcfg.model_path = self.cfg.model_path
        tcfg.img_width = self.img_width; tcfg.img_height = self.img_height
        ttuning = _TTuning()
        self._yaw_proxy = _YawProxyDriver()
        # 비전 엔진은 _perceive로 관측과 robot_state만 만든다.
        # 비전 엔진은 LOCKED로 두고 결정과 구동은 ball_follower에서 처리한다.
        self.vision_engine = _VisionEngine(
            tcfg, ttuning, _DummySource(), self._yaw_proxy,
            detector=_create_detector(tcfg), bgr_input=True)
        try:
            self.vision_engine.set_mode(_TMode.LOCKED)
        except Exception:
            pass
        self.vision_engine.loc.enabled = True
        for f in ("localization", "ball_stop", "rule_state", "vision_state", "setplay"):
            if f in self.vision_engine.features:
                self.vision_engine.features[f] = _TTri.ON

        # 결정 엔진(control_fsm + setplay 우선순위)
        self.decision_engine = _DecisionEngine()
        self._last_decision = None

        # GameController 수신 스레드
        self.gc = None
        if _GcReceiver is not None and getattr(self.cfg, "gc_enabled", False):
            try:
                self.gc = _GcReceiver(
                    getattr(self.cfg, "team_number", 0),
                    getattr(self.cfg, "player_number", 1),
                    verbose=False)
                self.gc.start()
                print(f"[통합] GC 수신 시작 team={getattr(self.cfg,'team_number',0)} "
                      f"player={getattr(self.cfg,'player_number',1)}")
            except Exception as e:
                print(f"[통합] GC 시작 실패(무시): {e}")
        self._gc_snap = {"game_controller_state": "UNKNOWN"}
        print("[통합] 비전=tronb_vision, 결정=control_fsm+setplay 구성 완료")

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
        """tronb_vision._perceive로 영상에서 robot_state를 만들고 제어 스레드에 전달한다."""
        rate = rospy.Rate(30)
        while not self.vision_ready and not rospy.is_shutdown():
            self.vision_wd.feed()
            rate.sleep()
        print("Vision Loop Active (통합: tronb_vision 검출+localization)")
        self.vision_wd.arm()
        fid = 0
        while not rospy.is_shutdown():
            try:
                self.vision_wd.feed()
                frame, depth = None, None
                with self.img_lock:
                    if self.new_frame and self.latest_rgb is not None:
                        frame = self.latest_rgb.copy()
                        depth = self.latest_depth.copy() if self.latest_depth is not None else None
                        self.new_frame = False
                if frame is None:
                    rate.sleep(); continue

                with self.lock:
                    imu_yaw = self.current_yaw_imu
                # PF odom(yaw) + GC/setplay 입력을 비전 엔진에 주입
                self._yaw_proxy._yaw = imu_yaw
                if self.gc is not None:
                    self._gc_snap = self.gc.snapshot()
                self.vision_engine.game_controller_state = self._gc_snap.get("game_controller_state", "UNKNOWN")
                self.vision_engine.cfg.restart_type = self._gc_snap.get("restart_type")
                self.vision_engine.cfg.restart_owner = self._gc_snap.get("restart_owner")

                fid += 1
                pkt = _FramePacket(rgb=frame, depth=depth, capture_time=time.time(),
                                   frame_id=fid, intrinsics=self._intr,
                                   imu_pitch_deg=self._cam_pitch_down_deg, pitch_source="manual")
                # tronb_vision 인지 → robot_state(self.vision_engine.robot_state) 생산
                obs = self.vision_engine._perceive(pkt)

                # ball_follower 호환 변수 반영(head 추종 등에서 참조)
                with self._obs_lock:
                    self._ball_obs = obs
                with self.lock:
                    self.ball_dist = obs.dist
                    self.ball_screen_x = obs.screen_x
                    self.ball_screen_y = obs.screen_y
                    self.lost_frames = obs.lost_frames

                # 디버그 이미지(선택): tronb_vision overlay
                try:
                    from tronb_vision.debug_view import draw_overlay
                    dbg = draw_overlay(self.vision_engine._last_bgr, self.vision_engine)
                    if dbg is not None and self.debug_pub is not None:
                        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
                except Exception:
                    pass

                rate.sleep()
            except Exception as e:
                print(f"Vision: {e}")
                import traceback; traceback.print_exc()
                time.sleep(0.1)

    def _exec_state_to_robotstate(self, name: str):
        """decision exec_state 문자열 → ball_follower RobotState (팔스윙/kick 포즈 게이팅용)."""
        try:
            return RobotState[name]
        except (KeyError, ValueError):
            return RobotState.SEARCH

    def _update_head_for(self, exec_state: str, obs):
        """결정 상태에 따라 머리를 제어한다.
        SEARCH는 스캔, TRACK/APPROACH는 공 추종, AIM/KICK은 공·전방 주시,
        RECOVER/COOLDOWN은 중앙 유지다. 실기에서 방향과 범위를 확인해야 한다."""
        hc = self.head_controller
        if exec_state == "SEARCH":
            ct = time.time()
            scan_yaw = self.cfg.scan_amplitude * math.sin(ct * self.cfg.scan_speed)
            hc.set_target_direct(scan_yaw, self.cfg.scan_pitch_center)
        elif exec_state in ("TRACK", "APPROACH"):
            if getattr(obs, "detected", False):
                yaw_err, _ = self.screen_to_angle(obs.screen_x, 0.0)
                new_yaw = float(np.clip(hc.target_yaw + 0.3 * yaw_err, *self.cfg.head_yaw_limit))
                pitch = self.get_approach_pitch(obs.dist) if exec_state == "APPROACH" else hc.target_pitch
                hc.set_target(new_yaw, pitch)
        elif exec_state in ("AIM", "KICK"):
            hc.set_target(0.0, self.cfg.approach_pitch_near)
        else:
            hc.set_target(0.0, self.cfg.scan_pitch_center)

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

        if self.cfg.skip_mode_switch:
            # 조이스틱으로 STAND/WALK를 설정한 경우 WebSocket 모드 전환을 생략한다.
            # 시작 전 WALK 모드에서 twist=0일 때 정지 자세가 유지되는지 확인한다.
            print("[부팅] 조이스틱 WALK 모드 가정 — WS STAND/WALK 전환 생략")
            time.sleep(1.0)
            self.sync_offsets()
        else:
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

        if not self.cfg.skip_mode_switch:
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

            # 상태 전이는 새 프레임에서만 (중복 방지)
            # 연속 제어(steering, speed)는 매 tick 수행 OK
            # 통합 결정: GC penalty/SET > setplay > control_fsm (decision.py)
            # 비전(tronb_vision)이 채운 robot_state 사용. (전이/상태머신/fall은 decision/ control_fsm 가 처리)
            t = time.time() - start_time
            turn_dir = -1 if self.cfg.body_turn_inverted else 1
            fallen = (math.isfinite(body_roll) and math.isfinite(body_pitch) and
                      (abs(body_roll) > self.cfg.roll_fall_limit or
                       abs(body_pitch) > self.cfg.pitch_fall_limit))
            rs = self.vision_engine.robot_state
            lower = {"is_fallen": bool(fallen), "roll": float(body_roll),
                     "pitch": float(body_pitch), "yaw": float(imu_yaw)}
            decision = self.decision_engine.decide(
                rs, lower, gc_state_name=self._gc_snap.get("game_controller_state"))
            self._last_decision = decision

            cvx, cwz = float(decision.vx), float(decision.wz)
            is_kick_now = bool(decision.kick)
            curr = self._exec_state_to_robotstate(decision.exec_state)
            with self.lock:
                self.state = curr
            self.kick_settling = False   # 킥 타이밍은 control_fsm 이 관리(settling 미사용)

            # 머리 추종은 exec_state를 기준으로 갱신한다.
            self._update_head_for(decision.exec_state, obs)

            self.send_twist_v2(cvx, cwz, is_kick=is_kick_now)

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
            kp = self.cfg.arm_swing_kp

            do_swing = curr not in [RobotState.AIM, RobotState.KICK, RobotState.RECOVER,
                                    RobotState.COOLDOWN]
            if do_swing:
                total = np.clip(self.cfg.arm_swing_base + (abs(cvx) + abs(cwz)) * self.cfg.arm_swing_speed_gain, 0, 0.4)
                swing = total * math.sin(t * self.cfg.arm_swing_freq)
                self.swing_hist.append((t, swing))   # 정상 swing 만 기록 (KICK 0.70 은 제외 → 복귀 시 튐 방지)

            if curr == RobotState.KICK and not self.kick_settling:
                swing = 0.70
                base_roll = 0.28

            left_roll = np.clip(-base_roll + bal, -0.35, 0.35)
            right_roll = np.clip(base_roll - bal, -0.35, 0.35)

            # 어깨 pitch (l=앞+, r=앞-: inverted=True 면 양쪽 +swing → 거울대칭 anti-phase)
            if self.cfg.arm_pitch_inverted:
                self.send_motor_offset(self.cfg.l_pitch_id, swing, kp)
                self.send_motor_offset(self.cfg.r_pitch_id, swing, kp)
            else:
                self.send_motor_offset(self.cfg.l_pitch_id, swing, kp)
                self.send_motor_offset(self.cfg.r_pitch_id, -swing, kp)

            self.send_motor_offset(self.cfg.l_roll_id, left_roll, kp)
            self.send_motor_offset(self.cfg.r_roll_id, right_roll, kp)

            # 팔꿈치 secondary motion (overlapping action): 어깨 swing 을 지연·감쇠 추종
            if do_swing:
                tt = t - self.cfg.elbow_delay
                delayed = swing
                for i in range(len(self.swing_hist) - 1, 0, -1):
                    t1, v1 = self.swing_hist[i]
                    t0, v0 = self.swing_hist[i - 1]
                    if t0 <= tt <= t1:
                        r = (tt - t0) / max(t1 - t0, 1e-6)
                        delayed = v0 + (v1 - v0) * r
                        break
                sec = self.cfg.elbow_follow * delayed
            else:
                sec = 0.0
            self.send_motor_offset(self.cfg.l_elbow_id,  self.cfg.elbow_base_bend + sec, self.cfg.elbow_kp)
            self.send_motor_offset(self.cfg.r_elbow_id, -self.cfg.elbow_base_bend + sec, self.cfg.elbow_kp)

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
        if mid in (self.cfg.l_elbow_id, self.cfg.r_elbow_id): return self.cfg.rel_limit_arm_elbow
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

    def send_twist_v2(self, x, z, fy=0.0, is_kick=False):
        if not self.is_robot_ready: return
        if x is None or not np.isfinite(x): x = 0.0   # NaN 가드
        if z is None or not np.isfinite(z): z = 0.0
        if fy is None or not np.isfinite(fy): fy = 0.0   # 게걸음 y (현재 decision 은 0)
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

        # 역관절 결합에 맞춰 송신할 축별 방향을 반전한다.
        out_x = -fx if self.cfg.reverse_gait else fx
        fy_c = float(np.clip(fy, -self.cfg.max_strafe_speed, self.cfg.max_strafe_speed))
        out_y = -fy_c if self.cfg.reverse_strafe else fy_c
        out_z = -fz if self.cfg.reverse_turn else fz

        with self.ws_lock:
            if self.is_connected and self.ws:
                try:
                    self.ws.send(json.dumps({
                        "accid": self.cfg.robot_sn, "title": "request_twist",
                        "timestamp": int(time.time() * 1000),
                        "guid": str(uuid.uuid4()).replace("-", ""),
                        "data": {"x": float(out_x), "y": float(out_y), "z": float(out_z)}
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
