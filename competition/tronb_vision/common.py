from __future__ import annotations
"""
common.py — 공유 타입/설정/헬퍼.

기준: v8.2의 타입 구성 + v6의 BallObservation/RobotState 계약.
하나의 진실: BallObservation 은 vision 이 한 번에 쓰고 control 이 한 번에 읽는 atomic 단위.
RobotState 는 control FSM 의 유일한 상태(=app.BallFollower.state).
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# 헬퍼
def now_sec() -> float:
    return time.time()


def wrap_to_pi(a: float) -> float:
    return (float(a) + math.pi) % (2 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def normalize_class_name(name: Any) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


# 모드 / 상태
class OperationMode(Enum):
    """운용 모드. RobotState(FSM) 와는 별개. LOCKED 가 기본(명령 차단)."""
    LOCKED = auto()    # 영상만, body/head 명령 차단 (안전 대기)
    AUTO = auto()      # detector + FSM 추적
    MANUAL = auto()    # 키보드 조작만 (detector/auto 정지)


class RobotState(Enum):
    """Control FSM. v9/v10 계열의 compact 상태셋 (구버전 BACK_OFF/BLIND 제거)."""
    SEARCH = auto()
    TRACK = auto()
    APPROACH = auto()
    AIM = auto()       # 정지 + 골 정렬 + 킥판정 (구 STOP/ALIGN_GOAL 통합)
    KICK = auto()
    COOLDOWN = auto()
    RECOVER = auto()


class GameControllerState(Enum):
    """RoboCup GameController 상태 (§7.1). comms 모듈이 채우고 control 이 게이팅에 사용."""
    UNKNOWN = auto()
    INITIAL = auto()
    READY = auto()
    SET = auto()
    PLAYING = auto()
    FINISHED = auto()
    PENALIZED = auto()
    STOP = auto()      # emergency stop


class TriState(Enum):
    """기능 토글 (1~9 키). ON=기능+표시, SEMI=계산만, OFF=비활성."""
    ON = auto()
    SEMI = auto()
    OFF = auto()

    def next(self) -> "TriState":
        return {TriState.ON: TriState.SEMI, TriState.SEMI: TriState.OFF, TriState.OFF: TriState.ON}[self]

    @property
    def enabled(self) -> bool:
        return self is not TriState.OFF

    @property
    def visible(self) -> bool:
        return self is TriState.ON


# 관측 / 검출
@dataclass
class BallObservation:
    """Vision 이 한 번에 쓰고 Control 이 한 번에 읽는 atomic 단위 (v6 계약 보존)."""
    area: float = 0.0
    dist: float = 9.99
    dist_from_area: float = 9.99
    screen_x: float = 0.0
    screen_y: float = 0.0
    last_dir: float = 0.0
    lost_frames: int = 0
    last_valid_dist: float = 2.0
    conf: float = 0.0
    timestamp: float = 0.0
    detected: bool = False
    consecutive_detect: int = 0
    kalman_accepted: bool = False
    frame_id: int = 0

    @property
    def confidence(self) -> float:
        """vision_state.py(제어팀) 호환 별칭. 내부 필드는 conf."""
        return self.conf


@dataclass
class GoalObservation:
    detected: bool = False
    screen_x: float = 0.0
    screen_y: float = 0.0
    area: float = 0.0
    width: float = 0.0
    conf: float = 0.0


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


# 튜닝 / 설정
@dataclass
class TuningParams:
    """현장 튜닝값."""
    # 골 정렬 / 킥
    align_good_angle: float = 0.10
    align_max_angle: float = 0.16
    align_turn_speed: float = 0.08
    kick_settle_time: float = 0.30
    kick_speed: float = 0.80
    # AIM (기억 우선 + 짧은 글랜스 + 직진 폴백)
    aim_settle_time: float = 0.25
    aim_glance_time: float = 1.2
    aim_timeout: float = 2.5
    aim_look_pitch: float = 0.20
    # 지연 보상 조기 정지
    perception_latency: float = 0.05   # ONNX 기준 (PT면 0.15)
    stop_dist: float = 0.50
    stop_confirm_frames: int = 2
    # 곡선 접근
    curved_approach_enabled: bool = True
    curved_approach_dist: float = 0.6
    curved_approach_gain: float = 0.15
    # 공 감지
    ball_conf_threshold: float = 0.40
    ball_min_area: int = 300
    ball_max_area: int = 90000
    lost_frames_threshold: int = 35
    # 골 기억
    goal_memory_decay: float = 0.02
    goal_memory_timeout: float = 30.0
    goal_conf_threshold: float = 0.25
    goal_yaw_rate_decay: float = 0.4
    # 이동 속도
    approach_speed_far: float = 0.060
    approach_speed_near: float = 0.025
    search_turn_speed: float = 0.10
    search_drift_compensation: float = -0.015
    search_turn_period: float = 6.0
    search_turn_bias: float = 0.03
    search_head_pitch_center: float = -0.05
    straight_drift_correction: float = -0.05
    approach_drift_correction: float = -0.05

    def print_summary(self):
        print(f"[Tuning] aim_glance/timeout={self.aim_glance_time}/{self.aim_timeout}s "
              f"latency={self.perception_latency}s stop_dist={self.stop_dist}m")


@dataclass
class RobotConfig:
    """시스템 설정 (하드웨어/모델/네트워크)."""
    # 모델
    model_path: str = "weights/sim_best.onnx"
    detector_backend: str = "onnx"      # onnx | ultralytics | dummy
    imgsz: int = 640
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    # 클래스: 이름 기반 해석이 우선. 아래 id는 폴백.
    # 모델 클래스 (best.onnx 기준 9개, id 순서 고정). 재학습으로 순서 바뀌면 여기만 수정.
    # ※ 실제로는 detector 가 onnx 내장 names 를 우선 사용하므로 이건 참조/폴백용.
    class_names: List[str] = field(default_factory=lambda: [
        "ball", "corner_arc", "goal", "goal_bar",
        "line_l", "line_t", "line_x", "penalty_mark", "robot",
    ])
    ball_class_id: int = 0
    goal_class_id: int = 2
    goal_class_ids: Tuple[int, ...] = (2, 3)   # goal, goal_bar
    robot_class_id: int = 8
    use_class_filter: bool = True
    # 카메라/검출
    camera_fov_h: float = 1.047
    camera_fov_v: float = 0.785
    img_width: int = 640
    img_height: int = 480
    realsense_fps: int = 30
    realsense_serial: str = ""
    realsense_emitter: float = -1.0
    realsense_laser_power: float = -1.0
    use_imu_pitch: bool = False
    use_sysfs_imu_pitch: bool = True
    imu_pitch_alpha: float = 0.15
    imu_pitch_offset_deg: float = 0.0
    imu_accel_fps: int = 0
    camera_height: float = 1.23
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_pitch_down_deg: float = 20.0
    real_position_enabled: bool = True
    real_floor_ray_fallback: bool = True
    real_ball_bbox_fallback: bool = True
    real_depth_min: float = 0.08
    real_depth_max: float = 10.0
    capture_dir: str = "captures_camera"
    depth_scale: float = 0.001
    depth_sample_radius: int = 5
    depth_min_valid: float = 0.1
    depth_max_valid: float = 3.0
    ball_real_diameter: float = 0.22
    camera_focal_length: float = 600.0
    conf_threshold_strict: float = 0.55
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.50
    max_area_change_ratio: float = 3.5
    min_consecutive_detect: int = 3
    close_ball_area_threshold: int = 15000
    close_ball_max_area_change: float = 5.0
    goal_conf_weak: float = 0.15
    goal_min_area: int = 300
    detection_buffer_size: int = 12
    kalman_predict_time: float = 0.1
    # head/IMU 부호
    head_yaw_inverted: bool = True
    head_pitch_inverted: bool = False
    imu_compensation_gain: float = 0.5
    head_yaw_limit: Tuple[float, float] = (-0.6, 0.6)
    head_pitch_limit: Tuple[float, float] = (-0.70, 0.30)
    # 모터 ID (실로봇 driver 용)
    head_yaw_id: int = 5
    head_pitch_id: int = 6
    l_roll_id: int = 1
    l_pitch_id: int = 3
    r_roll_id: int = 2
    r_pitch_id: int = 4
    # 속도 제한
    max_linear_speed: float = 0.08
    max_angular_speed: float = 0.35
    acc_limit: float = 0.025
    dec_limit: float = 0.040
    body_slew_rate: float = 0.040
    kick_max_speed: float = 0.85
    kick_acc_limit: float = 0.5
    # 상태 전이
    track_to_approach_yaw: float = 0.15
    track_min_frames: int = 5
    track_min_dist: float = 2.0
    approach_to_track_yaw: float = 0.35
    lost_ball_turn_speed: float = 0.08
    lost_ball_search_frames: int = 15
    body_turn_deadzone: float = 0.03
    body_turn_gain: float = 0.90
    body_turn_inverted: bool = False
    approach_pitch_near: float = -0.65
    # 안정성
    roll_fall_limit: float = 0.35
    pitch_fall_limit: float = 0.40
    cooldown_time: float = 2.5
    recover_time: float = 1.5
    # localization (기본 OFF — 카메라자세/시딩 해결 전)
    localization_enabled: bool = False
    # 제어 FSM 통합/검증: True 면 control_fsm 이 robot_state 를 읽어 구동(노트북 단독 검증),
    # False(기본)면 내부 FSM 구동. control_log_path 지정 시 매 프레임 JSONL 로그.
    external_control: bool = False
    control_log_path: Optional[str] = None
    # GameController 상태 주입(미주입=None→UNKNOWN, permissive). vision_state 가 참고.
    game_controller_state: Optional[str] = None
    # setplay sim 테스트용 restart 주입(실전은 GC gc_state 가 robot_state 에 머지). None→NONE.
    restart_type: Optional[str] = None
    restart_owner: Optional[str] = None
    attack_positive_x: bool = True
    # PF 에 넣을 랜드마크 타입(이 필드 모델 기준 신뢰 가능한 것만). center/goalpost 제외.
    # 골 설치+모델 개선되면 ("L","T","goalpost") 등으로 확장.
    loc_landmark_classes: tuple = ("L", "T", "X", "corner_arc", "penalty_mark", "goalpost")
    # 하프코트(노트북+카메라 벤치): 존재하는 절반만 맵으로 → 대칭 모호성 제거.
    # x_pos(골 +x 절반, 기본) / x_neg / full(대회 전체)
    loc_field_mode: str = "x_pos"
    # PF 초기 분포: uniform(맵 범위 전체 균일, 어디서 시작하든) / own_half / 코드에서 seed
    loc_init_mode: str = "uniform"
    # 시뮬 (MuJoCo scene 의 정확한 이름. 네 xml 에 맞춰 여기만 바꾸면 됨)
    step_mujoco_physics: bool = True
    sim_robot_joint: str = "free"            # 로봇 floating base freejoint
    sim_neck_yaw_joint: str = "neck_yaw"
    sim_neck_pitch_joint: str = "neck_pitch"
    sim_camera_name: str = "realsense"
    sim_ball_body: str = "soccer_ball"
    sim_ball_joint: str = "soccer_ball_free"
    # 통신
    gc_enabled: bool = False
    gc_listen_port: int = 3838
    team_number: int = 0
    team_send_enabled: bool = False


# 클래스 이름 alias (모델마다 이름이 달라서 normalize 후 매칭)
BALL_ALIASES = ["ball", "ball-Hz6O", "soccerball", "soccer_ball"]
GOAL_ALIASES = ["goal", "goalframe", "goal-frame", "goalmouth"]
BAR_ALIASES = ["goal_bar", "bar", "crossbar", "goalpost", "post"]
ROBOT_ALIASES = ["robot", "opponent", "nao", "dummy_robot"]
LINE_L_ALIASES = ["line_l", "two_angle", "2-angle", "l", "lcross", "corner"]
LINE_T_ALIASES = ["line_t", "three_angle", "3-angle", "t", "tcross"]
LINE_X_ALIASES = ["line_x", "x", "xcross", "cross"]
POINT_ALIASES = ["center_mark", "penalty_mark", "point", "mark"]


class FpsMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self.times: List[float] = []

    def tick(self) -> float:
        t = now_sec()
        self.times.append(t)
        if len(self.times) > self.window:
            self.times.pop(0)
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / dt if dt > 0 else 0.0


def apply_overrides(cfg: RobotConfig, tuning: TuningParams, data: dict):
    """yaml dict 의 키를 RobotConfig/TuningParams 에 덮어쓴다 (있는 필드만)."""
    for section, obj in (("robot", cfg), ("tuning", tuning), ("config", cfg)):
        for k, v in (data.get(section, {}) or {}).items():
            if hasattr(obj, k):
                setattr(obj, k, v)
    # 평면 키도 허용
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        elif hasattr(tuning, k):
            setattr(tuning, k, v)
    return cfg, tuning


def load_config(path: Optional[str] = None):
    cfg, tuning = RobotConfig(), TuningParams()
    if path:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        apply_overrides(cfg, tuning, data)
    return cfg, tuning


import os as _os


def check_model_and_backend(cfg, source_kind: str):
    """모델 파일 존재 확인 + sim/real 혼동 경고 + .pt 자동 백엔드.
    source_kind: 'mujoco' | 'realsense' | 'dummy'. 문제가 치명적이면 SystemExit."""
    if cfg.detector_backend == "dummy":
        return
    mp = cfg.model_path
    # .pt 인데 onnx 백엔드면 자동 교정
    if mp.endswith(".pt") and cfg.detector_backend == "onnx":
        cfg.detector_backend = "pt"
        print(f"[check] '{mp}' 는 .pt 라 backend 를 pt(ultralytics)로 자동 변경")
    if mp.endswith(".onnx") and cfg.detector_backend in ("pt", "ultralytics"):
        cfg.detector_backend = "onnx"
        print(f"[check] '{mp}' 는 .onnx 라 backend 를 onnx 로 자동 변경")
    # 존재 확인
    if not _os.path.exists(mp):
        raise SystemExit(
            f"[ERROR] 모델 파일 없음: {mp}\n"
            f"  - sim 은 weights/sim_best.(onnx|pt), 실카메라는 weights/real_best.onnx 를 쓴다.\n"
            f"  - PT->ONNX 변환: yolo export model=best.pt format=onnx imgsz=640 opset=12")
    # sim/real 혼동 경고 (파일명 휴리스틱, 경고만)
    base = _os.path.basename(mp).lower()
    if source_kind == "mujoco" and "real" in base:
        print("[WARN] MuJoCo 시뮬에 'real' 모델을 쓰고 있다. 보통 sim 모델(sim_best)을 쓴다.")
    if source_kind == "realsense" and "sim" in base:
        print("[WARN] 실 카메라에 'sim' 모델을 쓰고 있다. 잔디/실환경엔 real 모델(real_best)이 맞다.")
