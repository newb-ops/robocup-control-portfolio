from __future__ import annotations
"""
features.py — ball_stop 만 유지 (v10.4.2 컴팩트화).
공 정지/속도 추정. localization 불필요. 입력 없어도 켜면 무에러(빈 관측→기본 snap).
(이전의 latency/landmark_comp/setplay/telemetry 는 제거됨.)
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from .common import BallObservation, now_sec


@dataclass
class BallMotionSnap:
    moving: bool = False
    speed: float = 0.0
    local_x: Optional[float] = None     # forward (m)
    local_y: Optional[float] = None     # left (m)
    stop_x: Optional[float] = None
    stop_y: Optional[float] = None
    eta: Optional[float] = None


def _predict_stop(px, py, vx, vy, mu=0.4, g=9.81):
    """등감속 마찰 모델. 정지까지 거리 = v^2/(2 mu g)."""
    sp = math.hypot(vx, vy)
    if sp < 1e-3:
        return px, py, 0.0
    d = sp * sp / (2.0 * mu * g)
    ux, uy = vx / sp, vy / sp
    eta = sp / (mu * g)
    return px + ux * d, py + uy * d, eta


class BallStopEstimator:
    """로봇-로컬 좌표(forward/left)로 공 위치를 추적해 정지점 추정. localization 없이 동작."""
    def __init__(self, fov_h: float, move_thresh: float = 0.15, miss_reset: int = 8, friction_mu: float = 0.4):
        self.fov_h = fov_h
        self.move_thresh = move_thresh
        self.miss_reset = miss_reset
        self.mu = friction_mu
        self._hist: deque = deque(maxlen=8)
        self._miss = 0

    def reset(self) -> None:
        self._hist.clear()
        self._miss = 0

    def _local_xy(self, obs: BallObservation) -> Optional[Tuple[float, float]]:
        if not obs.detected or not math.isfinite(obs.dist) or obs.dist >= 9.0:
            return None
        ang = obs.screen_x * (self.fov_h / 2.0)   # 화면각 → 방위
        return obs.dist * math.cos(ang), -obs.dist * math.sin(ang)

    def update(self, obs: BallObservation) -> BallMotionSnap:
        lx = self._local_xy(obs)
        if lx is None:
            self._miss += 1
            if self._miss >= self.miss_reset:
                self._hist.clear()
            return BallMotionSnap()
        self._miss = 0
        self._hist.append((now_sec(), lx[0], lx[1]))
        if len(self._hist) < 3:
            return BallMotionSnap(moving=False, local_x=lx[0], local_y=lx[1])
        t0, x0, y0 = self._hist[0]
        t1, x1, y1 = self._hist[-1]
        dt = max(1e-3, t1 - t0)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        sp = math.hypot(vx, vy)
        if sp < self.move_thresh:
            return BallMotionSnap(moving=False, speed=sp, local_x=x1, local_y=y1, stop_x=x1, stop_y=y1, eta=0.0)
        sx, sy, eta = _predict_stop(x1, y1, vx, vy, self.mu)
        return BallMotionSnap(moving=True, speed=sp, local_x=x1, local_y=y1, stop_x=sx, stop_y=sy, eta=eta)


# robot_state 훅 (제어 FSM 계약)
def attach_ball_stop_to_robot_state(robot_state: dict, motion: "BallMotionSnap") -> None:
    """ball_stop 결과를 robot_state 에 기록. setplay/제어 FSM 이 공 이동 판단에 사용."""
    if motion is None:
        robot_state["ball_stop_status"] = "UNKNOWN"
        robot_state["ball_stop_reason"] = "no motion snapshot"
        robot_state["ball_speed_est"] = None
        robot_state["ball_fit_n"] = 0
        return
    has_xy = motion.local_x is not None
    if not has_xy:
        status, reason = "UNKNOWN", "no ball local xy"
    elif motion.moving:
        status, reason = "MOVING", "speed above move threshold"
    else:
        status, reason = "STOPPED", "speed below move threshold"
    robot_state["ball_stop_status"] = status
    robot_state["ball_stop_reason"] = reason
    robot_state["ball_speed_est"] = float(motion.speed)
    robot_state["ball_fit_n"] = 2 if has_xy else 0
    if motion.stop_x is not None:
        robot_state["ball_pred_stop_local_x"] = float(motion.stop_x)
        robot_state["ball_pred_stop_local_y"] = float(motion.stop_y)


def build_vision_packet(*, obs, goal_visible: bool, goal_centered: bool,
                        motion, localization_reliable: bool) -> dict:
    """setplay adapter 가 읽는 경량 packet (vision_state 의 full packet 과 별개)."""
    return {
        "ball_found": bool(obs.detected),
        "ball_dist": float(obs.dist),
        "ball_screen_x": float(obs.screen_x),
        "goal_visible": bool(goal_visible),
        "goal_centered": bool(goal_centered),
        "ball_moving": bool(motion.moving) if motion is not None else False,
        "ball_speed": float(motion.speed) if motion is not None else 0.0,
        "localization_reliable": bool(localization_reliable),
    }
