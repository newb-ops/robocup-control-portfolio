from __future__ import annotations
"""
drivers.py — 구동 추상화 (기준: v8.2). 공통 로직은 그대로, 전송만 교체.

RobotDriver:
  send_twist(x, z)        하체 속도 명령
  set_head(yaw, pitch)    머리 각 명령
  get_base_pose()         (sim) 로봇 pose / (real) None
  get_ball_relative()     (sim) 공 상대거리 GT / (real) None
  step()                  (sim) 물리 step
주의: 실로봇 driver(WebSocketRobotDriver)는 v11 의 모터 안전 파이프라인을 반드시 이식해야 한다.
      여기 RealStubDriver 는 결선 지점만 잡아둔 stub.
"""

import math
import time
from typing import Optional, Tuple

import numpy as np

from .common import RobotConfig, clamp


class RobotDriver:
    def start(self): ...
    def stop(self): ...
    def send_twist(self, x: float, z: float, is_kick: bool = False): ...
    def set_head(self, yaw: float, pitch: float): ...
    def get_base_pose(self) -> Optional[Tuple[float, float, float]]: return None
    def get_ball_relative(self) -> Optional[dict]: return None
    def read_imu(self) -> Tuple[float, float, float]: return 0.0, 0.0, 0.0
    def step(self): ...
    def reset_pose(self): ...


class DryRunDriver(RobotDriver):
    """명령을 받기만 하고 아무 데도 안 보냄. headless/selftest."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.last_twist = (0.0, 0.0); self.last_head = (0.0, 0.0)
    def send_twist(self, x, z, is_kick=False):
        self.last_twist = (float(x), float(z))
        if self.verbose: print(f"[dry] twist=({x:.3f},{z:.3f})")
    def set_head(self, yaw, pitch):
        self.last_head = (float(yaw), float(pitch))


class MuJoCoRobotDriver(RobotDriver):
    """MuJoCo: freejoint qpos 직접 갱신(teleport). 실제 보행 아님 — 비전/전략 시각 검증용.
    scene 이름은 cfg.sim_* 에서 가져온다 (정확한 단일 이름. 바꾸려면 config 한 곳만)."""
    def __init__(self, source, cfg: RobotConfig):
        self.source = source; self.cfg = cfg
        self.qpos_start = None; self.yaw_qpos = None; self.pitch_qpos = None
        self.base_z = 0.30; self._ball_qpos = None; self._ball_body = None
        self.prev_x = 0.0; self.prev_z = 0.0
        self.mujoco = self.model = self.data = None

    def _joint_qpos(self, name, required=True):
        j = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            if required:
                raise RuntimeError(f"MuJoCo joint '{name}' 못 찾음. cfg.sim_* 이름을 scene 과 맞춰라.")
            return None
        return int(self.model.jnt_qposadr[j])

    def start(self):
        h = self.source.get_sim_handles()
        if h is None:
            raise RuntimeError("MuJoCoRobotDriver: source 가 sim 핸들을 안 줌")
        self.mujoco, self.model, self.data = h
        c = self.cfg
        self.qpos_start = self._joint_qpos(c.sim_robot_joint, required=True)
        self.base_z = float(self.data.qpos[self.qpos_start + 2]) or self.base_z
        self.yaw_qpos = self._joint_qpos(c.sim_neck_yaw_joint, required=False)
        self.pitch_qpos = self._joint_qpos(c.sim_neck_pitch_joint, required=False)
        print(f"[MuJoCoRobotDriver] robot='{c.sim_robot_joint}' "
              f"neck=('{c.sim_neck_yaw_joint}','{c.sim_neck_pitch_joint}')")
        self._cache_ball()

    def _cache_ball(self):
        c = self.cfg
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, c.sim_ball_body)
        self._ball_body = bid if bid >= 0 else None
        self._ball_qpos = self._joint_qpos(c.sim_ball_joint, required=False)

    def _yaw(self) -> float:
        if self.qpos_start is None: return 0.0
        s = self.qpos_start
        qw, qx, qy, qz = self.data.qpos[s + 3:s + 7]
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def send_twist(self, x, z, is_kick=False):
        """속도 명령을 qpos 변위로 적분(teleport). slew/clamp 로 부드럽게."""
        if self.qpos_start is None: return
        mx = self.cfg.kick_max_speed if is_kick else self.cfg.max_linear_speed
        fx = clamp(self.prev_x + clamp(x - self.prev_x, -self.cfg.acc_limit, self.cfg.dec_limit), -mx, mx)
        fz = clamp(self.prev_z + clamp(z - self.prev_z, -self.cfg.body_slew_rate, self.cfg.body_slew_rate),
                   -self.cfg.max_angular_speed, self.cfg.max_angular_speed)
        self.prev_x, self.prev_z = fx, fz
        s = self.qpos_start; yaw = self._yaw()
        scale = 1.0   # 시각 확인용 변위 스케일
        self.data.qpos[s] += math.cos(yaw) * fx * scale
        self.data.qpos[s + 1] += math.sin(yaw) * fx * scale
        nyaw = yaw + fz * scale
        self.data.qpos[s + 3] = math.cos(nyaw / 2); self.data.qpos[s + 6] = math.sin(nyaw / 2)
        self.data.qpos[s + 4] = 0.0; self.data.qpos[s + 5] = 0.0
        self.data.qpos[s + 2] = self.base_z

    def set_head(self, yaw, pitch):
        if self.yaw_qpos is not None: self.data.qpos[self.yaw_qpos] = float(yaw)
        if self.pitch_qpos is not None: self.data.qpos[self.pitch_qpos] = float(pitch)

    def get_base_pose(self):
        if self.qpos_start is None: return None
        s = self.qpos_start
        return float(self.data.qpos[s]), float(self.data.qpos[s + 1]), self._yaw()

    def get_ball_relative(self):
        if self._ball_qpos is None or self.qpos_start is None: return None
        bx, by = float(self.data.qpos[self._ball_qpos]), float(self.data.qpos[self._ball_qpos + 1])
        rx, ry, ryaw = self.get_base_pose()
        dx, dy = bx - rx, by - ry
        fwd = math.cos(ryaw) * dx + math.sin(ryaw) * dy
        lat = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
        return {"dist": math.hypot(dx, dy), "forward": fwd, "lateral": lat, "angle": math.atan2(lat, fwd)}

    def step(self):
        if self.cfg.step_mujoco_physics and self.model is not None:
            self.mujoco.mj_forward(self.model, self.data)


class RealStubDriver(RobotDriver):
    """실로봇 결선 지점 stub. ⚠ v11 의 모터 안전(follow-error/limit/staleness/NaN)을
    여기 send_twist/set_head 로 이식해야 실로봇에 안전하게 쓸 수 있다. v9 자체는 비전/시뮬용."""
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        print("[RealStubDriver] ⚠ 실로봇 안전 로직 미포함. v11 driver 로 교체 필요.")
    def send_twist(self, x, z, is_kick=False):
        pass  # TODO: v11 안전 + TRON1 WS
    def set_head(self, yaw, pitch):
        pass  # TODO: v11 안전 + CubeMars 모터
