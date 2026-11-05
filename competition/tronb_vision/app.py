from __future__ import annotations
"""인식 결과를 모아 공 추적과 동작 제어를 실행한다.

시뮬레이션은 MuJoCo 동시 접근을 피하려고 단일 루프를 사용한다.
스레드 모드에서는 인식과 제어를 분리하고, 제어는 최신 관측값만 읽는다.
OperationMode는 운용 모드, RobotState는 동작 상태를 나타낸다."""

import math
import threading
import time
from typing import Optional

import numpy as np

from .common import (BallObservation, GoalObservation, GameControllerState,
                     OperationMode, RobotConfig, RobotState, TriState,
                     TuningParams, clamp, now_sec)
from .detector import BaseDetector, create_detector, resolve_class_ids
from .drivers import DryRunDriver, RobotDriver
from .features import (BallStopEstimator, attach_ball_stop_to_robot_state,
                       build_vision_packet)
from .game_controller import game_state_flags, normalize_game_controller_state
from .localization import LocalizationEngine, Observation
from .real_position import RealPositionEstimator
from .rule_state import build_rule_state
from .setplay import SetPlayVisionAdapter, SetPlayVisionConfig
from .sources import CameraSource, DummySource, FramePacket
from .vision_core import AdaptiveHeadController, HeadStabilizer, VisionCore
from .vision_state import VisionStateConfig, build_rule_based_vision_packet

# detector 는 항상 ON (토글 없음). 토글 가능한 기능:
# localization / ball_stop : 비전 추정 기능
# rule_state / vision_state / setplay: 제어 FSM용 상태 생성
FEATURES = ["localization", "ball_stop", "rule_state", "vision_state", "setplay"]


class BallFollower:
    def __init__(self, cfg: RobotConfig, tuning: TuningParams,
                 source: CameraSource, driver: RobotDriver,
                 detector: Optional[BaseDetector] = None,
                 bgr_input: bool = True):
        self.cfg = cfg; self.t = tuning
        self.source = source; self.driver = driver
        self.detector = detector if detector is not None else create_detector(cfg)
        self.bgr_input = bgr_input   # sim 렌더는 RGB → BGR 변환 필요

        cmap = resolve_class_ids(getattr(self.detector, "names", {}) or {})
        self.cmap = cmap
        self.vision = VisionCore(cfg, tuning, cmap)
        self.loc = LocalizationEngine(cfg)
        self.head = AdaptiveHeadController()
        self.head_stab = HeadStabilizer(cfg.imu_compensation_gain)

        self.ball_stop = BallStopEstimator(cfg.camera_fov_h)
        self.real_pos = (RealPositionEstimator(cfg, getattr(self.detector, "names", {}) or {})
                         if bool(getattr(cfg, "real_position_enabled", False)) and not bgr_input
                         else None)
        self.feat_out = {}
        # 인식·세트플레이 상태를 제어 FSM에 전달한다.
        self.robot_state: dict = {}
        self.vision_state_cfg = VisionStateConfig()
        # restart_type/owner 와 동일하게 GameController/상위 로직이 주입. 미주입=UNKNOWN(permissive).
        self.game_controller_state = normalize_game_controller_state(
            getattr(cfg, "game_controller_state", None))
        self.rule_state_snapshot = None
        self.vision_packet = None
        self.setplay_adapter = SetPlayVisionAdapter(SetPlayVisionConfig())
        self.setplay_packet = None
        # external_control=True면 control_fsm, False면 내부 FSM을 사용한다.
        self.external_control = bool(getattr(cfg, "external_control", False))
        self.control_bridge = None
        if self.external_control:
            from .integration import ControlBridge
            self.control_bridge = ControlBridge(
                self.driver,
                jsonl_path=getattr(cfg, "control_log_path", None),
                verbose=True)
        self._manual_twist = (0.0, 0.0)   # MANUAL 모드 UI 조작용
        self._last_bgr = None             # UI 표시용 (perceive 가 쓴 프레임)
        self._last_depth = None
        self._last_boxes = []
        self._last_landmark_obs = []      # top-view 표시용

        self.mode = OperationMode.LOCKED       # 기본 LOCKED (안전)
        self.state = RobotState.SEARCH
        self.features = {f: TriState.ON for f in FEATURES}
        self.features["localization"] = TriState.OFF if not cfg.localization_enabled else TriState.SEMI

        self._latest_obs = BallObservation()
        self._latest_goal = GoalObservation()
        self._lock = threading.Lock()
        self._run = False
        self._t_start = now_sec()

        # FSM 타이머/카운터
        self.stop_confirm = 0; self.track_frames = 0
        self.aim_start = 0.0; self.kick_start = 0.0; self.kick_settling = False; self.kick_settle_start = 0.0
        self.cooldown_start = 0.0; self.recover_start = 0.0
        self._prev_yaw = 0.0; self._prev_t = now_sec()
        self._prev_x = 0.0; self._prev_y = 0.0   # odom (sim base pose delta)

    # 모드/토글
    def set_mode(self, m: OperationMode):
        if m != self.mode:
            print(f"[MODE] {self.mode.name} -> {m.name}")
            self.mode = m
            if m == OperationMode.AUTO:
                self.set_state(RobotState.SEARCH)

    def toggle_feature(self, key: str):
        if key in self.features:
            self.features[key] = self.features[key].next()
            if key == "localization":
                self.loc.enabled = self.features[key].enabled
            print(f"[FEATURE] {key} -> {self.features[key].name}")

    def set_state(self, s: RobotState):
        if s != self.state:
            print(f"{self.state.name} -> {s.name}")
            self.state = s
            if s == RobotState.AIM:
                self.aim_start = now_sec()
            elif s == RobotState.KICK:
                self.kick_settling = True; self.kick_settle_start = now_sec()
            elif s == RobotState.TRACK:
                self.track_frames = 0
            elif s == RobotState.SEARCH:
                self.vision.tracker.reset(); self.vision.kalman.reset()

    def reset_localization(self, mode: str = "global") -> None:
        """PF를 초기화한다. 기본은 전체 필드, own_half는 우리 진영에 분포한다.
        (x, y, theta) 튜플을 받으면 해당 위치 주변으로 초기화한다."""
        if isinstance(mode, (tuple, list)) and len(mode) == 3:
            self.loc.seed_pose(*mode)
            print(f"[BF] reset_localization seed {tuple(round(v,2) for v in mode)} loc_enabled={self.loc.enabled}")
        elif mode == "own_half":
            self.loc.reset_own_half()
            print(f"[BF] reset_localization own_half loc_enabled={self.loc.enabled}")
        else:
            self.loc.reset_uniform()
            print(f"[BF] reset_localization GLOBAL(uniform) loc_enabled={self.loc.enabled}")

    # 헬퍼
    def _imu_yaw(self) -> float:
        bp = self.driver.get_base_pose()
        if bp is not None:
            return bp[2]
        return self.driver.read_imu()[2]

    def _to_bgr(self, rgb):
        if not self.bgr_input:
            return rgb
        try:
            import cv2
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return rgb[:, :, ::-1].copy()

    def _odom(self, imu_yaw: float):
        """PF predict 입력 (dx,dy,dtheta, 로봇 프레임).
        sim: base pose 변위를 로봇프레임으로 환산. 실로봇: odom 없어 회전(IMU)만(전진은 noise 흡수)."""
        now = now_sec(); dt = max(1e-3, now - self._prev_t)
        bp = self.driver.get_base_pose()
        dtheta = (imu_yaw - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi
        if bp is not None:
            dxw = bp[0] - self._prev_x; dyw = bp[1] - self._prev_y
            c, s = math.cos(-self._prev_yaw), math.sin(-self._prev_yaw)
            dxr = c * dxw - s * dyw; dyr = s * dxw + c * dyw
            self._prev_x, self._prev_y, self._prev_yaw, self._prev_t = bp[0], bp[1], imu_yaw, now
            return (dxr, dyr, dtheta)
        self._prev_yaw, self._prev_t = imu_yaw, now
        return (0.0, 0.0, dtheta)

    def _landmark_obs(self, boxes, depth, img_w, img_h):
        """검출 클래스와 바닥 투영 좌표를 PF 관측값으로 변환한다."""
        if self.real_pos is not None:
            return self.real_pos.landmark_observations(boxes, depth)
        from .localization import YOLO_CLS_TO_OBS_TYPE
        out = []
        # detector.names({id: name}) 로 id→obs_type 매핑 구성
        type_by_id = {}
        for cid, cname in (getattr(self.detector, "names", {}) or {}).items():
            obs_type = YOLO_CLS_TO_OBS_TYPE.get(str(cname).strip().lower())
            if obs_type is not None:
                type_by_id[int(cid)] = obs_type
        # names 없는 경우 cmap 폴백 (DummyDetector 등)
        if not type_by_id:
            for t in ("line_l", "line_t", "line_x"):
                for i in self.cmap.get(t, ()):
                    type_by_id[i] = {"line_l": "L", "line_t": "T", "line_x": "X"}[t]
            for i in self.cmap.get("goal", ()):
                type_by_id[i] = "goalpost"
            for i in self.cmap.get("point", ()):
                type_by_id[i] = "center_mark"   # POINT_ALIASES fallback
        for b in boxes:
            cid = int(b.cls.cpu().numpy()[0]) if hasattr(b.cls, "cpu") else int(b.cls)
            obs_type = type_by_id.get(cid)
            if obs_type is None:
                continue
            v = b.xywh.cpu().numpy()[0] if hasattr(b.xywh, "cpu") else np.asarray(b.xywh)[0]
            # 마크류(center_mark/penalty_mark)는 박스 중앙, 라인류는 바닥 접점
            if obs_type in ("center_mark", "penalty_mark"):
                u, vv = float(v[0]), float(v[1])
            else:
                u, vv = float(v[0]), float(v[1] + v[3] / 2)
            d = self.vision.get_depth(depth, u, vv)
            floor = self.loc.transform.pixel_to_floor_direct(u, vv, d) if d else None
            if floor:
                out.append(Observation(obs_type, floor[0], floor[1]))
        return out[:6]

    # 인지
    def _perceive(self, pkt: FramePacket):
        imu_yaw = self._imu_yaw()
        bgr = self._to_bgr(pkt.rgb)
        self._last_bgr = bgr   # UI 표시용
        self._last_depth = pkt.depth
        if self.real_pos is not None:
            self.real_pos.update_packet(pkt)
        boxes = self.detector.predict_boxes(bgr)   # detector 상시 ON
        self._last_boxes = list(boxes)
        obs, goal_data, _ = self.vision.process(boxes, pkt.depth, self.cfg.img_width,
                                                self.cfg.img_height, imu_yaw,
                                                self.head.filtered_yaw)
        goal = GoalObservation(detected=goal_data is not None,
                               screen_x=goal_data["norm_x"] if goal_data else 0.0,
                               area=goal_data["area"] if goal_data else 0.0,
                               width=goal_data["width"] if goal_data else 0.0,
                               conf=goal_data["conf"] if goal_data else 0.0)

        feat = {}
        ball_xy = None; ball_method = None
        if self.real_pos is not None:
            ball_xy, ball_method = self.real_pos.ball_robot_xy(boxes, pkt.depth, self.cmap.get("ball", ()))
            feat["real_position"] = self.real_pos.last_snap
            # 공 관측값과 별도로 RealSense 바닥 좌표도 제공한다.
            if ball_xy is not None and self.features["ball_stop"].enabled:
                feat["real_ball_xy"] = {"x": ball_xy[0], "y": ball_xy[1], "method": ball_method}

        # 위치추정이 켜졌을 때만 랜드마크를 투영한다.
        need_lm = self.loc.enabled and self.features["localization"].enabled
        landmark_obs = self._landmark_obs(boxes, pkt.depth, self.cfg.img_width, self.cfg.img_height) if need_lm else []
        self._last_landmark_obs = landmark_obs

        # localization
        loc_reliable = False
        if self.loc.enabled and self.features["localization"].enabled:
            self.loc.update(landmark_obs, self._odom(imu_yaw))
            loc_reliable = self.loc.trust()

        # ball_stop (공 정지/속도, localization 불필요)
        motion = None
        if self.features["ball_stop"].enabled:
            motion = self.ball_stop.update(obs)
            feat["ball_stop"] = motion

        self._update_robot_state(obs, goal, loc_reliable, ball_xy, motion)

        with self._lock:
            self._latest_obs = obs; self._latest_goal = goal
            self.feat_out = feat
        return obs

    def _update_robot_state(self, obs, goal, loc_reliable, ball_xy, motion):
        rs_dict = {}   # 로컬에 채운 뒤 lock 으로 통째 스왑 (control 스레드가 일관된 스냅샷 읽도록)
        # 공 월드좌표: localization 신뢰 + 로봇기준 공좌표 있을 때만.
        bwx = bwy = None
        if loc_reliable and ball_xy is not None:
            x, y, th = self.loc.pose
            c, s = math.cos(th), math.sin(th)
            bwx = x + c * ball_xy[0] - s * ball_xy[1]
            bwy = y + s * ball_xy[0] + c * ball_xy[1]
        rs_dict.update({
            "ball_dist": float(obs.dist),
            "ball_screen_x": float(obs.screen_x),
            "raw_ball_offset_y": float(obs.screen_y),
            "ball_seen_streak": int(obs.consecutive_detect),
            "ball_lost_streak": int(obs.lost_frames),
            "vision_goal_visible": bool(goal.detected),
            "vision_goal_centered": bool(goal.detected and abs(goal.screen_x) < 0.15),
            "goal_screen_x": float(goal.screen_x) if goal.detected else None,
            "frame_id": int(obs.frame_id),
            "vision_localization_reliable": bool(loc_reliable),
            "ball_world_x_est": bwx,
            "ball_world_y_est": bwy,
        })
        # ball_stop → robot_state
        attach_ball_stop_to_robot_state(rs_dict, motion)

        # rule_state (공 룰 상태)
        if self.features["rule_state"].enabled:
            rstate = build_rule_state(bwx, bwy, self.loc.fm)
            self.rule_state_snapshot = rstate
            rs_dict.update({f"vision_{k}": v for k, v in rstate.as_dict().items()})
        else:
            self.rule_state_snapshot = None

        # vision_state (공 인식 라벨 + kick 힌트)
        if self.features["vision_state"].enabled:
            gc_state = normalize_game_controller_state(self.game_controller_state)
            locomotion, active = game_state_flags(gc_state)
            ball_rule_status = (self.rule_state_snapshot.ball_rule_status
                                if self.rule_state_snapshot else
                                build_rule_state(None, None, self.loc.fm).ball_rule_status)
            self.vision_packet = build_rule_based_vision_packet(
                obs=obs, goal=goal, localization_reliable=loc_reliable,
                ball_world_x=bwx, ball_world_y=bwy, ball_rule_status=ball_rule_status,
                game_controller_state=gc_state, game_allows_locomotion=locomotion,
                game_allows_active_play=active, cfg=self.vision_state_cfg)
            rs_dict["vision_packet"] = self.vision_packet
            rs_dict["vision_packet_dict"] = self.vision_packet.as_dict()
            rs_dict["ball_perception_state"] = self.vision_packet.ball_state.name
            rs_dict["vision_kick_candidate_hint"] = self.vision_packet.vision_kick_candidate_hint
        else:
            self.vision_packet = None
        # setplay 용 경량 packet (simple) — setplay adapter 와 별개로 항상 유지
        rs_dict["simple_vision_packet"] = build_vision_packet(
            obs=obs, goal_visible=goal.detected,
            goal_centered=bool(goal.detected and abs(goal.screen_x) < 0.15),
            motion=motion, localization_reliable=loc_reliable)

        # setplay (penalty/corner/goal kick context). restart_type/owner 는 GC(gc_state) 머지값
        # 또는 sim 테스트용 cfg. 둘 다 없으면 NONE → setplay INACTIVE.
        if self.features["setplay"].enabled:
            rtype = rs_dict.get("restart_type", getattr(self.cfg, "restart_type", None))
            rowner = rs_dict.get("restart_owner", getattr(self.cfg, "restart_owner", None))
            self.setplay_packet = self.setplay_adapter.update_robot_state(
                rs_dict, restart_type=rtype, restart_owner=rowner)
        else:
            self.setplay_packet = None
        with self._lock:
            self.robot_state = rs_dict

    def _robot_state_snapshot(self) -> dict:
        """control 스레드가 읽는 robot_state 스냅샷 (lock 보호)."""
        with self._lock:
            return self.robot_state


    # 위치추정 오차가 킥 정렬에 섞이지 않도록 카메라로 본 골 방향을 우선한다.
    # 골이 안 보일 때만 추정 위치로 방향을 구한다.
    def _goal_dir(self, imu_yaw):
        gd = self.vision.goal_memory.get_goal_direction(imu_yaw)
        if self.vision.goal_memory.is_valid():
            return gd, True
        d = self.loc.goal_direction()
        if d is not None:
            return d, True
        return gd, False

    # 제어 FSM
    def _control(self, dt: float):
        with self._lock:
            obs = self._latest_obs
        imu_yaw = self._imu_yaw()
        turn = -1 if self.cfg.body_turn_inverted else 1

        if self.mode == OperationMode.LOCKED:
            self.driver.send_twist(0.0, 0.0)   # 명령 차단(정지 유지)
            return
        if self.mode == OperationMode.MANUAL:
            self.driver.send_twist(*self._manual_twist)   # UI 키보드 조작만
            return

        # 외부 FSM 경로에서는 내부 FSM을 건너뛴다.
        if self.external_control and self.control_bridge is not None:
            lower = {"is_fallen": False, "pitch": 0.0, "roll": 0.0, "yaw": imu_yaw}
            self.control_bridge.step(self._robot_state_snapshot(), lower)
            return

        # 이하 내부 FSM (단독 비전 테스트 기본 경로)

        # head stabilize
        bp = self.driver.get_base_pose()
        body_roll = 0.0; body_pitch = 0.0
        yaw_comp, pitch_comp = self.head_stab.update(body_roll, body_pitch)

        # 지연 보상 거리
        age = max(0.0, now_sec() - obs.timestamp)
        dist_comp = max(0.05, obs.dist - 0.0 * (age + self.t.perception_latency))  # sim 전진=teleport라 보상 0

        # 상태 전이 (감지/상실)
        if obs.detected and obs.consecutive_detect >= self.cfg.min_consecutive_detect and self.state == RobotState.SEARCH:
            self.set_state(RobotState.TRACK)
        if (not obs.detected and obs.lost_frames > self.t.lost_frames_threshold
                and self.state in (RobotState.TRACK, RobotState.APPROACH)):
            if self.state == RobotState.APPROACH and obs.last_valid_dist < self.t.stop_dist:
                self.set_state(RobotState.AIM)
            else:
                self.set_state(RobotState.SEARCH)

        vx = wz = 0.0
        st = self.state
        yaw_cmd = self.head.filtered_yaw

        if st == RobotState.SEARCH:
            self.stop_confirm = 0
            vx = self.t.search_drift_compensation
            phase = (2 * math.pi / self.t.search_turn_period) * (now_sec() - self._t_start)
            wz = self.t.search_turn_speed * math.sin(phase) + self.t.search_turn_bias
            self.head.set_target_direct(0.0, self.t.search_head_pitch_center + 0.6 * math.sin(now_sec() * 2.0))

        elif st == RobotState.TRACK:
            self.track_frames += 1
            yaw_err, _ = self.vision.screen_to_angle(obs.screen_x, 0.0)
            self.head.set_target(clamp(self.head.target_yaw + 0.3 * yaw_err, *self.cfg.head_yaw_limit),
                                 self.head.target_pitch)
            if abs(yaw_cmd) > self.cfg.body_turn_deadzone:
                wz = clamp(-turn * yaw_cmd * self.cfg.body_turn_gain, -0.35, 0.35)
            if (abs(yaw_cmd) < self.cfg.track_to_approach_yaw and self.track_frames >= self.cfg.track_min_frames
                    and obs.dist < self.cfg.track_min_dist):
                self.set_state(RobotState.APPROACH)

        elif st == RobotState.APPROACH:
            if abs(yaw_cmd) > self.cfg.approach_to_track_yaw:
                self.set_state(RobotState.TRACK)
            else:
                vx = self.t.approach_speed_far if obs.dist > 1.0 else self.t.approach_speed_near
                base_turn = -turn * yaw_cmd * 0.70
                curve = 0.0
                if self.t.curved_approach_enabled and obs.dist < self.t.curved_approach_dist:
                    gd, gv = self._goal_dir(imu_yaw)
                    if gv:
                        curve = -np.sign(gd) * self.t.curved_approach_gain
                wz = base_turn + self.t.approach_drift_correction + curve
                yaw_err, _ = self.vision.screen_to_angle(obs.screen_x, 0.0)
                self.head.set_target(clamp(self.head.target_yaw + 0.2 * yaw_err, *self.cfg.head_yaw_limit),
                                     self.cfg.approach_pitch_near)
                if dist_comp < self.t.stop_dist and abs(yaw_cmd) < 0.15:
                    self.stop_confirm += 1
                else:
                    self.stop_confirm = max(0, self.stop_confirm - 1)
                if self.stop_confirm >= self.t.stop_confirm_frames:
                    self.stop_confirm = 0; self.set_state(RobotState.AIM)

        elif st == RobotState.AIM:
            elapsed = now_sec() - self.aim_start
            vx = self.t.search_drift_compensation
            gd, have = self._goal_dir(imu_yaw)
            if elapsed < self.t.aim_settle_time:
                pass
            elif not have and elapsed < self.t.aim_settle_time + self.t.aim_glance_time:
                self.head.set_target_direct(0.5 * math.sin(elapsed * 3.0), self.t.aim_look_pitch)
            else:
                aligned = abs(gd) < self.t.align_good_angle if have else False
                if aligned or elapsed > self.t.aim_timeout:
                    self.set_state(RobotState.KICK)
                else:
                    wz = clamp(-turn * gd * (self.t.align_turn_speed / 0.25),
                               -self.t.align_turn_speed, self.t.align_turn_speed)
                    self.head.set_target(0.0, self.cfg.approach_pitch_near)

        elif st == RobotState.KICK:
            if self.kick_settling:
                if now_sec() - self.kick_settle_start >= self.t.kick_settle_time:
                    self.kick_settling = False; self.kick_start = now_sec()
                else:
                    vx = wz = 0.0
            else:
                vx = self.t.kick_speed; wz = self.t.straight_drift_correction
                if now_sec() - self.kick_start > 1.5:
                    self.cooldown_start = now_sec(); self.set_state(RobotState.COOLDOWN)

        elif st == RobotState.COOLDOWN:
            vx = -0.02
            if now_sec() - self.cooldown_start > self.cfg.cooldown_time:
                self.set_state(RobotState.SEARCH)

        elif st == RobotState.RECOVER:
            if now_sec() - self.recover_start > self.cfg.recover_time:
                self.set_state(RobotState.SEARCH)

        self.driver.send_twist(vx, wz, is_kick=(st == RobotState.KICK and not self.kick_settling))
        yaw_f, pitch_f = self.head.get_command()
        self.driver.set_head(clamp(yaw_f + yaw_comp, *self.cfg.head_yaw_limit),
                             clamp(pitch_f + pitch_comp, *self.cfg.head_pitch_limit))

    # 한 틱 (sim/selftest)
    def tick(self) -> Optional[BallObservation]:
        pkt = self.source.read()
        if pkt is None:
            return None
        obs = self._perceive(pkt)
        now = now_sec(); dt = now - getattr(self, "_last_tick", now); self._last_tick = now
        self._control(dt)
        self.driver.step()
        return obs

    # 실행 모드
    def run_single(self, max_iters: Optional[int] = None, ui=None):
        """시뮬/headless 단일 루프."""
        self.source.start(); self.driver.start()
        self._run = True; n = 0
        try:
            while self._run:
                self.tick()
                if ui is not None:
                    ui(self)
                n += 1
                if max_iters and n >= max_iters:
                    break
        finally:
            self.driver.send_twist(0.0, 0.0); self.source.stop(); self.driver.stop()

    def start_threads(self, control_hz: float = 50.0):
        """vision/control 스레드만 시작하고 즉시 반환 (UI 는 메인스레드에서 별도로)."""
        self.source.start(); self.driver.start()
        self._run = True

        def vision_loop():
            while self._run:
                pkt = self.source.read()
                if pkt is not None:
                    try:
                        self._perceive(pkt)
                    except Exception as e:
                        print(f"[vision] {e}")
                time.sleep(0.001)

        def control_loop():
            period = 1.0 / control_hz
            while self._run:
                t0 = now_sec()
                try:
                    self._control(period)
                except Exception as e:
                    self.driver.send_twist(0.0, 0.0); print(f"[control] {e}")
                dt = now_sec() - t0
                if dt < period:
                    time.sleep(period - dt)

        threading.Thread(target=vision_loop, daemon=True).start()
        threading.Thread(target=control_loop, daemon=True).start()

    def run_threaded(self, control_hz: float = 50.0):
        """Jetson 헤드리스: vision/control 분리 후 블로킹 대기."""
        self.start_threads(control_hz)
        try:
            while self._run:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._run = False
        try:
            self.driver.send_twist(0.0, 0.0); self.source.stop(); self.driver.stop()
        except Exception:
            pass
        if self.control_bridge is not None:
            self.control_bridge.close()

    def reset_vision_tracks(self) -> None:
        self.vision.tracker.reset()
        self.vision.kalman.reset()
        if hasattr(self.ball_stop, "reset"):
            self.ball_stop.reset()
        print("[BF] vision tracks reset")

    def capture_debug_frame(self, capture_dir: Optional[str] = None) -> Optional[str]:
        if self._last_bgr is None:
            return None
        from pathlib import Path
        import cv2
        import numpy as np
        out_dir = Path(capture_dir or getattr(self.cfg, "capture_dir", "captures_camera")).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        color_path = out_dir / f"color_{stamp}.jpg"
        cv2.imwrite(str(color_path), self._last_bgr)
        if self._last_depth is not None:
            np.save(out_dir / f"depth_{stamp}.npy", self._last_depth)
        print(f"[BF] captured {color_path}")
        return str(color_path)

    # 디버그 조회
    def snapshot(self):
        with self._lock:
            return self._latest_obs, self._latest_goal
