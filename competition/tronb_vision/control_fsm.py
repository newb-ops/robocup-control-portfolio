#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""인식 결과와 경기 상태에 따라 로봇의 동작 상태를 결정한다."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class BallPerceptionState(str, Enum):
    """
    비전팀 vision_state.py가 robot_state["ball_perception_state"]에 써주는 값.
    str 상속이므로 robot_state.get("ball_perception_state") 결과와 == 비교 가능.
    """
    BALL_LOST               = "BALL_LOST"
    BALL_TEMP_LOST          = "BALL_TEMP_LOST"
    BALL_UNSTABLE           = "BALL_UNSTABLE"
    BALL_STABLE_OFF_CENTER  = "BALL_STABLE_OFF_CENTER"
    BALL_STABLE_CENTER_FAR  = "BALL_STABLE_CENTER_FAR"
    BALL_STABLE_CENTER_NEAR = "BALL_STABLE_CENTER_NEAR"
    BALL_TOO_CLOSE          = "BALL_TOO_CLOSE"
    UNKNOWN                 = "UNKNOWN"   # robot_state에 아직 값이 없는 경우 fallback


class VisionStateAdapter:
    """
    robot_state dict → FSM이 쓰는 정규화된 값으로 변환.
    비전팀 출력 key가 바뀌면 여기만 수정하면 된다.
    """

    # robot_state key 이름 매핑 (비전팀과 협의 후 여기서 일괄 관리)
    KEY_BALL_PERCEPTION  = "ball_perception_state"
    KEY_BALL_DIST        = "ball_dist"
    KEY_BALL_SCREEN_X    = "ball_screen_x"
    KEY_BALL_SCREEN_Y    = "ball_screen_y"           # 보조값, 없어도 됨
    KEY_KICK_HINT        = "vision_kick_candidate_hint"
    KEY_GOAL_VISIBLE     = "vision_goal_visible"
    KEY_GOAL_CENTERED    = "vision_goal_centered"
    KEY_LOCALIZATION_OK  = "vision_localization_reliable"
    KEY_LOST_FRAMES      = "lost_frames"              # 비전팀이 직접 노출 시
    KEY_CONSEC_DETECT    = "consecutive_detect"       # 비전팀이 직접 노출 시

    @classmethod
    def parse(cls, robot_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        robot_state dict에서 FSM에 필요한 값을 추출·정규화해서 반환.
        없는 key는 안전한 기본값으로 채운다.
        """
        raw_bps = robot_state.get(cls.KEY_BALL_PERCEPTION, "UNKNOWN")
        try:
            bps = BallPerceptionState(raw_bps)
        except ValueError:
            bps = BallPerceptionState.UNKNOWN

        return {
            "ball_perception_state": bps,
            "ball_dist":             float(robot_state.get(cls.KEY_BALL_DIST,    9.99)),
            "ball_screen_x":         float(robot_state.get(cls.KEY_BALL_SCREEN_X, 0.0)),
            "ball_screen_y":         float(robot_state.get(cls.KEY_BALL_SCREEN_Y, 0.0)),
            "kick_hint":             bool (robot_state.get(cls.KEY_KICK_HINT,     False)),
            "goal_visible":          bool (robot_state.get(cls.KEY_GOAL_VISIBLE,  False)),
            "goal_centered":         bool (robot_state.get(cls.KEY_GOAL_CENTERED, False)),
            "localization_ok":       bool (robot_state.get(cls.KEY_LOCALIZATION_OK, False)),
            # 비전팀이 lost_frames/consecutive_detect를 직접 노출하지 않으면 bps로 추론
            "lost_frames":           int  (robot_state.get(cls.KEY_LOST_FRAMES,   0)),
            "consecutive_detect":    int  (robot_state.get(cls.KEY_CONSEC_DETECT, 0)),
        }


class RobotState(Enum):
    """하체 PC로도 전달할 기본 행동 상태. BLIND_APPROACH는 사용하지 않는다."""

    SEARCH = 0
    TRACK = 1
    APPROACH = 2
    STOP = 3
    ALIGN_GOAL = 4
    BACK_OFF = 5
    KICK = 6
    COOLDOWN = 7
    RECOVER = 8


@dataclass
class BallObservation:
    """
    비전팀 BallFollowerVisionCore의 BallObservation과 호환되는 최소 필드.
    현재 상체 노드에 이미 같은 이름의 dataclass가 있으면 그 객체를 그대로 넣어도 된다.
    """

    area: float = 0.0
    dist: float = 9.99
    dist_from_area: float = 9.99
    screen_x: float = 0.0
    screen_y: float = 0.0
    last_dir: float = 0.0
    lost_frames: int = 999
    last_valid_dist: float = 2.0
    timestamp: float = 0.0
    detected: bool = False
    consecutive_detect: int = 0
    kalman_accepted: bool = False
    frame_id: int = 0


@dataclass
class GoalObservation:
    """골대 인식 결과. goal_data dict가 있으면 from_goal_data()로 변환해서 사용."""

    detected: bool = False
    screen_x: float = 0.0  # -1.0(left) ~ +1.0(right)
    screen_y: float = 0.0
    conf: float = 0.0
    timestamp: float = 0.0

    @staticmethod
    def from_goal_data(goal_data: Optional[Dict[str, Any]], now: Optional[float] = None) -> "GoalObservation":
        """
        다양한 비전팀 dict key를 최대한 받아들이는 보조 변환 함수.
        expected examples:
          {"norm_x": 0.1, "norm_y": -0.2, "conf": 0.8}
          {"screen_x": 0.1, "screen_y": -0.2, "confidence": 0.8}
        """
        if goal_data is None:
            return GoalObservation(detected=False, timestamp=now or time.time())

        x = goal_data.get("norm_x", goal_data.get("screen_x", goal_data.get("x", 0.0)))
        y = goal_data.get("norm_y", goal_data.get("screen_y", goal_data.get("y", 0.0)))
        conf = goal_data.get("conf", goal_data.get("confidence", 0.0))
        ts = goal_data.get("timestamp", now or time.time())

        return GoalObservation(
            detected=True,
            screen_x=float(x),
            screen_y=float(y),
            conf=float(conf),
            timestamp=float(ts),
        )


@dataclass
class LowerStatus:
    """하체 PC에서 올라오는 최소 상태. dict 입력도 update()에서 지원한다."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    is_fallen: bool = False
    timestamp: float = 0.0

    @staticmethod
    def from_pkt(pkt: Dict[str, Any]) -> "LowerStatus":
        return LowerStatus(
            roll=float(pkt.get("roll", 0.0)),
            pitch=float(pkt.get("pitch", 0.0)),
            yaw=float(pkt.get("yaw", 0.0)),
            is_fallen=bool(pkt.get("is_fallen", False)),
            timestamp=float(pkt.get("ts", time.time())),
        )


@dataclass
class ControlCommand:
    """
    FSM이 하체/상체 adapter에 넘기는 명령 후보.
    현재 UDP가 state만 보낸다면 우선 state만 쓰고, 나중에 vx/vy/wz/kick_trigger를 패킷에 추가하면 된다.
    """

    state: RobotState
    target_body_vx: float = 0.0
    target_body_vy: float = 0.0
    target_body_wz: float = 0.0
    target_head_yaw: float = 0.0
    target_head_pitch: float = -0.20
    kick_trigger: bool = False
    recover_trigger: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class StateUpdateResult:
    """update() 1회 호출 결과."""

    state: RobotState
    prev_state: RobotState
    changed: bool
    command: ControlCommand
    reason: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BasicFSMConfig:
    # vision freshness / lost
    vision_stale_timeout: float = 0.70
    lost_to_search_frames: int = 35
    track_enter_min_detect: int = 2

    # screen alignment thresholds
    track_center_x: float = 0.30
    approach_recenter_x: float = 0.45
    stop_recenter_x: float = 0.20
    align_recenter_x: float = 0.25
    kick_center_x: float = 0.12
    goal_center_x: float = 0.20
    min_goal_conf: float = 0.20

    # distance thresholds [m]
    track_to_approach_dist: float = 2.00
    stop_min_dist: float = 0.35
    stop_max_dist: float = 0.60
    too_close_dist: float = 0.30
    approach_restart_dist: float = 0.70

    # timers [s]
    recover_min_time: float = 2.00
    stop_hold_time: float = 0.40
    kick_duration: float = 1.00
    cooldown_duration: float = 2.00
    backoff_min_time: float = 0.50
    backoff_timeout: float = 2.00

    # align / kick
    align_ready_count_th: int = 4
    require_goal_for_kick: bool = True

    # safety
    fallen_roll_rad: float = 0.35
    recovered_roll_rad: float = 0.15

    # command gains / limits
    search_wz: float = 0.25
    max_wz: float = 0.45
    max_vx: float = 0.18
    close_vx: float = 0.05
    backoff_vx: float = -0.05
    yaw_gain: float = 0.45
    goal_yaw_gain: float = 0.35
    head_yaw_gain: float = 0.60
    head_pitch_base: float = -0.20
    head_pitch_gain: float = 0.25


class BasicRobotStateFSM:
    """
    기본 RobotState FSM.

    사용 예:
        fsm = BasicRobotStateFSM()
        result = fsm.update(pkt, obs, vision_fresh=True, goal_obs=goal)
        lower_state = result.state
        cmd = result.command
    """

    def __init__(self, config: Optional[BasicFSMConfig] = None):
        self.cfg = config or BasicFSMConfig()
        self.state = RobotState.SEARCH
        self._entered_at = time.time()
        self._state_enter_time: Dict[RobotState, float] = {self.state: self._entered_at}
        self._align_ready_count = 0
        self._last_transition_reason = "init"

    def set_state(self, new_state: RobotState, reason: str = "") -> bool:
        """상태를 변경하고 진입 시간을 기록한다. 변경되면 True."""
        if self.state == new_state:
            return False
        self.state = new_state
        self._entered_at = time.time()
        self._state_enter_time[new_state] = self._entered_at
        self._last_transition_reason = reason

        if new_state != RobotState.ALIGN_GOAL:
            self._align_ready_count = 0
        return True

    def time_in_state(self, now: Optional[float] = None) -> float:
        return (now or time.time()) - self._entered_at

    def update(
        self,
        lower: Dict[str, Any] | LowerStatus,
        robot_state: Dict[str, Any],
        now: Optional[float] = None,
    ) -> StateUpdateResult:
        """
        FSM 1 tick 업데이트.

        Args:
            lower:        하체 상태 dict 또는 LowerStatus.
            robot_state:  비전팀이 채워준 robot_state dict.
                          최소 필요 key:
                            ball_perception_state, ball_dist, ball_screen_x,
                            vision_kick_candidate_hint,
                            vision_goal_visible, vision_goal_centered,
                            vision_localization_reliable
            now:          테스트용 시간 주입 (None이면 time.time()).
        """
        now = now or time.time()
        status = LowerStatus.from_pkt(lower) if isinstance(lower, dict) else lower

        # 비전팀 robot_state → 정규화된 vision dict
        vis = VisionStateAdapter.parse(robot_state)

        prev   = self.state
        reason = ""

        flags = self._build_flags(vis, now)

        # 0) 최우선 안전 조건: 넘어짐
        if abs(status.roll) > self.cfg.fallen_roll_rad or status.is_fallen:
            changed = self.set_state(RobotState.RECOVER, "fallen")
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, "fallen", flags)

        # 1) RECOVER
        if self.state == RobotState.RECOVER:
            if self.time_in_state(now) > self.cfg.recover_min_time and abs(status.roll) < self.cfg.recovered_roll_rad:
                reason = "recovered -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            else:
                changed = False
                reason = "recovering"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 2) KICK
        if self.state == RobotState.KICK:
            if self.time_in_state(now) > self.cfg.kick_duration:
                reason = "kick duration done -> cooldown"
                changed = self.set_state(RobotState.COOLDOWN, reason)
            else:
                changed = False
                reason = "kicking"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 3) COOLDOWN
        if self.state == RobotState.COOLDOWN:
            if self.time_in_state(now) <= self.cfg.cooldown_duration:
                changed = False
                reason = "cooldown wait"
            elif flags["ball_lost_hard"]:
                reason = "cooldown done, ball lost -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"]:
                reason = "cooldown done, ball still close -> stop"
                changed = self.set_state(RobotState.STOP, reason)
            elif flags["ball_seen"] and vis["ball_dist"] > self.cfg.approach_restart_dist:
                reason = "cooldown done, ball visible far -> approach"
                changed = self.set_state(RobotState.APPROACH, reason)
            elif flags["ball_seen"]:
                reason = "cooldown done, ball visible -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            else:
                reason = "cooldown done -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 4) SEARCH
        if self.state == RobotState.SEARCH:
            if flags["ball_stable"]:
                reason = "stable ball -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            else:
                changed = False
                reason = "searching"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 5) TRACK
        if self.state == RobotState.TRACK:
            if flags["ball_lost_hard"]:
                reason = "ball lost -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"]:
                reason = "close centered ball -> stop"
                changed = self.set_state(RobotState.STOP, reason)
            elif flags["ball_seen"] and vis["ball_dist"] <= self.cfg.track_to_approach_dist and flags["ball_centered_track"]:
                reason = "centered ball in range -> approach"
                changed = self.set_state(RobotState.APPROACH, reason)
            else:
                changed = False
                reason = "tracking / recent lost hold"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 6) APPROACH
        if self.state == RobotState.APPROACH:
            if flags["ball_lost_hard"]:
                reason = "ball lost -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_too_close"]:
                reason = "ball too close -> back off"
                changed = self.set_state(RobotState.BACK_OFF, reason)
            elif flags["ball_seen"] and abs(vis["ball_screen_x"]) > self.cfg.approach_recenter_x:
                reason = "ball off center -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"]:
                reason = "arrived at ball -> stop"
                changed = self.set_state(RobotState.STOP, reason)
            else:
                changed = False
                reason = "approaching"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 7) STOP
        if self.state == RobotState.STOP:
            if flags["ball_lost_hard"]:
                reason = "ball lost -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_too_close"]:
                reason = "too close while stop -> back off"
                changed = self.set_state(RobotState.BACK_OFF, reason)
            elif flags["ball_seen"] and vis["ball_dist"] > self.cfg.approach_restart_dist:
                reason = "ball moved away -> approach"
                changed = self.set_state(RobotState.APPROACH, reason)
            elif flags["ball_seen"] and abs(vis["ball_screen_x"]) > self.cfg.stop_recenter_x:
                reason = "close but off center -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"] and self.time_in_state(now) > self.cfg.stop_hold_time:
                reason = "stop stable -> align goal"
                changed = self.set_state(RobotState.ALIGN_GOAL, reason)
            else:
                changed = False
                reason = "stopping / stabilizing"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 8) ALIGN_GOAL
        if self.state == RobotState.ALIGN_GOAL:
            if flags["ball_lost_hard"]:
                self._align_ready_count = 0
                reason = "ball lost while align -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_seen"] and vis["ball_dist"] > self.cfg.approach_restart_dist:
                self._align_ready_count = 0
                reason = "ball moved away while align -> approach"
                changed = self.set_state(RobotState.APPROACH, reason)
            elif flags["ball_seen"] and abs(vis["ball_screen_x"]) > self.cfg.align_recenter_x:
                self._align_ready_count = 0
                reason = "ball off center while align -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"] and flags["goal_ready_for_kick"]:
                self._align_ready_count += 1
                if self._align_ready_count >= self.cfg.align_ready_count_th:
                    self._align_ready_count = 0
                    reason = "align ready count reached -> kick"
                    changed = self.set_state(RobotState.KICK, reason)
                else:
                    changed = False
                    reason = "align count accumulating"
            else:
                self._align_ready_count = max(self._align_ready_count - 1, 0)
                changed = False
                reason = "aligning goal"
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 9) BACK_OFF
        if self.state == RobotState.BACK_OFF:
            elapsed = self.time_in_state(now)
            if elapsed < self.cfg.backoff_min_time:
                changed = False
                reason = "backoff minimum time"
            elif flags["ball_lost_hard"] or elapsed > self.cfg.backoff_timeout:
                reason = "backoff done/lost -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            elif flags["ball_stop_range"] and flags["ball_centered_kick"]:
                reason = "backoff done, ball close centered -> stop"
                changed = self.set_state(RobotState.STOP, reason)
            elif flags["ball_seen"] and vis["ball_dist"] > self.cfg.approach_restart_dist:
                reason = "backoff done, ball far -> approach"
                changed = self.set_state(RobotState.APPROACH, reason)
            elif flags["ball_seen"]:
                reason = "backoff done, ball visible -> track"
                changed = self.set_state(RobotState.TRACK, reason)
            else:
                reason = "backoff done -> search"
                changed = self.set_state(RobotState.SEARCH, reason)
            cmd = self._make_command(self.state, vis, flags, now)
            return self._result(prev, changed, cmd, reason, flags)

        # 예외: 알 수 없는 상태면 SEARCH로 복귀
        changed = self.set_state(RobotState.SEARCH, "unknown state fallback")
        cmd = self._make_command(self.state, vis, flags, now)
        return self._result(prev, changed, cmd, "unknown state fallback", flags)

    def _build_flags(
        self,
        vis: Dict[str, Any],
        now: float,
    ) -> Dict[str, Any]:
        """
        비전팀 robot_state 파싱 결과(vis)를 받아 FSM 전환 조건 flag를 만든다.

        핵심 설계 원칙:
        - ball_seen / ball_stable / ball_lost_hard 는 BallPerceptionState에서 직접 유도.
        - 거리/중앙 정렬 관련 수치 threshold는 BasicFSMConfig에서 관리.
        - vision_kick_candidate_hint는 ALIGN_GOAL→KICK 조건의 보조 신호로 사용.
          (hint가 True여도 FSM이 추가 조건을 확인 후 최종 결정)
        """
        bps  = vis["ball_perception_state"]
        dist = vis["ball_dist"]
        sx   = vis["ball_screen_x"]

        # 비전 state 기반 기본 존재 판단
        # "공이 보이는" 상태: UNSTABLE 이상. LOST / TEMP_LOST / UNKNOWN은 미검출 취급.
        ball_seen   = bps not in (
            BallPerceptionState.BALL_LOST,
            BallPerceptionState.BALL_TEMP_LOST,
            BallPerceptionState.UNKNOWN,
        )
        # "안정적으로 보임": OFF_CENTER 이상의 안정 state
        ball_stable = bps in (
            BallPerceptionState.BALL_STABLE_OFF_CENTER,
            BallPerceptionState.BALL_STABLE_CENTER_FAR,
            BallPerceptionState.BALL_STABLE_CENTER_NEAR,
            BallPerceptionState.BALL_TOO_CLOSE,
        )
        # "완전히 잃음": LOST 또는 UNKNOWN (TEMP_LOST는 아직 유지)
        ball_lost_hard = bps in (
            BallPerceptionState.BALL_LOST,
            BallPerceptionState.UNKNOWN,
        )

        # 거리 / 정렬 flag (수치 threshold는 Config에서 관리)
        ball_centered_track = ball_seen and abs(sx) <= self.cfg.track_center_x
        ball_centered_kick  = ball_seen and abs(sx) <= self.cfg.kick_center_x
        ball_stop_range     = ball_seen and self.cfg.stop_min_dist <= dist <= self.cfg.stop_max_dist
        ball_too_close      = bps == BallPerceptionState.BALL_TOO_CLOSE

        # 골대 flag: 비전팀 판단값 직접 사용
        goal_seen     = vis["goal_visible"]
        goal_centered = vis["goal_centered"]

        if self.cfg.require_goal_for_kick:
            # vision_kick_candidate_hint AND 골대 중앙 정렬 둘 다 요구
            goal_ready_for_kick = goal_centered and vis["kick_hint"]
        else:
            # 초기 실험용: kick_hint만 True이면 허용
            goal_ready_for_kick = vis["kick_hint"]

        return {
            "ball_seen":           ball_seen,
            "ball_stable":         ball_stable,
            "ball_lost_hard":      ball_lost_hard,
            "ball_centered_track": ball_centered_track,
            "ball_centered_kick":  ball_centered_kick,
            "ball_stop_range":     ball_stop_range,
            "ball_too_close":      ball_too_close,
            "goal_seen":           goal_seen,
            "goal_centered":       goal_centered,
            "goal_ready_for_kick": goal_ready_for_kick,
            "kick_hint":           vis["kick_hint"],
            "localization_ok":     vis["localization_ok"],
            "align_ready_count":   self._align_ready_count,
            # 수치값 — _make_command / _result에서 참조
            "dist":                dist,
            "screen_x":            sx,
            "goal_x":              0.0,   # vision_goal_centered로 대체됨
            "ball_perception_state": bps.value,
        }

    def _make_command(
        self,
        state: RobotState,
        vis: Dict[str, Any],
        flags: Dict[str, Any],
        now: float,
    ) -> ControlCommand:
        """현재 state와 비전 정규화값을 바탕으로 제어 명령 후보 생성."""
        notes = []
        vx    = 0.0
        vy    = 0.0
        wz    = 0.0
        kick  = False
        recover = False

        sx   = vis["ball_screen_x"]
        sy   = vis["ball_screen_y"]
        dist = vis["ball_dist"]

        # head target 기본값
        if flags["ball_seen"] and state not in (RobotState.SEARCH, RobotState.COOLDOWN, RobotState.RECOVER):
            head_yaw   = self._clip(sx * self.cfg.head_yaw_gain, -0.60, 0.60)
            head_pitch = self.cfg.head_pitch_base + self._clip(sy * self.cfg.head_pitch_gain, -0.18, 0.18)
            if dist < self.cfg.stop_max_dist:
                head_pitch += 0.12
            head_pitch = self._clip(head_pitch, -0.70, 0.30)
        elif state == RobotState.SEARCH:
            head_yaw   = 0.35 * math.sin(now * 0.7)
            head_pitch = -0.10 + 0.12 * math.sin(now * 1.3)
        else:
            head_yaw   = 0.0
            head_pitch = -0.15

        if state == RobotState.SEARCH:
            wz = self.cfg.search_wz
            notes.append("search_spin")

        elif state == RobotState.TRACK:
            wz = self._clip(sx * self.cfg.yaw_gain, -self.cfg.max_wz, self.cfg.max_wz)
            notes.append("track_ball_center")

        elif state == RobotState.APPROACH:
            if flags["ball_seen"]:
                dist_error = max(0.0, dist - self.cfg.stop_max_dist)
                vx = self._clip(0.06 + 0.08 * dist_error, 0.0, self.cfg.max_vx)
                wz = self._clip(sx * self.cfg.yaw_gain, -self.cfg.max_wz, self.cfg.max_wz)
            notes.append("approach_ball")

        elif state == RobotState.STOP:
            vx = 0.0
            wz = self._clip(sx * 0.15, -0.10, 0.10) if flags["ball_seen"] else 0.0
            notes.append("stabilize_before_align")

        elif state == RobotState.ALIGN_GOAL:
            vx = 0.0
            if flags["goal_seen"]:
                # vision_goal_centered가 True면 미세 수정, False면 탐색 회전
                wz = self._clip(sx * self.cfg.goal_yaw_gain, -0.18, 0.18) if not flags["goal_centered"] else 0.0
                notes.append("align_to_goal")
            elif flags["ball_seen"]:
                wz = self._clip(sx * 0.12, -0.10, 0.10)
                notes.append("hold_ball_wait_goal")
            else:
                wz = 0.0
                notes.append("align_no_target")

        elif state == RobotState.BACK_OFF:
            vx = self.cfg.backoff_vx
            wz = self._clip(sx * 0.10, -0.10, 0.10) if flags["ball_seen"] else 0.0
            notes.append("back_off")

        elif state == RobotState.KICK:
            kick = True
            notes.append("kick_trigger")

        elif state == RobotState.COOLDOWN:
            notes.append("cooldown")

        elif state == RobotState.RECOVER:
            recover = True
            notes.append("recover_trigger")

        return ControlCommand(
            state=state,
            target_body_vx=float(vx),
            target_body_vy=float(vy),
            target_body_wz=float(wz),
            target_head_yaw=float(head_yaw),
            target_head_pitch=float(head_pitch),
            kick_trigger=kick,
            recover_trigger=recover,
            notes=tuple(notes),
        )

    def _result(
        self,
        prev: RobotState,
        changed: bool,
        cmd: ControlCommand,
        reason: str,
        flags: Dict[str, Any],
    ) -> StateUpdateResult:
        debug = dict(flags)
        debug["time_in_state"] = self.time_in_state()
        debug["last_transition_reason"] = self._last_transition_reason
        return StateUpdateResult(
            state=self.state,
            prev_state=prev,
            changed=changed,
            command=cmd,
            reason=reason,
            debug=debug,
        )

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))


# 간단 단독 테스트용. 실제 ROS 노드에서는 import해서 사용한다.
if __name__ == "__main__":
    fsm = BasicRobotStateFSM(config=BasicFSMConfig(require_goal_for_kick=False))
    lower = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0, "is_fallen": False}

    print("=== 시나리오 1: 공이 멀리서 점점 가까워지는 상황 ===")
    scenarios = [
        # (ball_perception_state,             ball_dist, ball_screen_x, kick_hint, goal_visible, goal_centered)
        ("BALL_LOST",                          9.99,     0.00,          False,     False, False),
        ("BALL_TEMP_LOST",                     9.99,     0.00,          False,     False, False),
        ("BALL_STABLE_OFF_CENTER",             1.80,     0.40,          False,     False, False),
        ("BALL_STABLE_CENTER_FAR",             1.60,     0.05,          False,     False, False),
        ("BALL_STABLE_CENTER_FAR",             1.00,     0.03,          False,     False, False),
        ("BALL_STABLE_CENTER_NEAR",            0.50,     0.02,          False,     True,  False),
        ("BALL_STABLE_CENTER_NEAR",            0.45,     0.01,          True,      True,  True),
        ("BALL_STABLE_CENTER_NEAR",            0.42,     0.01,          True,      True,  True),
        ("BALL_STABLE_CENTER_NEAR",            0.40,     0.01,          True,      True,  True),
        ("BALL_STABLE_CENTER_NEAR",            0.38,     0.01,          True,      True,  True),
        ("BALL_TOO_CLOSE",                     0.14,     0.00,          False,     True,  True),
    ]

    for i, (bps, dist, sx, kick_hint, gv, gc) in enumerate(scenarios):
        robot_state = {
            "ball_perception_state":      bps,
            "ball_dist":                  dist,
            "ball_screen_x":              sx,
            "ball_screen_y":              0.0,
            "vision_kick_candidate_hint": kick_hint,
            "vision_goal_visible":        gv,
            "vision_goal_centered":       gc,
            "vision_localization_reliable": True,
        }
        out = fsm.update(lower, robot_state)
        print(
            f"tick={i:2d}  bps={bps:<28s}  "
            f"state={out.state.name:<12s}  changed={str(out.changed):<5s}  "
            f"reason={out.reason}"
        )
        time.sleep(0.05)
