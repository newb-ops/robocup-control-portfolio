#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""인식, GameController와 세트플레이 상태를 실기 동작 상태로 연결한다."""
from typing import Any, Dict, Optional, Tuple

from .control_fsm import BasicRobotStateFSM
from .control_fsm import RobotState as CtrlState
from .game_controller import game_state_flags, normalize_game_controller_state
from .setplay import SetPlayVisionAdapter, SetPlayVisionConfig

# control_fsm RobotState(9개) → ball_follower RobotState 이름(7개) 매핑.
# ball_follower 에 없는 STOP/ALIGN_GOAL/BACK_OFF → AIM(정지 후 골 조준)으로.
_CTRL_TO_BF = {
    "SEARCH": "SEARCH",
    "TRACK": "TRACK",
    "APPROACH": "APPROACH",
    "STOP": "AIM",          # 도착·정지 안정화 → ball_follower AIM(정지/조준)
    "ALIGN_GOAL": "AIM",    # 골 정렬 → AIM
    "KICK": "KICK",
    "BACK_OFF": "AIM",      # 너무 가까움(후진). ball_follower 후진 동작 없음 → AIM(정지)로 안전 처리
    "RECOVER": "RECOVER",
    "COOLDOWN": "COOLDOWN",
}


class BrainDecision:
    __slots__ = ("state", "force_stop", "suppress_kick", "reason", "control_state",
                 "setplay_state", "game_state", "command")

    def __init__(self, state, force_stop, suppress_kick, reason,
                 control_state, setplay_state, game_state, command):
        self.state = state                    # ball_follower RobotState 이름(str)
        self.force_stop = force_stop          # True 면 caller 가 twist(0,0)
        self.suppress_kick = suppress_kick    # True 면 kick 금지
        self.reason = reason
        self.control_state = control_state    # control_fsm 원 상태(디버그)
        self.setplay_state = setplay_state
        self.game_state = game_state
        self.command = command                # control_fsm ControlCommand 또는 None


class RealRobotBrain:
    """robot_state → (ball_follower state, force_stop, suppress_kick) 결정."""

    def __init__(self, fsm: Optional[BasicRobotStateFSM] = None,
                 setplay: Optional[SetPlayVisionAdapter] = None):
        self.fsm = fsm or BasicRobotStateFSM()
        self.setplay = setplay or SetPlayVisionAdapter(SetPlayVisionConfig())

    def decide(self, robot_state: Dict[str, Any],
               lower: Optional[Dict[str, Any]] = None,
               now: Optional[float] = None) -> BrainDecision:
        lower = lower or {"is_fallen": False}
        gcs = normalize_game_controller_state(robot_state.get("game_controller_state"))
        locomotion, active = game_state_flags(gcs)
        setplay_state = str(robot_state.get("setplay_vision_state", "INACTIVE"))

        def _decide_general():
            """control_fsm 일반 play 결정."""
            res = self.fsm.update(lower, robot_state, now=now)
            return _CTRL_TO_BF.get(res.state.name, "SEARCH"), res.state.name, res

        # 1) PENALIZED → 정지
        if gcs.name == "PENALIZED":
            return BrainDecision("AIM", True, True, "penalized -> stop",
                                 None, setplay_state, gcs.name, None)

        # 2) 이동 불가 게임상태(SET/INITIAL/FINISHED 등). PLAYING/READY 는 locomotion 허용.
        if locomotion is False:
            return BrainDecision("AIM", True, True, f"game state {gcs.name} -> hold",
                                 None, setplay_state, gcs.name, None)

        # 3) setplay 진행 중
        if setplay_state not in ("INACTIVE", "CONTEXT_UNKNOWN"):
            if setplay_state in ("DEFENSE_BALL_NOT_READY", "DEFENSE_WAIT_RESTART"):
                return BrainDecision("AIM", True, True, "opponent restart wait -> hold",
                                     None, setplay_state, gcs.name, None)
            if setplay_state in ("ATTACK_BALL_NOT_READY", "ATTACK_TARGET_NOT_READY"):
                return BrainDecision("AIM", True, True, f"our restart setup ({setplay_state}) -> hold",
                                     None, setplay_state, gcs.name, None)
            # DEFENSE_BALL_MOVED_CANDIDATE(공 움직임) / ATTACK_CONTEXT_READY(준비됨) → 일반 play 해제
            state, cstate, res = _decide_general()
            # active play 불가 상태(READY 등)면 킥만 억제
            return BrainDecision(state, False, active is False, f"setplay {setplay_state} -> play",
                                 cstate, setplay_state, gcs.name, res.command)

        # 4) 일반 play
        state, cstate, res = _decide_general()
        return BrainDecision(state, False, active is False, "general play",
                             cstate, setplay_state, gcs.name, res.command)
