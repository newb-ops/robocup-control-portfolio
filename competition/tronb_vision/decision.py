from __future__ import annotations
"""
decision.py — 통합 결정 엔진 (실전 ball_follower 가 호출).

한 프레임에 여러 결정원이 있다. 우선순위로 하나의 body 명령을 만든다:
  1) PENALIZED (penalty 받음) → 무조건 정지. 최우선.
  2) setplay ≠ INACTIVE (세트플레이 진행) → setplay 규칙. 상대 킥 대기 중 공 접근 금지 등.
  3) 그 외 → control_fsm (일반 play: 추적/접근/킥).

⚠ 이 우선순위/매핑은 합리적 기본값이다. 실기에서 튜닝 필요(주석 [TUNE] 참고).
   head 제어는 여기서 안 한다 — ball_follower 의 검증된 head 추종을 control_loop 이 exec_state 로 처리.
   여기선 body(vx,wz) + kick + exec_state(팔스윙/kick 포즈 게이팅용)만 결정.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .control_fsm import BasicRobotStateFSM, BasicFSMConfig

# setplay 가 '대기/미준비' 라 공에 접근하면 안 되는 상태들.
_SETPLAY_HOLD = {
    "ATTACK_BALL_NOT_READY",     # 우리 킥인데 공 아직 안 놓임 → 제자리 대기
    "ATTACK_TARGET_NOT_READY",   # 우리 킥, 공은 됐는데 골/타겟 context 미확보 → 대기
    "DEFENSE_BALL_NOT_READY",    # 상대 킥, 공 인식 안됨 → 대기
    "DEFENSE_WAIT_RESTART",      # 상대 킥 대기 — 룰상 공에서 거리 유지, 접근 금지
    "CONTEXT_UNKNOWN",           # restart 정보 불명 → 보수적으로 대기
}
# setplay 가 '진행 가능' (control_fsm 에게 위임).
_SETPLAY_GO = {
    "INACTIVE",                  # 세트플레이 아님 → 일반 play
    "ATTACK_CONTEXT_READY",      # 우리 킥 준비 완료 → 접근/킥 진행
    "DEFENSE_BALL_MOVED_CANDIDATE",  # 공 움직임 감지 → restart 끝, 일반 play 복귀
}

# control_fsm RobotState → ball_follower 실행 게이팅용 exec_state.
# ball_follower 는 {AIM,KICK,RECOVER,COOLDOWN} 에서 팔스윙을 끈다. 매핑은 그 관점에서.
_FSM_TO_EXEC = {
    "SEARCH": "SEARCH",
    "TRACK": "TRACK",
    "APPROACH": "APPROACH",
    "STOP": "AIM",          # [TUNE] 정지=제자리 안정 → 보행 팔스윙 OFF (AIM 취급)
    "ALIGN_GOAL": "AIM",    # [TUNE] 제자리 골 정렬 → 팔스윙 OFF
    "KICK": "KICK",
    "BACK_OFF": "APPROACH", # 후진도 보행 → 팔스윙 ON
    "RECOVER": "RECOVER",
    "COOLDOWN": "COOLDOWN",
}


@dataclass
class UnifiedDecision:
    vx: float = 0.0
    wz: float = 0.0
    kick: bool = False
    exec_state: str = "SEARCH"   # 팔스윙/kick 포즈 게이팅용 (ball_follower RobotState 이름)
    source: str = "control_fsm"  # penalized / setplay / control_fsm
    reason: str = ""
    fsm_state: str = "SEARCH"    # control_fsm 원 상태 (로깅용)
    setplay_state: str = "INACTIVE"


class DecisionEngine:
    def __init__(self, fsm_cfg: Optional[BasicFSMConfig] = None):
        self.fsm = BasicRobotStateFSM(fsm_cfg) if fsm_cfg else BasicRobotStateFSM()
        self.last: Optional[UnifiedDecision] = None

    def decide(self, robot_state: Dict[str, Any], lower: Optional[Dict[str, Any]] = None,
               *, gc_state_name: Optional[str] = None, now: Optional[float] = None) -> UnifiedDecision:
        lower = lower or {"is_fallen": False}
        gcs = gc_state_name or robot_state.get("game_controller_state", "UNKNOWN")
        sp = robot_state.get("setplay_vision_state", "INACTIVE")

        # control_fsm 은 항상 돌려서 내부 상태/타이머를 유지(전이 연속성). 명령은 우선순위로 덮어쓴다.
        result = self.fsm.update(lower, robot_state, now)
        cmd = result.command
        fsm_state = result.state.name

        # 1) PENALIZED / 비상정지 → 전면 정지
        if gcs in ("PENALIZED", "STOP", "FINISHED", "INITIAL"):
            d = UnifiedDecision(0.0, 0.0, False, "RECOVER" if gcs == "INITIAL" else "COOLDOWN",
                                "penalized", f"game_controller_state={gcs} → 정지", fsm_state, sp)
            self.last = d
            return d

        # SET: 정지 대기 (움직이면 penalty). PLAYING/READY 만 이동 허용.
        if gcs == "SET":
            d = UnifiedDecision(0.0, 0.0, False, "AIM", "game_state", "SET → 정지 대기", fsm_state, sp)
            self.last = d
            return d

        # 2) setplay 진행 중 + 대기/미준비 상태 → 접근/킥 보류
        if sp in _SETPLAY_HOLD:
            # 제자리 대기. (상대 킥이면 룰상 공에서 거리 유지. 여기선 정지로 충분, 거리확보 이동은 [TUNE])
            d = UnifiedDecision(0.0, 0.0, False, "STOP", "setplay",
                                f"setplay={sp} → 대기(접근/킥 보류)", fsm_state, sp)
            self.last = d
            return d

        # 3) 일반 play (또는 setplay GO 상태) → control_fsm 명령 사용
        exec_state = _FSM_TO_EXEC.get(fsm_state, "SEARCH")
        d = UnifiedDecision(
            vx=float(cmd.target_body_vx), wz=float(cmd.target_body_wz),
            kick=bool(cmd.kick_trigger), exec_state=exec_state,
            source="control_fsm", reason=result.reason, fsm_state=fsm_state, setplay_state=sp)
        self.last = d
        return d
