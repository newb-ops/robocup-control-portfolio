from __future__ import annotations
"""
game_controller.py — GameController 상태 → 비전이 참고할 허용 플래그.

restart_type/owner 와 마찬가지로 게임 상태는 비전이 카메라로 알 수 없다.
GameController(또는 테스트 config/상위 로직)가 넣어준 GameControllerState 를
'움직여도 되나(locomotion)', 'active play(공 접촉/킥) 해도 되나(active)' 두 bool 로 환산.

알 수 없음(UNKNOWN/미주입)은 (None, None) — 단독 비전 테스트에서 막지 않도록 permissive.
제어 FSM 이 None 을 '제약 없음'으로 해석할지 보수적으로 볼지는 제어쪽 정책.
"""

from typing import Optional, Tuple

from .common import GameControllerState


def normalize_game_controller_state(state) -> GameControllerState:
    """문자열/enum/None 을 GameControllerState 로 정규화."""
    if state is None:
        return GameControllerState.UNKNOWN
    if isinstance(state, GameControllerState):
        return state
    try:
        return GameControllerState[str(state).strip().upper()]
    except (KeyError, ValueError):
        return GameControllerState.UNKNOWN


def game_state_flags(state: GameControllerState) -> Tuple[Optional[bool], Optional[bool]]:
    """(allows_locomotion, allows_active_play). 모르면 (None, None)."""
    S = GameControllerState
    table = {
        S.PLAYING:   (True,  True),    # 정상 경기: 이동·active play 모두 허용
        S.READY:     (True,  False),   # 배치 이동만, 공 접촉 금지
        S.SET:       (False, False),   # 정지 대기
        S.INITIAL:   (False, False),
        S.FINISHED:  (False, False),
        S.PENALIZED: (False, False),   # 페널티 중: 정지
        S.STOP:      (False, False),   # 비상 정지
    }
    return table.get(state, (None, None))  # UNKNOWN 등 → permissive
