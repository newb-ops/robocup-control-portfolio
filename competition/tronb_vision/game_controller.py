from __future__ import annotations
"""GameController 상태를 이동 허용 여부와 공 접촉·킥 허용 여부로 바꾼다.

상태는 GameController나 상위 로직에서 전달받는다. UNKNOWN이면 (None, None)을
반환하므로, 이를 허용으로 처리할지는 호출하는 제어 로직이 결정해야 한다."""

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
