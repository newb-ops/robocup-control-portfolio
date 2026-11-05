from __future__ import annotations
"""공의 필드 좌표로 경계 안쪽, 라인 위, 경계 밖 후보를 구분한다.

공 좌표를 알 수 없으면 UNKNOWN을 반환한다. 위치추정 신뢰 여부는 호출부에서
확인하며, 최종 판정과 동작 선택은 제어 FSM이 담당한다."""

from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional


class BallRuleStatus(Enum):
    UNKNOWN = auto()                      # 월드좌표 없음/불신 → 판단 불가
    IN_FIELD_OR_TOUCHING_LINE = auto()    # 필드 안 또는 라인 위(경계 포함)
    OUT_CANDIDATE = auto()                # 경계 밖 후보 (touchline/goalline out)


@dataclass
class RuleStateSnapshot:
    ball_rule_status: BallRuleStatus = BallRuleStatus.UNKNOWN
    ball_world_x: Optional[float] = None
    ball_world_y: Optional[float] = None
    margin_m: Optional[float] = None      # 경계까지 여유(+안쪽, -밖). None=알수없음
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ball_rule_status"] = self.ball_rule_status.name
        return d


def build_rule_state(
    ball_world_x: Optional[float],
    ball_world_y: Optional[float],
    field,
    line_tol_m: float = 0.10,
) -> RuleStateSnapshot:
    """
    field: half_len/half_wid 속성을 가진 객체(FieldMap 등) 또는 None.
    line_tol_m: 경계 ±이 범위는 '라인 위(터칭)'로 본다.
    """
    if ball_world_x is None or ball_world_y is None:
        return RuleStateSnapshot(BallRuleStatus.UNKNOWN, None, None, None, "no ball world coord")

    half_len = float(getattr(field, "half_len", 7.0))
    half_wid = float(getattr(field, "half_wid", 4.5))
    # 경계까지 여유 (가장 가까운 변 기준). +면 안쪽, -면 밖.
    mx = half_len - abs(float(ball_world_x))
    my = half_wid - abs(float(ball_world_y))
    margin = min(mx, my)

    if margin >= -line_tol_m:
        status = BallRuleStatus.IN_FIELD_OR_TOUCHING_LINE
        notes = "inside" if margin > line_tol_m else "on line (touching)"
    else:
        status = BallRuleStatus.OUT_CANDIDATE
        notes = "out candidate"
    return RuleStateSnapshot(status, float(ball_world_x), float(ball_world_y), float(margin), notes)
