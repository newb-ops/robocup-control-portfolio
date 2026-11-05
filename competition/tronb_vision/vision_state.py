from __future__ import annotations
"""관측값을 제어 FSM에서 읽는 공 인식 상태와 킥 후보 힌트로 바꾼다.
출력은 app이 robot_state의 ball_perception_state와 vision_packet에 기록한다."""
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional, Any, Dict
import math
import time
from .common import BallObservation, GoalObservation, GameControllerState
from .rule_state import BallRuleStatus


class BallPerceptionState(Enum):
    BALL_LOST = auto()
    BALL_TEMP_LOST = auto()
    BALL_UNSTABLE = auto()
    BALL_STABLE_OFF_CENTER = auto()
    BALL_STABLE_CENTER_FAR = auto()
    BALL_STABLE_CENTER_NEAR = auto()
    BALL_TOO_CLOSE = auto()


@dataclass
class VisionStateConfig:
    center_tol_norm: float = 0.15
    goal_center_tol_norm: float = 0.20
    far_dist_m: float = 1.2
    too_close_dist_m: float = 0.16
    min_stable_detect_frames: int = 2
    temp_lost_max_frames: int = 12
    hard_lost_frames: int = 35
    min_ball_conf: float = 0.25
    vision_kick_hint_center_tol_norm: float = 0.12
    vision_kick_hint_min_dist_m: float = 0.17
    vision_kick_hint_max_dist_m: float = 0.8


@dataclass
class RuleBasedVisionPacket:
    ball_state: BallPerceptionState
    ball_found: bool
    ball_stable: bool
    ball_screen_x: Optional[float]
    ball_screen_y: Optional[float]
    ball_dist: Optional[float]
    ball_conf: Optional[float]
    lost_frames: int
    detect_frames: int
    goal_visible: bool
    goal_centered: bool
    goal_screen_x: Optional[float]
    goal_screen_y: Optional[float]
    localization_reliable: bool
    ball_world_x: Optional[float]
    ball_world_y: Optional[float]
    ball_rule_status: BallRuleStatus
    game_controller_state: GameControllerState
    game_allows_locomotion: Optional[bool]
    game_allows_active_play: Optional[bool]
    ball_near_centered_by_vision: bool
    vision_kick_candidate_hint: bool
    source: str = "v6_vision_state"
    notes: str = ""
    timestamp: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ball_state"] = self.ball_state.name
        d["ball_rule_status"] = self.ball_rule_status.name
        d["game_controller_state"] = self.game_controller_state.name
        return d


def classify_ball_perception(obs: BallObservation, cfg: VisionStateConfig) -> tuple[BallPerceptionState, bool, str]:
    if not obs.detected:
        if 0 < obs.lost_frames <= cfg.temp_lost_max_frames:
            return BallPerceptionState.BALL_TEMP_LOST, False, "temporary lost"
        return BallPerceptionState.BALL_LOST, False, "no usable ball detection"
    if obs.dist <= 0.0 or obs.dist >= 9.0 or not math.isfinite(obs.dist):
        return BallPerceptionState.BALL_UNSTABLE, False, "invalid distance"
    if obs.confidence and obs.confidence < cfg.min_ball_conf:
        return BallPerceptionState.BALL_UNSTABLE, False, "low confidence"
    if obs.consecutive_detect < cfg.min_stable_detect_frames:
        return BallPerceptionState.BALL_UNSTABLE, False, "not enough consecutive detections"
    if obs.dist <= cfg.too_close_dist_m:
        return BallPerceptionState.BALL_TOO_CLOSE, True, "too close"
    if abs(obs.screen_x) > cfg.center_tol_norm:
        return BallPerceptionState.BALL_STABLE_OFF_CENTER, True, "stable off center"
    if obs.dist > cfg.far_dist_m:
        return BallPerceptionState.BALL_STABLE_CENTER_FAR, True, "stable centered far"
    return BallPerceptionState.BALL_STABLE_CENTER_NEAR, True, "stable centered near"


def build_rule_based_vision_packet(
    *,
    obs: BallObservation,
    goal: GoalObservation,
    localization_reliable: bool,
    ball_world_x: Optional[float],
    ball_world_y: Optional[float],
    ball_rule_status: BallRuleStatus,
    game_controller_state: GameControllerState,
    game_allows_locomotion: Optional[bool],
    game_allows_active_play: Optional[bool],
    cfg: VisionStateConfig | None = None,
) -> RuleBasedVisionPacket:
    cfg = cfg or VisionStateConfig()
    bstate, stable, notes = classify_ball_perception(obs, cfg)
    ball_near_centered = (
        bstate is BallPerceptionState.BALL_STABLE_CENTER_NEAR
        and abs(obs.screen_x) <= cfg.vision_kick_hint_center_tol_norm
        and cfg.vision_kick_hint_min_dist_m <= obs.dist <= cfg.vision_kick_hint_max_dist_m
    )
    kick_hint = bool(ball_near_centered and game_allows_active_play is not False and ball_rule_status in (BallRuleStatus.UNKNOWN, BallRuleStatus.IN_FIELD_OR_TOUCHING_LINE))
    return RuleBasedVisionPacket(
        ball_state=bstate,
        ball_found=bool(obs.detected),
        ball_stable=stable,
        ball_screen_x=float(obs.screen_x) if obs is not None else None,
        ball_screen_y=float(obs.screen_y) if obs is not None else None,
        ball_dist=float(obs.dist) if obs is not None else None,
        ball_conf=float(obs.confidence),
        lost_frames=int(obs.lost_frames),
        detect_frames=int(obs.consecutive_detect),
        goal_visible=bool(goal.detected),
        goal_centered=bool(goal.detected and abs(goal.screen_x) <= cfg.goal_center_tol_norm),
        goal_screen_x=float(goal.screen_x) if goal.detected else None,
        goal_screen_y=float(goal.screen_y) if goal.detected else None,
        localization_reliable=bool(localization_reliable),
        ball_world_x=ball_world_x,
        ball_world_y=ball_world_y,
        ball_rule_status=ball_rule_status,
        game_controller_state=game_controller_state,
        game_allows_locomotion=game_allows_locomotion,
        game_allows_active_play=game_allows_active_play,
        ball_near_centered_by_vision=ball_near_centered,
        vision_kick_candidate_hint=kick_hint,
        notes=notes,
        timestamp=time.time(),
    )
