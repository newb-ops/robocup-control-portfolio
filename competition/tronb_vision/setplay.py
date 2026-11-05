#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""세트플레이 상황에서 필요한 인식 상태를 계산한다.
실제 로봇 동작의 선택은 제어 FSM에서 담당한다."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Union
import math
import time


# Set-play context enums


class RestartType(Enum):
    """Set-play type supplied by GameController / upper game logic."""

    NONE = auto()
    PENALTY_KICK = auto()
    CORNER_KICK = auto()
    GOAL_KICK = auto()
    TOUCHLINE_RESTART = auto()   # Kick-in / Throw-in merged for now.
    PUSHING_FREE_KICK = auto()
    UNKNOWN = auto()


class RestartOwner(Enum):
    """Which team owns the active restart, supplied from outside vision."""

    NONE = auto()
    OUR_TEAM = auto()
    OPPONENT_TEAM = auto()
    UNKNOWN = auto()


class TargetContextPolicy(Enum):
    """Attack-branch target context requirement."""

    OPTIONAL = auto()
    GOAL_REQUIRED = auto()
    GOAL_OR_ROBOT_REQUIRED = auto()


class SetPlayVisionState(Enum):
    """Vision-side compact FSM. Not a behavior/control FSM."""

    INACTIVE = auto()
    CONTEXT_UNKNOWN = auto()

    # Our team owns the restart.
    ATTACK_BALL_NOT_READY = auto()
    ATTACK_TARGET_NOT_READY = auto()
    ATTACK_CONTEXT_READY = auto()

    # Opponent owns the restart.
    DEFENSE_BALL_NOT_READY = auto()
    DEFENSE_WAIT_RESTART = auto()
    DEFENSE_BALL_MOVED_CANDIDATE = auto()


# Config / packet


@dataclass
class SetPlayVisionConfig:
    """Thresholds and policy for the set-play vision FSM."""


    # Prefer the ball motion estimator over anchor displacement.

    # Tune against recorded kicks and stationary-ball jitter.
    motion_estimator_min_speed_mps: float = 0.20

    # A fitted speed based on too few samples is not trusted as restart evidence.
    motion_estimator_min_fit_n: int = 2


    # Use anchor displacement only when robot_state has no estimator fields.

    fallback_anchor_motion_threshold_m: float = 0.10
    fallback_anchor_confirm_frames: int = 2
    allow_anchor_motion_fallback: bool = True

    # World-coordinate motion judgment requires reliable pose context. Set False
    # only for controlled debugging, not for normal integration.
    require_localization_for_world_motion: bool = True

    # If the integrated loop exposes robot_detections, treat it as a weak context
    # hint. It is not interpreted as "teammate found".
    allow_generic_robot_detections: bool = True

    # Penalties require goal context. Corner/touchline strategy belongs to control;
    # generic robot detections cannot reliably determine whether play is ready.
    penalty_policy: TargetContextPolicy = TargetContextPolicy.GOAL_REQUIRED
    corner_policy: TargetContextPolicy = TargetContextPolicy.OPTIONAL
    touchline_policy: TargetContextPolicy = TargetContextPolicy.OPTIONAL
    goal_kick_policy: TargetContextPolicy = TargetContextPolicy.OPTIONAL
    pushing_free_kick_policy: TargetContextPolicy = TargetContextPolicy.OPTIONAL


@dataclass
class SetPlayVisionPacket:
    """Vision-only set-play packet written into robot_state."""

    restart_type: RestartType
    restart_owner: RestartOwner
    state: SetPlayVisionState

    # Existing ball/goal vision summarized from robot_state["vision_packet"].
    ball_state_name: str
    ball_ready: bool
    goal_context_visible: bool
    goal_context_centered: bool

    # Generic robot-context cue. This is NOT "teammate found".
    robot_context_visible: bool
    robot_context_source: str

    # ATTACK-branch target-context result.
    target_context_policy: TargetContextPolicy
    target_context_ready: bool
    target_context_source: str

    # Reliability/context needed for world-coordinate motion logic.
    localization_reliable: bool
    vision_latency_status: str
    world_context_reliable: bool

    # DEFENSE-branch restart motion observation.
    ball_moved_candidate: bool
    restart_motion_confirmed: bool
    motion_evidence_source: str
    ball_anchor_displacement_m: Optional[float]

    # Motion estimator status and measurements, when available.
    motion_estimator_status: str
    motion_estimator_reason: str
    motion_estimator_speed_mps: Optional[float]
    motion_estimator_fit_n: Optional[int]

    # Optional robot metadata; not used in state decisions.
    robot_id_hint: Optional[int]
    team_role_hint: Optional[str]

    # Debug / provenance.
    source: str
    notes: str
    timestamp: float

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["restart_type"] = self.restart_type.name
        data["restart_owner"] = self.restart_owner.name
        data["state"] = self.state.name
        data["target_context_policy"] = self.target_context_policy.name
        return data


@dataclass
class _MotionEvidence:
    """Internal motion-evidence result."""

    confirmed: bool = False
    source: str = "NONE"
    anchor_displacement_m: Optional[float] = None
    estimator_status: str = "UNAVAILABLE"
    estimator_reason: str = "UNAVAILABLE"
    estimator_speed_mps: Optional[float] = None
    estimator_fit_n: Optional[int] = None
    note: str = "no restart motion evidence"


# Adapter implementation


class SetPlayVisionAdapter:
    """Stateful per-robot set-play vision adapter."""

    def __init__(self, cfg: Optional[SetPlayVisionConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else SetPlayVisionConfig()
        self._prev_restart_key: Optional[Tuple[RestartType, RestartOwner]] = None

        # Keep an anchor for frames without motion estimator output.
        self._anchor_ball_world_xy: Optional[Tuple[float, float]] = None
        self._anchor_confirm_streak: int = 0


    # Public API

    def update_robot_state(
        self,
        robot_state: Dict[str, Any],
        *,
        restart_type: Optional[Union[str, RestartType]] = None,
        restart_owner: Optional[Union[str, RestartOwner]] = None,
        write_back: bool = True,
    ) -> SetPlayVisionPacket:
        packet = self.estimate_from_robot_state(
            robot_state,
            restart_type=restart_type,
            restart_owner=restart_owner,
        )
        if write_back:
            attach_setplay_packet_to_robot_state(robot_state, packet)
        return packet

    def estimate_from_robot_state(
        self,
        robot_state: Dict[str, Any],
        *,
        restart_type: Optional[Union[str, RestartType]] = None,
        restart_owner: Optional[Union[str, RestartOwner]] = None,
    ) -> SetPlayVisionPacket:
        now = time.time()
        rtype = _normalize_restart_type(
            restart_type if restart_type is not None else robot_state.get("restart_type")
        )
        owner = _normalize_restart_owner(
            restart_owner if restart_owner is not None else robot_state.get("restart_owner")
        )

        # Reset fallback motion memory when restart context changes.
        restart_key = (rtype, owner)
        if restart_key != self._prev_restart_key:
            self._reset_context_memory()
            self._prev_restart_key = restart_key

        vision_packet = robot_state.get("vision_packet")
        ball_state_name = _packet_attr_name(vision_packet, "ball_state", default="UNKNOWN")
        ball_ready = bool(_packet_attr(vision_packet, "ball_stable", default=False))
        goal_visible = bool(_packet_attr(vision_packet, "goal_visible", default=False))
        goal_centered = bool(_packet_attr(vision_packet, "goal_centered", default=False))

        robot_context_visible, robot_context_source = self._robot_context_visible(robot_state)
        target_policy = self._target_policy(rtype)
        target_ready, target_source = self._target_context_ready(
            policy=target_policy,
            goal_visible=goal_visible,
            robot_context_visible=robot_context_visible,
        )

        localization_reliable = self._localization_reliable(robot_state, vision_packet)
        latency_status = self._vision_latency_status(robot_state)

        world_x, world_y = self._ball_world_xy(robot_state, vision_packet)
        world_coords_available = world_x is not None and world_y is not None
        world_context_reliable = (
            world_coords_available
            and latency_status != "STALE"
            and (
                localization_reliable
                or not self.cfg.require_localization_for_world_motion
            )
        )

        motion = self._restart_motion_evidence(
            robot_state=robot_state,
            ball_ready=ball_ready,
            world_context_reliable=world_context_reliable,
            ball_world_xy=(world_x, world_y) if world_coords_available else None,
        )

        state, state_note = self._state(
            restart_type=rtype,
            restart_owner=owner,
            ball_ready=ball_ready,
            target_ready=target_ready,
            moved_candidate=motion.confirmed,
        )

        notes = state_note
        if owner == RestartOwner.OPPONENT_TEAM:
            notes = f"{state_note}; motion={motion.note}"

        return SetPlayVisionPacket(
            restart_type=rtype,
            restart_owner=owner,
            state=state,
            ball_state_name=ball_state_name,
            ball_ready=ball_ready,
            goal_context_visible=goal_visible,
            goal_context_centered=goal_centered,
            robot_context_visible=robot_context_visible,
            robot_context_source=robot_context_source,
            target_context_policy=target_policy,
            target_context_ready=target_ready,
            target_context_source=target_source,
            localization_reliable=localization_reliable,
            vision_latency_status=latency_status,
            world_context_reliable=world_context_reliable,
            ball_moved_candidate=motion.confirmed,
            restart_motion_confirmed=motion.confirmed,
            motion_evidence_source=motion.source,
            ball_anchor_displacement_m=motion.anchor_displacement_m,
            motion_estimator_status=motion.estimator_status,
            motion_estimator_reason=motion.estimator_reason,
            motion_estimator_speed_mps=motion.estimator_speed_mps,
            motion_estimator_fit_n=motion.estimator_fit_n,
            robot_id_hint=_safe_int(robot_state.get("robot_id"), None),
            team_role_hint=_safe_str(robot_state.get("team_role"), None),
            source="setplay_vision_state_adapter_v2",
            notes=notes,
            timestamp=now,
        )


    # FSM state selection

    def _state(
        self,
        *,
        restart_type: RestartType,
        restart_owner: RestartOwner,
        ball_ready: bool,
        target_ready: bool,
        moved_candidate: bool,
    ) -> Tuple[SetPlayVisionState, str]:
        if restart_type == RestartType.NONE:
            return SetPlayVisionState.INACTIVE, "no active restart/set-play context"

        if restart_type == RestartType.UNKNOWN or restart_owner in (
            RestartOwner.NONE,
            RestartOwner.UNKNOWN,
        ):
            return SetPlayVisionState.CONTEXT_UNKNOWN, "restart type/owner unavailable to vision adapter"

        if restart_owner == RestartOwner.OUR_TEAM:
            if not ball_ready:
                return SetPlayVisionState.ATTACK_BALL_NOT_READY, "our restart; ball perception not ready"
            if not target_ready:
                return SetPlayVisionState.ATTACK_TARGET_NOT_READY, "our restart; ball ready but required target context not ready"
            return SetPlayVisionState.ATTACK_CONTEXT_READY, "our restart; required perception context ready"

        if restart_owner == RestartOwner.OPPONENT_TEAM:
            if not ball_ready:
                return SetPlayVisionState.DEFENSE_BALL_NOT_READY, "opponent restart; ball perception not ready"
            if moved_candidate:
                return SetPlayVisionState.DEFENSE_BALL_MOVED_CANDIDATE, "opponent restart; restart motion evidence confirmed"
            return SetPlayVisionState.DEFENSE_WAIT_RESTART, "opponent restart; ball ready and no confirmed restart motion"

        return SetPlayVisionState.CONTEXT_UNKNOWN, "unhandled restart owner"


    # Target context policy

    def _target_policy(self, restart_type: RestartType) -> TargetContextPolicy:
        if restart_type == RestartType.PENALTY_KICK:
            return self.cfg.penalty_policy
        if restart_type == RestartType.CORNER_KICK:
            return self.cfg.corner_policy
        if restart_type == RestartType.TOUCHLINE_RESTART:
            return self.cfg.touchline_policy
        if restart_type == RestartType.GOAL_KICK:
            return self.cfg.goal_kick_policy
        if restart_type == RestartType.PUSHING_FREE_KICK:
            return self.cfg.pushing_free_kick_policy
        return TargetContextPolicy.OPTIONAL

    @staticmethod
    def _target_context_ready(
        *,
        policy: TargetContextPolicy,
        goal_visible: bool,
        robot_context_visible: bool,
    ) -> Tuple[bool, str]:
        if policy == TargetContextPolicy.OPTIONAL:
            return True, "OPTIONAL"
        if policy == TargetContextPolicy.GOAL_REQUIRED:
            return goal_visible, "GOAL" if goal_visible else "NONE"
        if policy == TargetContextPolicy.GOAL_OR_ROBOT_REQUIRED:
            if goal_visible:
                return True, "GOAL"
            if robot_context_visible:
                return True, "ROBOT_CONTEXT"
            return False, "NONE"
        return False, "NONE"

    def _robot_context_visible(self, robot_state: Dict[str, Any]) -> Tuple[bool, str]:
        # Prefer an explicit robot-context flag when provided.
        explicit = robot_state.get("setplay_robot_context_visible")
        if explicit is not None:
            return bool(explicit), "robot_state.setplay_robot_context_visible"

        # Generic detections are only a hint under the default policies.
        if self.cfg.allow_generic_robot_detections:
            detections = robot_state.get("robot_detections")
            if isinstance(detections, (list, tuple)) and len(detections) > 0:
                return True, "robot_state.robot_detections"

        return False, "NONE"


    # Reliability helpers

    @staticmethod
    def _vision_latency_status(robot_state: Dict[str, Any]) -> str:
        value = robot_state.get("vision_latency_status", "UNKNOWN")
        text = str(value).strip().upper() if value is not None else "UNKNOWN"
        return text if text else "UNKNOWN"

    @staticmethod
    def _localization_reliable(robot_state: Dict[str, Any], vision_packet: Any) -> bool:
        explicit = robot_state.get("vision_localization_reliable")
        if explicit is not None:
            return bool(explicit)
        return bool(_packet_attr(vision_packet, "localization_reliable", default=False))

    @staticmethod
    def _ball_world_xy(robot_state: Dict[str, Any], vision_packet: Any) -> Tuple[Optional[float], Optional[float]]:
        x = _safe_float(_packet_attr(vision_packet, "ball_world_x", default=None))
        y = _safe_float(_packet_attr(vision_packet, "ball_world_y", default=None))
        if x is None:
            x = _safe_float(robot_state.get("ball_world_x_est"), None)
        if y is None:
            y = _safe_float(robot_state.get("ball_world_y_est"), None)
        return x, y


    # Restart motion evidence

    def _restart_motion_evidence(
        self,
        *,
        robot_state: Dict[str, Any],
        ball_ready: bool,
        world_context_reliable: bool,
        ball_world_xy: Optional[Tuple[float, float]],
    ) -> _MotionEvidence:
        """Check whether the opponent has moved the ball.
        Use estimator fields when present; otherwise compare with the saved anchor.
        """
        if not ball_ready:
            return _MotionEvidence(note="ball perception not ready")

        estimator_present = any(
            key in robot_state
            for key in (
                "ball_stop_status",
                "ball_speed_est",
                "ball_fit_n",
                "ball_stop_reason",
            )
        )

        if estimator_present:
            return self._motion_from_v6_estimator(
                robot_state=robot_state,
                world_context_reliable=world_context_reliable,
            )

        if not self.cfg.allow_anchor_motion_fallback:
            return _MotionEvidence(
                source="NONE",
                note="v6 estimator unavailable and anchor fallback disabled",
            )

        return self._motion_from_anchor_fallback(
            ball_world_xy=ball_world_xy,
            world_context_reliable=world_context_reliable,
        )

    def _motion_from_v6_estimator(
        self,
        *,
        robot_state: Dict[str, Any],
        world_context_reliable: bool,
    ) -> _MotionEvidence:
        status = _safe_str(robot_state.get("ball_stop_status"), "UNAVAILABLE") or "UNAVAILABLE"
        reason = _safe_str(robot_state.get("ball_stop_reason"), "UNAVAILABLE") or "UNAVAILABLE"
        status = status.upper()
        reason = reason.lower()
        speed = _safe_float(robot_state.get("ball_speed_est"), None)
        fit_n = _safe_int(robot_state.get("ball_fit_n"), None)

        base = _MotionEvidence(
            confirmed=False,
            source="V6_STOP_ESTIMATOR",
            estimator_status=status,
            estimator_reason=reason,
            estimator_speed_mps=speed,
            estimator_fit_n=fit_n,
        )

        if not world_context_reliable:
            base.note = "v6 estimator present but world context not reliable"
            return base

        if status != "VALID":
            base.note = f"v6 estimator status={status}; motion not confirmed"
            return base

        if speed is None or speed < self.cfg.motion_estimator_min_speed_mps:
            base.note = (
                "v6 estimator valid but speed below threshold "
                f"({0.0 if speed is None else speed:.3f} < {self.cfg.motion_estimator_min_speed_mps:.3f})"
            )
            return base

        if fit_n is None or fit_n < self.cfg.motion_estimator_min_fit_n:
            base.note = (
                "v6 estimator valid but fit_n below threshold "
                f"({-1 if fit_n is None else fit_n} < {self.cfg.motion_estimator_min_fit_n})"
            )
            return base

        base.confirmed = True
        base.note = (
            "v6 estimator confirms restart motion "
            f"(speed={speed:.3f}m/s, fit_n={fit_n})"
        )
        return base

    def _motion_from_anchor_fallback(
        self,
        *,
        ball_world_xy: Optional[Tuple[float, float]],
        world_context_reliable: bool,
    ) -> _MotionEvidence:
        if not world_context_reliable or ball_world_xy is None:
            return _MotionEvidence(
                source="ANCHOR_FALLBACK",
                note="anchor fallback blocked: world context unavailable/unreliable",
            )

        current = (float(ball_world_xy[0]), float(ball_world_xy[1]))

        if self._anchor_ball_world_xy is None:
            self._anchor_ball_world_xy = current
            self._anchor_confirm_streak = 0
            return _MotionEvidence(
                source="ANCHOR_FALLBACK",
                anchor_displacement_m=0.0,
                note="anchor fallback initialized at first reliable ball sample",
            )

        dx = current[0] - self._anchor_ball_world_xy[0]
        dy = current[1] - self._anchor_ball_world_xy[1]
        displacement = math.hypot(dx, dy)

        if displacement >= self.cfg.fallback_anchor_motion_threshold_m:
            self._anchor_confirm_streak += 1
        else:
            self._anchor_confirm_streak = 0

        confirmed = self._anchor_confirm_streak >= self.cfg.fallback_anchor_confirm_frames
        note = (
            "anchor fallback confirms restart motion"
            if confirmed
            else (
                "anchor displacement not yet confirmed "
                f"(disp={displacement:.3f}m, streak={self._anchor_confirm_streak}/"
                f"{self.cfg.fallback_anchor_confirm_frames})"
            )
        )
        return _MotionEvidence(
            confirmed=confirmed,
            source="ANCHOR_FALLBACK",
            anchor_displacement_m=displacement,
            note=note,
        )

    def _reset_context_memory(self) -> None:
        self._anchor_ball_world_xy = None
        self._anchor_confirm_streak = 0


# Integration helpers


def attach_setplay_packet_to_robot_state(
    robot_state: Dict[str, Any],
    packet: SetPlayVisionPacket,
) -> Dict[str, Any]:
    """Write only setplay_* fields back into robot_state."""
    robot_state["setplay_vision_packet"] = packet
    robot_state["setplay_vision_packet_dict"] = packet.as_dict()
    robot_state["setplay_vision_state"] = packet.state.name
    robot_state["setplay_restart_type"] = packet.restart_type.name
    robot_state["setplay_restart_owner"] = packet.restart_owner.name
    robot_state["setplay_ball_ready"] = packet.ball_ready
    robot_state["setplay_target_context_ready"] = packet.target_context_ready
    robot_state["setplay_target_context_source"] = packet.target_context_source
    robot_state["setplay_goal_context_visible"] = packet.goal_context_visible
    robot_state["setplay_robot_context_visible"] = packet.robot_context_visible
    robot_state["setplay_world_context_reliable"] = packet.world_context_reliable
    robot_state["setplay_vision_latency_status"] = packet.vision_latency_status
    robot_state["setplay_localization_reliable"] = packet.localization_reliable
    robot_state["setplay_ball_moved_candidate"] = packet.ball_moved_candidate
    robot_state["setplay_restart_motion_confirmed"] = packet.restart_motion_confirmed
    robot_state["setplay_motion_evidence_source"] = packet.motion_evidence_source
    robot_state["setplay_ball_anchor_displacement_m"] = packet.ball_anchor_displacement_m
    robot_state["setplay_motion_estimator_status"] = packet.motion_estimator_status
    robot_state["setplay_motion_estimator_reason"] = packet.motion_estimator_reason
    robot_state["setplay_motion_estimator_speed_mps"] = packet.motion_estimator_speed_mps
    robot_state["setplay_motion_estimator_fit_n"] = packet.motion_estimator_fit_n
    robot_state["setplay_robot_id_hint"] = packet.robot_id_hint
    robot_state["setplay_team_role_hint"] = packet.team_role_hint
    robot_state["setplay_notes"] = packet.notes
    robot_state["setplay_packet_time"] = packet.timestamp
    return robot_state


def update_v6_robot_state_with_setplay_rules(
    robot_state: Dict[str, Any],
    *,
    restart_type: Optional[Union[str, RestartType]] = None,
    restart_owner: Optional[Union[str, RestartOwner]] = None,
    adapter: Optional[SetPlayVisionAdapter] = None,
) -> SetPlayVisionPacket:
    """기존 인식 상태에 세트플레이 규칙을 적용해 robot_state를 갱신한다."""
    if adapter is None:
        adapter = SetPlayVisionAdapter()
    return adapter.update_robot_state(
        robot_state,
        restart_type=restart_type,
        restart_owner=restart_owner,
        write_back=True,
    )


def update_v5_robot_state_with_setplay_rules(
    robot_state: Dict[str, Any],
    *,
    restart_type: Optional[Union[str, RestartType]] = None,
    restart_owner: Optional[Union[str, RestartOwner]] = None,
    adapter: Optional[SetPlayVisionAdapter] = None,
) -> SetPlayVisionPacket:
    """Backward-compatible alias for earlier v1 integration code."""
    return update_v6_robot_state_with_setplay_rules(
        robot_state,
        restart_type=restart_type,
        restart_owner=restart_owner,
        adapter=adapter,
    )


# Small utilities


def _packet_attr(packet: Any, key: str, default: Any = None) -> Any:
    if packet is None:
        return default
    if isinstance(packet, dict):
        return packet.get(key, default)
    return getattr(packet, key, default)


def _packet_attr_name(packet: Any, key: str, default: str = "UNKNOWN") -> str:
    value = _packet_attr(packet, key, default=None)
    if value is None:
        return default
    return getattr(value, "name", str(value))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_str(value: Any, default: Optional[str] = None) -> Optional[str]:
    if value is None:
        return default
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _normalize_restart_type(value: Any) -> RestartType:
    if isinstance(value, RestartType):
        return value
    if value is None:
        return RestartType.NONE
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NONE": "NONE",
        "NO_RESTART": "NONE",
        "NORMAL_PLAY": "NONE",
        "PENALTY": "PENALTY_KICK",
        "PENALTYKICK": "PENALTY_KICK",
        "PENALTY_KICK": "PENALTY_KICK",
        "PK": "PENALTY_KICK",
        "CORNER": "CORNER_KICK",
        "CORNERKICK": "CORNER_KICK",
        "CORNER_KICK": "CORNER_KICK",
        "GOAL_KICK": "GOAL_KICK",
        "GOALKICK": "GOAL_KICK",
        "KICK_IN": "TOUCHLINE_RESTART",
        "KICKIN": "TOUCHLINE_RESTART",
        "THROW_IN": "TOUCHLINE_RESTART",
        "THROWIN": "TOUCHLINE_RESTART",
        "TOUCHLINE_RESTART": "TOUCHLINE_RESTART",
        "PUSHING_FREE_KICK": "PUSHING_FREE_KICK",
        "FREE_KICK": "PUSHING_FREE_KICK",
        "FREEKICK": "PUSHING_FREE_KICK",
    }
    mapped = aliases.get(text)
    if mapped and mapped in RestartType.__members__:
        return RestartType[mapped]
    return RestartType.UNKNOWN


def _normalize_restart_owner(value: Any) -> RestartOwner:
    if isinstance(value, RestartOwner):
        return value
    if value is None:
        return RestartOwner.NONE
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "NONE": "NONE",
        "OUR": "OUR_TEAM",
        "US": "OUR_TEAM",
        "OUR_TEAM": "OUR_TEAM",
        "ATTACK": "OUR_TEAM",
        "ATTACKING": "OUR_TEAM",
        "OPP": "OPPONENT_TEAM",
        "OPPONENT": "OPPONENT_TEAM",
        "OPPONENT_TEAM": "OPPONENT_TEAM",
        "THEM": "OPPONENT_TEAM",
        "DEFENSE": "OPPONENT_TEAM",
        "DEFENDING": "OPPONENT_TEAM",
    }
    mapped = aliases.get(text)
    if mapped and mapped in RestartOwner.__members__:
        return RestartOwner[mapped]
    return RestartOwner.UNKNOWN


# Minimal self-test. No MuJoCo / YOLO dependency.


if __name__ == "__main__":
    class DummyVisionPacket:
        def __init__(
            self,
            *,
            stable: bool,
            goal: bool,
            x: Optional[float],
            y: Optional[float],
            loc: bool = True,
            ball_state: str = "BALL_STABLE_CENTER_NEAR",
        ) -> None:
            self.ball_stable = stable
            self.goal_visible = goal
            self.goal_centered = goal
            self.ball_world_x = x
            self.ball_world_y = y
            self.localization_reliable = loc
            self.ball_state = ball_state

    print("[SELFTEST] setplay_vision_state_adapter_v2")


    # ATTACK: penalty needs goal context.

    attack_adapter = SetPlayVisionAdapter()
    attack_missing_goal = {
        "vision_packet": DummyVisionPacket(stable=True, goal=False, x=0.0, y=0.0),
        "vision_latency_status": "VALID",
    }
    pkt = attack_adapter.update_robot_state(
        attack_missing_goal,
        restart_type="penaltykick",
        restart_owner="our",
    )
    print(f"[pk_attack_target_missing] state={pkt.state.name} target={pkt.target_context_ready}:{pkt.target_context_source}")

    attack_ready = {
        "vision_packet": DummyVisionPacket(stable=True, goal=True, x=0.0, y=0.0),
        "vision_latency_status": "VALID",
    }
    pkt = attack_adapter.update_robot_state(
        attack_ready,
        restart_type="penaltykick",
        restart_owner="our",
    )
    print(f"[pk_attack_ready] state={pkt.state.name} target={pkt.target_context_ready}:{pkt.target_context_source}")


    # DEFENSE: use motion estimator output when present.

    defense_estimator = SetPlayVisionAdapter()
    defense_wait = {
        "vision_packet": DummyVisionPacket(stable=True, goal=False, x=1.0, y=0.0),
        "vision_latency_status": "VALID",
        "vision_localization_reliable": True,
        "ball_stop_status": "STATIONARY",
        "ball_stop_reason": "gt_stationary",
        "ball_speed_est": 0.0,
        "ball_fit_n": 0,
    }
    pkt = defense_estimator.update_robot_state(
        defense_wait,
        restart_type="cornerkick",
        restart_owner="opponent",
    )
    print(f"[defense_estimator_wait] state={pkt.state.name} moved={pkt.ball_moved_candidate} source={pkt.motion_evidence_source} status={pkt.motion_estimator_status}")

    defense_motion = {
        "vision_packet": DummyVisionPacket(stable=True, goal=False, x=1.2, y=0.0),
        "vision_latency_status": "VALID",
        "vision_localization_reliable": True,
        "ball_stop_status": "VALID",
        "ball_stop_reason": "valid",
        "ball_speed_est": 0.42,
        "ball_fit_n": 3,
    }
    pkt = defense_estimator.update_robot_state(
        defense_motion,
        restart_type="cornerkick",
        restart_owner="opponent",
    )
    print(f"[defense_estimator_motion] state={pkt.state.name} moved={pkt.ball_moved_candidate} source={pkt.motion_evidence_source} speed={pkt.motion_estimator_speed_mps}")


    # DEFENSE fallback: without estimator fields, anchor displacement needs
    # consecutive confirmation.

    defense_fallback = SetPlayVisionAdapter()
    fallback_samples = [
        ("anchor_init", 1.00, 0.00),
        ("anchor_cross_1", 1.14, 0.00),
        ("anchor_cross_2", 1.16, 0.00),
    ]
    for name, x, y in fallback_samples:
        sample = {
            "vision_packet": DummyVisionPacket(stable=True, goal=False, x=x, y=y),
            "vision_latency_status": "VALID",
            "vision_localization_reliable": True,
        }
        pkt = defense_fallback.update_robot_state(
            sample,
            restart_type="kickin",
            restart_owner="opponent",
        )
        print(
            f"[{name}] state={pkt.state.name} moved={pkt.ball_moved_candidate} "
            f"source={pkt.motion_evidence_source} disp={pkt.ball_anchor_displacement_m}"
        )
