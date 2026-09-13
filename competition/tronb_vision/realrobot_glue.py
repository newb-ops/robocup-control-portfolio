#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실기 코드의 관측값을 인식·판단 모듈이 사용하는 상태 형식으로 변환한다."""
import math
from typing import Any, Dict, Optional, Tuple

from .common import BallObservation, GoalObservation
from .features import attach_ball_stop_to_robot_state, build_vision_packet
from .game_controller import game_state_flags, normalize_game_controller_state
from .rule_state import build_rule_state
from .setplay import SetPlayVisionAdapter
from .vision_state import VisionStateConfig, build_rule_based_vision_packet


def make_ball_obs(*, detected: bool, dist: float, screen_x: float, screen_y: float = 0.0,
                  lost_frames: int = 0, consecutive_detect: int = 0, conf: float = 0.0,
                  last_valid_dist: float = 2.0, frame_id: int = 0) -> BallObservation:
    """ball_follower 의 _ball_obs 필드 → tronb_vision BallObservation."""
    return BallObservation(
        dist=float(dist), screen_x=float(screen_x), screen_y=float(screen_y),
        lost_frames=int(lost_frames), consecutive_detect=int(consecutive_detect),
        conf=float(conf), last_valid_dist=float(last_valid_dist),
        detected=bool(detected), frame_id=int(frame_id))


def make_goal_obs(*, detected: bool, screen_x: float = 0.0, screen_y: float = 0.0,
                  conf: float = 0.0) -> GoalObservation:
    return GoalObservation(detected=bool(detected), screen_x=float(screen_x),
                           screen_y=float(screen_y), conf=float(conf))


def build_robot_state(
    *,
    ball: BallObservation,
    goal: GoalObservation,
    loc_pose: Tuple[float, float, float],
    loc_trust: bool,
    ball_xy_robot: Optional[Tuple[float, float]],
    field,                                  # FieldMap (loc.fm)
    motion=None,                            # BallMotionSnap | None (ball_stop)
    vision_cfg: Optional[VisionStateConfig] = None,
    game_controller_state: Optional[str] = None,
    restart_type: Optional[str] = None,
    restart_owner: Optional[str] = None,
    setplay_adapter: Optional[SetPlayVisionAdapter] = None,
    goal_center_tol: float = 0.15,
) -> Dict[str, Any]:
    """ball_follower 1프레임 관측 → control_fsm/setplay/brain 이 읽는 robot_state dict."""
    vision_cfg = vision_cfg or VisionStateConfig()
    rs: Dict[str, Any] = {}

    # 공 월드좌표: localization 신뢰 + 로봇기준 공좌표 있을 때만
    bwx = bwy = None
    if loc_trust and ball_xy_robot is not None:
        x, y, th = loc_pose
        c, s = math.cos(th), math.sin(th)
        bwx = x + c * ball_xy_robot[0] - s * ball_xy_robot[1]
        bwy = y + s * ball_xy_robot[0] + c * ball_xy_robot[1]

    goal_centered = bool(goal.detected and abs(goal.screen_x) < goal_center_tol)
    rs.update({
        "ball_dist": float(ball.dist),
        "ball_screen_x": float(ball.screen_x),
        "raw_ball_offset_y": float(ball.screen_y),
        "ball_seen_streak": int(ball.consecutive_detect),
        "ball_lost_streak": int(ball.lost_frames),
        "vision_goal_visible": bool(goal.detected),
        "vision_goal_centered": goal_centered,
        "goal_screen_x": float(goal.screen_x) if goal.detected else None,
        "frame_id": int(ball.frame_id),
        "vision_localization_reliable": bool(loc_trust),
        "ball_world_x_est": bwx,
        "ball_world_y_est": bwy,
    })
    attach_ball_stop_to_robot_state(rs, motion)

    # rule_state (공 룰 상태)
    rstate = build_rule_state(bwx, bwy, field)
    rs.update({f"vision_{k}": v for k, v in rstate.as_dict().items()})

    # vision_state (공 인식 라벨 + kick 힌트)
    gc_state = normalize_game_controller_state(game_controller_state)
    locomotion, active = game_state_flags(gc_state)
    vpacket = build_rule_based_vision_packet(
        obs=ball, goal=goal, localization_reliable=loc_trust,
        ball_world_x=bwx, ball_world_y=bwy, ball_rule_status=rstate.ball_rule_status,
        game_controller_state=gc_state, game_allows_locomotion=locomotion,
        game_allows_active_play=active, cfg=vision_cfg)
    rs["vision_packet"] = vpacket
    rs["vision_packet_dict"] = vpacket.as_dict()
    rs["ball_perception_state"] = vpacket.ball_state.name
    rs["vision_kick_candidate_hint"] = vpacket.vision_kick_candidate_hint
    rs["game_controller_state"] = gc_state.name
    rs["simple_vision_packet"] = build_vision_packet(
        obs=ball, goal_visible=goal.detected, goal_centered=goal_centered,
        motion=motion, localization_reliable=loc_trust)

    # setplay (있을 때만)
    if setplay_adapter is not None:
        rs["restart_type"] = restart_type or "NONE"
        rs["restart_owner"] = restart_owner or "NONE"
        setplay_adapter.update_robot_state(rs, restart_type=restart_type, restart_owner=restart_owner)
    return rs
