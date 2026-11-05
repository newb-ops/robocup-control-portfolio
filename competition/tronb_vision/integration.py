from __future__ import annotations
"""인식 상태를 제어 FSM에 전달하고 명령과 상태 전이를 기록한다.

RealStubDriver를 연결하면 실제 구동 없이 흐름을 확인할 수 있다.
JSONL 경로를 지정하면 매 프레임의 입력과 명령을 저장한다."""

import json
import time
from typing import Any, Dict, Optional


class JsonlLogger:
    """프레임별 상태와 명령을 JSONL로 기록한다."""
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class ControlBridge:
    """인식 상태를 FSM에 전달하고 드라이버 명령과 로그를 출력한다."""
    def __init__(self, driver, jsonl_path: Optional[str] = None, verbose: bool = True):
        from .control_fsm import BasicRobotStateFSM
        self.fsm = BasicRobotStateFSM()
        self.driver = driver
        self.verbose = verbose
        self.logger = JsonlLogger(jsonl_path) if jsonl_path else None
        self.last_result = None
        self._n = 0

    def step(self, robot_state: Dict[str, Any], lower: Optional[Dict[str, Any]] = None):
        lower = lower or {"is_fallen": False}
        result = self.fsm.update(lower, robot_state)
        cmd = result.command
        # 드라이버 전송 (RealStubDriver 면 no-op → 로그로만 확인).
        self.driver.send_twist(float(cmd.target_body_vx), float(cmd.target_body_wz),
                               is_kick=bool(cmd.kick_trigger))
        self.last_result = result
        self._n += 1

        if self.verbose and result.changed:
            print(f"[CTRL] {result.prev_state.name} → {result.state.name} "
                  f"({result.reason}) | ball={robot_state.get('ball_perception_state')} "
                  f"kick_hint={robot_state.get('vision_kick_candidate_hint')} "
                  f"→ vx={cmd.target_body_vx:+.2f} wz={cmd.target_body_wz:+.2f} kick={cmd.kick_trigger}")

        if self.logger is not None:
            self.logger.write({
                "t": round(time.time(), 3),
                "n": self._n,
                # 비전 입력 라벨
                "ball_perception_state": robot_state.get("ball_perception_state"),
                "ball_dist": robot_state.get("ball_dist"),
                "ball_screen_x": robot_state.get("ball_screen_x"),
                "vision_kick_candidate_hint": robot_state.get("vision_kick_candidate_hint"),
                "vision_goal_visible": robot_state.get("vision_goal_visible"),
                "vision_goal_centered": robot_state.get("vision_goal_centered"),
                "vision_localization_reliable": robot_state.get("vision_localization_reliable"),
                "ball_world_x_est": robot_state.get("ball_world_x_est"),
                "ball_world_y_est": robot_state.get("ball_world_y_est"),
                "vision_ball_rule_status": robot_state.get("vision_ball_rule_status"),
                "ball_stop_status": robot_state.get("ball_stop_status"),
                # 제어 출력
                "fsm_state": result.state.name,
                "fsm_changed": bool(result.changed),
                "cmd_vx": round(float(cmd.target_body_vx), 3),
                "cmd_wz": round(float(cmd.target_body_wz), 3),
                "cmd_kick": bool(cmd.kick_trigger),
                "cmd_recover": bool(cmd.recover_trigger),
            })
        return result

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
