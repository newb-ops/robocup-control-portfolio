#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GameController UDP 패킷을 받아 경기 상태를 갱신한다."""
import socket
import struct
import threading
import time
from enum import IntEnum
from typing import Any, Dict, Optional

GC_PORT = 3838
GC_RETURN_PORT = 3939
GC_HEADER = b"RGme"
GC_VERSION = 20

MAX_NUM_PLAYERS = 20
ROBOT_INFO_SIZE = 3
TEAM_INFO_SIZE = 10 + MAX_NUM_PLAYERS * ROBOT_INFO_SIZE
TEAMS_OFFSET = 18
PACKET_SIZE = TEAMS_OFFSET + 2 * TEAM_INFO_SIZE


class GcGameState(IntEnum):
    INITIAL = 0
    READY = 1
    SET = 2
    PLAYING = 3
    FINISHED = 4


PENALTY_MAP = {
    0: "PENALTY_NONE", 1: "PENALTY_ILLEGAL_POSITIONING", 2: "PENALTY_MOTION_IN_SET",
    3: "PENALTY_MOTION_IN_STOP", 4: "PENALTY_LOCAL_GAME_STUCK", 5: "PENALTY_INCAPABLE_ROBOT",
    6: "PENALTY_PICK_UP", 7: "PENALTY_BALL_HOLDING", 8: "PENALTY_LEAVING_THE_FIELD",
    9: "PENALTY_PLAYING_WITH_ARMS_HANDS", 10: "PENALTY_PUSHING", 11: "PENALTY_CAUTIONED",
    12: "PENALTY_SENT_OFF", 13: "PENALTY_SUBSTITUTE",
}
SET_PLAY_MAP = {
    0: "SET_PLAY_NONE", 1: "SET_PLAY_DIRECT_FREE_KICK", 2: "SET_PLAY_INDIRECT_FREE_KICK",
    3: "SET_PLAY_PENALTY_KICK", 4: "SET_PLAY_THROW_IN", 5: "SET_PLAY_GOAL_KICK",
    6: "SET_PLAY_CORNER_KICK",
}
# GC GameState → 내 common.GameControllerState 문자열(normalize_game_controller_state 가 흡수)
_GAME_STATE_TO_GCS = {0: "INITIAL", 1: "READY", 2: "SET", 3: "PLAYING", 4: "FINISHED"}


def parse_gc_packet(data: bytes, our_team_number_field: int, player_number: int):
    if len(data) < PACKET_SIZE:
        return None, f"packet too short: {len(data)} < {PACKET_SIZE}"
    if data[0:4] != GC_HEADER:
        return None, f"invalid header: {data[0:4]!r}"
    state_raw = data[10]
    if state_raw not in [s.value for s in GcGameState]:
        return None, f"unknown game_state raw={state_raw}"

    our_team = None
    for team_index in range(2):
        team_offset = TEAMS_OFFSET + team_index * TEAM_INFO_SIZE
        if data[team_offset] == our_team_number_field:
            our_team = {"team_index": team_index, "team_offset": team_offset,
                        "team_number": data[team_offset]}
            break
    if our_team is None:
        return None, (f"our team not found (field={our_team_number_field}, "
                      f"team0={data[TEAMS_OFFSET]}, team1={data[TEAMS_OFFSET + TEAM_INFO_SIZE]})")
    if not (1 <= player_number <= MAX_NUM_PLAYERS):
        return None, f"invalid player_number={player_number}"

    robot_offset = our_team["team_offset"] + 10 + (player_number - 1) * ROBOT_INFO_SIZE
    parsed = {
        "version": data[4], "packet_number": data[5], "players_per_team": data[6],
        "competition_type": data[7], "stopped": data[8], "game_phase": data[9],
        "game_state": GcGameState(state_raw), "state_raw": state_raw,
        "set_play_raw": data[11], "set_play_name": SET_PLAY_MAP.get(data[11], "UNKNOWN"),
        "first_half": data[12], "kicking_team": data[13],
        "secs_remaining": struct.unpack_from("<h", data, 14)[0],
        "secondary_time": struct.unpack_from("<h", data, 16)[0],
        "team_index": our_team["team_index"], "team_number_field": our_team["team_number"],
        "player_number": player_number,
        "penalty": data[robot_offset], "penalty_name": PENALTY_MAP.get(data[robot_offset], "UNKNOWN_PENALTY"),
        "secs_till_unpenalised": data[robot_offset + 1], "cautions": data[robot_offset + 2],
    }
    return parsed, None


def build_rgrt_response(team_number_field: int, player_number: int) -> bytes:
    # RoboCupGameControlReturnData: header, version, player, team, message,
    # pose[3](x,y,theta), ballAge, ball[2](x,y) = 6 floats. (원본 step6 은 7f 로 오타→crash)
    return struct.pack("<4sBBBB6f", b"RGrt", 4, player_number & 0xFF,
                       team_number_field & 0xFF, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)


def gc_to_robot_state(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """GC 파싱 결과 → robot_state 에 머지할 dict. penalty≠0 이면 PENALIZED."""
    if parsed["penalty"] != 0:
        gcs = "PENALIZED"
    else:
        gcs = _GAME_STATE_TO_GCS.get(int(parsed["game_state"]), "UNKNOWN")
    set_play_name = parsed["set_play_name"]            # "SET_PLAY_PENALTY_KICK" 등
    is_ours = (parsed["kicking_team"] == parsed["team_number_field"])
    active_restart = parsed["set_play_raw"] != 0
    return {
        "game_controller_state": gcs,
        "gc_penalty": parsed["penalty"],
        "gc_penalty_name": parsed["penalty_name"],
        "gc_secs_till_unpenalised": parsed["secs_till_unpenalised"],
        "gc_kicking_team": parsed["kicking_team"],
        "gc_secs_remaining": parsed["secs_remaining"],
        # setplay (4단계 setplay.py 입력): 원본 이름 + setplay.py 가 직접 읽는 정규화 키.
        "setplay_restart_type": set_play_name,
        "setplay_restart_raw": parsed["set_play_raw"],
        "setplay_is_ours": is_ours,
        # setplay._normalize_* 가 받는 형태: "SET_PLAY_" 접두사 제거(alias 흡수), owner 는 우리/상대.
        "restart_type": set_play_name.replace("SET_PLAY_", "") if active_restart else "NONE",
        "restart_owner": ("OUR_TEAM" if is_ours else "OPPONENT_TEAM") if active_restart else "NONE",
    }


class GcReceiver:
    """백그라운드 GC 수신 스레드. snapshot() 으로 robot_state 머지 dict 제공."""
    def __init__(self, team_number: int, player_number: int,
                 port: int = GC_PORT, respond: bool = True, verbose: bool = False):
        self.team_field = team_number & 0xFF
        self.player = player_number
        self.port = port
        self.respond = respond
        self.verbose = verbose
        self._latest: Dict[str, Any] = {"game_controller_state": "UNKNOWN"}
        self._lock = threading.Lock()
        self._run = False
        self._th: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self.last_rx_ts = 0.0

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(1.0)
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self) -> None:
        resp = build_rgrt_response(self.team_field, self.player)
        while self._run:
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed, err = parse_gc_packet(data, self.team_field, self.player)
            if err:
                if self.verbose:
                    print(f"[GC] parse fail: {err}")
                continue
            upd = gc_to_robot_state(parsed)
            with self._lock:
                self._latest = upd
                self.last_rx_ts = time.time()
            if self.verbose:
                print(f"[GC] state={upd['game_controller_state']} "
                      f"setplay={upd['setplay_restart_type']} penalty={upd['gc_penalty']}")
            if self.respond:
                try:
                    self._sock.sendto(resp, (addr[0], GC_RETURN_PORT))
                except OSError:
                    pass

    def snapshot(self) -> Dict[str, Any]:
        """robot_state.update(gc.snapshot()) 형태로 머지. 수신 없으면 UNKNOWN."""
        with self._lock:
            return dict(self._latest)

    def is_fresh(self, max_age: float = 3.0) -> bool:
        with self._lock:
            return self.last_rx_ts > 0 and (time.time() - self.last_rx_ts) < max_age

    def stop(self) -> None:
        self._run = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
