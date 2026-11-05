from __future__ import annotations
"""OpenCV 영상 위에 탐지 결과와 제어 상태를 표시한다."""
from typing import Optional

import math
import numpy as np

from .common import now_sec

# 기능 키 번호 라벨 (1~9)
FEAT_KEYS = [("1", "loc"), ("2", "stop")]   # detector 는 상시 ON (토글 없음)
MODE_COL = {"LOCKED": (60, 60, 230), "AUTO": (60, 220, 60), "MANUAL": (40, 200, 235)}
BOX_COL = {
    "ball": (0, 165, 255),
    "goal": (255, 170, 60),
    "goal_bar": (255, 170, 60),
    "goalpost": (255, 170, 60),
    "line_l": (80, 80, 255),
    "line_t": (0, 220, 255),
    "line_x": (70, 230, 70),
    "center_mark": (255, 255, 120),
    "penalty_mark": (255, 255, 120),
    "corner_arc": (120, 220, 120),
    "robot": (220, 120, 255),
}


def _text(img, txt, org, scale=0.5, col=(230, 230, 230), thick=1):
    import cv2
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, alpha=0.45):
    import cv2
    sub = img[y:y + h, x:x + w]
    if sub.size:
        img[y:y + h, x:x + w] = (sub.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _box_values(box):
    cid = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)
    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)
    xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, "cpu") else np.asarray(box.xyxy)[0]
    cname = str(getattr(box, "cls_name", cid))
    return cid, cname, conf, [float(v) for v in xyxy]


def _box_color(name: str):
    key = str(name).strip().lower()
    return BOX_COL.get(key, (80, 220, 180))


HELP_LINES = [
    "[KEYS]",
    "P : AUTO (tracking)   M : MANUAL   X : LOCKED",
    "1 : localization on/off   2 : ball_stop on/off",
    "L : 위치추정 재추정(global)   O : 우리 진영",
    "R : 트랙 리셋   C : 프레임 캡처",
    "H : help on/off   ESC/Q : quit",
]


def draw_overlay(bgr, bf, toast: Optional[tuple] = None, show_help: bool = False) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return bgr
    img = bgr.copy()
    h, w = img.shape[:2]
    obs, goal = bf.snapshot()
    feat = getattr(bf, "feat_out", {}) or {}

    # 상단 헤더 바
    _ext = getattr(bf, "external_control", False) and getattr(bf, "control_bridge", None) is not None
    _bar(img, 0, 0, w, 80 if _ext else 64)
    mcol = MODE_COL.get(bf.mode.name, (230, 230, 230))
    _text(img, f"{bf.mode.name}", (8, 22), 0.7, mcol, 2)
    _text(img, f"STATE {bf.state.name}", (160, 22), 0.6, (230, 230, 230), 1)
    # external_control: 제어 FSM 상태 + 비전 인식 라벨 + 킥을 별도 줄에 표시 (단독 검증 가시화)
    if getattr(bf, "external_control", False) and getattr(bf, "control_bridge", None) is not None:
        res = bf.control_bridge.last_result
        rs = getattr(bf, "robot_state", {}) or {}
        bp = rs.get("ball_perception_state", "-")
        kh = rs.get("vision_kick_candidate_hint", False)
        if res is not None:
            cmd = res.command
            kick = cmd.kick_trigger
            ccol = (60, 80, 240) if kick else (90, 230, 230)
            _text(img, f"CTRL {res.state.name}  bp={bp}  hint={'Y' if kh else 'N'}  "
                       f"vx={cmd.target_body_vx:+.2f} wz={cmd.target_body_wz:+.2f} "
                       f"{'KICK!' if kick else ''}", (8, 74), 0.5, ccol, 1)
    # 카메라 pitch (floor 투영에 쓰는 실제 각도) — 수치 + 출처. IMU=초록, manual=주황.
    rp_snap = feat.get("real_position")
    if rp_snap is not None:
        psrc = getattr(rp_snap, "pitch_source", "manual")
        pcol = (90, 230, 90) if psrc == "imu" else (60, 170, 250)
        _text(img, f"pitch {rp_snap.pitch_down_deg:+.1f}deg [{psrc}]", (350, 22), 0.6, pcol, 2)
    infer = getattr(bf.detector, "last_infer_ms", 0.0)
    _text(img, f"ball={'Y' if obs.detected else 'N'} d={obs.dist:.2f} sx={obs.screen_x:+.2f} "
               f"lost={obs.lost_frames} infer={infer:.0f}ms", (8, 46), 0.5, (180, 230, 180), 1)

    # detector boxes
    boxes = getattr(bf, "_last_boxes", []) or []
    for box in boxes:
        try:
            _cid, cname, conf, xyxy = _box_values(box)
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        except Exception:
            continue
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        col = _box_color(cname)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{cname} {conf:.2f}"
        ly = max(16, y1 - 6)
        _text(img, label, (x1 + 2, ly), 0.43, col, 1)

    # 공 마커
    if obs.detected:
        cx = int(obs.screen_x * (w / 2) + w / 2)
        cy = int(obs.screen_y * (h / 2) + h / 2)
        cv2.line(img, (w // 2, h // 2), (cx, cy), (0, 140, 255), 1)
        cv2.circle(img, (cx, cy), 11, (0, 165, 255), 2)
    cv2.drawMarker(img, (w // 2, h // 2), (120, 120, 120), cv2.MARKER_CROSS, 14, 1)

    # 우측 기능 출력 (visible 일 때만)
    rx, ry = w - 232, 86
    def put_r(txt, col=(220, 220, 90)):
        nonlocal ry
        _text(img, txt, (rx, ry), 0.45, col, 1); ry += 17
    F = bf.features
    if F["ball_stop"].visible and feat.get("ball_stop"):
        b = feat["ball_stop"]
        if b.moving and b.stop_x is not None:
            put_r(f"stop({b.stop_x:.2f},{b.stop_y:.2f}) eta{b.eta:.1f}")
        elif b.local_x is not None:
            put_r(f"ball L({b.local_x:.2f},{b.local_y:.2f}) still")
    if F["localization"].visible:
        x, y, th = bf.loc.pose
        put_r(f"loc trust={bf.loc.trust()} n={bf.loc.n_landmarks} "
              f"({x:.1f},{y:.1f},{math.degrees(th):.0f}°)  [L]=재추정")
    if feat.get("real_position"):
        rp = feat["real_position"]
        put_r(f"real pitch={rp.pitch_down_deg:+.1f} {rp.pitch_source}", (180, 220, 220))
        if rp.ball_xy is not None:
            put_r(f"real ball({rp.ball_xy[0]:.2f},{rp.ball_xy[1]:.2f}) {rp.ball_method}", (180, 220, 220))

    # 하단 토글 타일 (키 번호 + 상태색)
    _bar(img, 0, h - 30, w, 30)
    x = 8
    names = list(bf.features.keys())
    for (key, short), name in zip(FEAT_KEYS, names):
        st = bf.features[name].name
        col = {"ON": (60, 220, 60), "SEMI": (40, 190, 235), "OFF": (110, 110, 110)}[st]
        _text(img, f"{key}:{short}", (x, h - 10), 0.42, col, 1)
        x += 70

    # 토스트 (키 누르면 잠깐 표시)
    if toast and (now_sec() - toast[1] < 1.2):
        msg = toast[0]
        tw = 10 + len(msg) * 11
        _bar(img, w // 2 - tw // 2, h // 2 + 40, tw, 30, alpha=0.6)
        _text(img, msg, (w // 2 - tw // 2 + 6, h // 2 + 60), 0.6, (255, 255, 255), 2)

    # 도움말
    if show_help:
        _bar(img, w // 2 - 200, h // 2 - 110, 400, 200, alpha=0.65)
        yy = h // 2 - 84
        for ln in HELP_LINES:
            _text(img, ln, (w // 2 - 184, yy), 0.5, (240, 240, 240), 1); yy += 26
    else:
        _text(img, "H=help", (w - 70, h - 38), 0.42, (160, 160, 160), 1)
    return img


def handle_key(bf, key: int, ui: dict, allow_manual_drive: bool = True) -> dict:
    """키 입력 처리 (run_sim/run_camera 공용). 대/소문자 모두 허용.
    ui = {'toast':(msg,t), 'show_help':bool, 'running':bool}. 갱신해서 반환."""
    from .common import OperationMode, TriState
    if key in (255, -1, 0):   # 입력 없음
        return ui
    k = chr(key).lower() if 0 <= key < 256 else ""

    def toast(msg):
        ui["toast"] = (msg, now_sec())

    if key == 27 or k == "q":
        ui["running"] = False
    elif k == "p":
        bf.set_mode(OperationMode.AUTO); toast("MODE: AUTO")
    elif k == "m":
        bf.set_mode(OperationMode.MANUAL); bf._manual_twist = (0.0, 0.0); toast("MODE: MANUAL")
    elif k == "x":
        bf.set_mode(OperationMode.LOCKED); toast("MODE: LOCKED")
    elif k == "h":
        ui["show_help"] = not ui.get("show_help", False)
    elif k == "l":
        bf.reset_localization("global")
        toast("LOC RE-LOCALIZE (global, 아무데서나)")
    elif k == "o":
        bf.reset_localization("own_half")
        toast("LOC RESET (우리 진영)")
    elif k == "r":
        if hasattr(bf, "reset_vision_tracks"):
            bf.reset_vision_tracks()
        toast("TRACKS RESET")
    elif k == "c":
        path = bf.capture_debug_frame() if hasattr(bf, "capture_debug_frame") else None
        toast("CAPTURED" if path else "NO FRAME")
    elif k in "123456789":
        names = list(bf.features.keys())
        idx = int(k) - 1
        if idx < len(names):
            bf.toggle_feature(names[idx])
            toast(f"{names[idx]} -> {bf.features[names[idx]].name}")
    elif allow_manual_drive and bf.mode == OperationMode.MANUAL:
        # WASD 수동 주행 (MANUAL 일 때만)
        v = bf.cfg.max_linear_speed; wz = bf.cfg.max_angular_speed
        if k == "w":
            bf._manual_twist = (v, 0.0); toast("fwd")
        elif k == "s":
            bf._manual_twist = (-v, 0.0); toast("back")
        elif k == "a":
            bf._manual_twist = (0.0, wz); toast("left")
        elif k == "d":
            bf._manual_twist = (0.0, -wz); toast("right")
        elif k == " ":
            bf._manual_twist = (0.0, 0.0); toast("stop")
    return ui
