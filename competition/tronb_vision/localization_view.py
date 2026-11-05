from __future__ import annotations
"""필드, 파티클과 추정 위치를 별도 OpenCV 창에 표시한다.

필드 배경은 캐시하고 파티클은 NumPy 인덱싱으로 그린다.
TopViewManager가 표시 여부와 갱신 간격을 관리한다."""
from typing import List, Optional

import numpy as np

_LM_COLOR = {"L": (90, 200, 90), "T": (90, 200, 230), "X": (230, 180, 90), "goalpost": (90, 120, 240)}
_REAL_LM_COLOR = {
    "line_l": (80, 80, 255),
    "line_t": (0, 220, 255),
    "line_x": (70, 230, 70),
    "goal": (255, 170, 60),
    "goal_bar": (255, 170, 60),
    "goalpost": (255, 170, 60),
    "center_mark": (255, 255, 120),
    "penalty_mark": (255, 255, 120),
    "corner_arc": (120, 220, 120),
    "robot": (220, 120, 255),
}


class FieldTopView:
    def __init__(self, field_map, size_px: int = 360, pad: int = 18, particle_max: int = 600):
        self.fm = field_map
        self.W = size_px
        self.pad = pad
        self.particle_max = particle_max
        L, Wd = field_map.half_len, field_map.half_wid
        # 가로:세로 = 2L : 2W 비율 유지
        avail_w = size_px - 2 * pad
        self.scale = avail_w / (2 * L)
        self.H = int(2 * Wd * self.scale + 2 * pad)
        self.cx = self.W // 2
        self.cy = self.H // 2
        self._bg = self._render_bg()

    def _px(self, x, y):
        return int(self.cx + x * self.scale), int(self.cy - y * self.scale)

    def _render_bg(self) -> np.ndarray:
        import cv2
        img = np.full((self.H, self.W, 3), (35, 70, 35), dtype=np.uint8)  # 잔디톤
        L, Wd = self.fm.half_len, self.fm.half_wid
        white = (210, 210, 210)
        # 외곽 라인
        p1 = self._px(-L, Wd); p2 = self._px(L, -Wd)
        cv2.rectangle(img, p1, p2, white, 1)
        # 센터 라인 + 원
        cv2.line(img, self._px(0, Wd), self._px(0, -Wd), white, 1)
        cv2.circle(img, self._px(0, 0), int(1.5 * self.scale), white, 1)
        # 맵 랜드마크
        for lm in self.fm.landmarks:
            u, v = self._px(lm.x, lm.y)
            cv2.circle(img, (u, v), 3, _LM_COLOR.get(lm.ltype, (200, 200, 200)), -1)
        cv2.putText(img, "FIELD TOP VIEW", (self.pad, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        return img

    def _robot_to_world(self, loc, rx: float, ry: float):
        x, y, th = loc.pose
        c, s = np.cos(th), np.sin(th)
        return x + c * rx - s * ry, y + s * rx + c * ry

    def render(self, loc, show_landmark_comp: bool = False,
               landmark_obs: Optional[List] = None,
               real_ball_xy=None,
               real_ball_method: str = "none",
               real_landmarks: Optional[List] = None) -> np.ndarray:
        import cv2
        img = self._bg.copy()   # 정적 필드 배경 재사용
        pf = loc.pf

        # 파티클을 픽셀 좌표로 변환해 한 번에 그린다.
        p = pf.p
        n = len(p)
        if n > self.particle_max:
            idx = np.linspace(0, n - 1, self.particle_max).astype(int)
            p = p[idx]
        ui = (self.cx + p[:, 0] * self.scale).astype(int)
        vi = (self.cy - p[:, 1] * self.scale).astype(int)
        m = (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
        img[vi[m], ui[m]] = (60, 160, 255)   # 주황 점들

        # 추정 pose (화살표)
        x, y, th = loc.pose
        pu, pv = self._px(x, y)
        hu = int(pu + np.cos(th) * 0.6 * self.scale)
        hv = int(pv - np.sin(th) * 0.6 * self.scale)
        trust = loc.trust()
        pose_col = (80, 255, 80) if trust else (80, 80, 255)
        cv2.circle(img, (pu, pv), 4, pose_col, -1)
        cv2.arrowedLine(img, (pu, pv), (hu, hv), pose_col, 2, tipLength=0.4)

        # 공격 골대
        gx, gy = loc.fm.attack_goal_xy(loc.attack_positive_x)
        cv2.drawMarker(img, self._px(gx, gy), (0, 215, 255), cv2.MARKER_TILTED_CROSS, 12, 2)

        # landmark_comp: 검출 랜드마크 world 점 (pose 기준 변환)
        if show_landmark_comp and landmark_obs:
            c, s = np.cos(th), np.sin(th)
            for o in landmark_obs:
                wx = x + c * o.rx - s * o.ry
                wy = y + s * o.rx + c * o.ry
                u, v = self._px(wx, wy)
                if 0 <= u < self.W and 0 <= v < self.H:
                    cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_DIAMOND, 8, 1)

        # RealSense geometry: detected landmarks and ball projected onto field
        n_real_used = 0
        for det in real_landmarks or []:
            if getattr(det, "robot_xy", None) is None:
                continue
            rx, ry = float(det.robot_xy[0]), float(det.robot_xy[1])
            wx, wy = self._robot_to_world(loc, rx, ry)
            u, v = self._px(wx, wy)
            if 0 <= u < self.W and 0 <= v < self.H:
                col = _REAL_LM_COLOR.get(getattr(det, "obs_type", ""), (230, 230, 230))
                marker = cv2.MARKER_DIAMOND if getattr(det, "used_for_pf", False) else cv2.MARKER_CROSS
                cv2.drawMarker(img, (u, v), col, marker, 9, 1)
                n_real_used += int(bool(getattr(det, "used_for_pf", False)))

        if real_ball_xy is not None:
            wx, wy = self._robot_to_world(loc, float(real_ball_xy[0]), float(real_ball_xy[1]))
            u, v = self._px(wx, wy)
            if 0 <= u < self.W and 0 <= v < self.H:
                cv2.circle(img, (u, v), 5, (0, 140, 255), -1)
                cv2.circle(img, (u, v), 9, (0, 220, 255), 1)
                cv2.putText(img, f"ball {real_ball_method}", (min(u + 8, self.W - 85), max(14, v - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)

        # 상태 텍스트
        sx, sy, sth = pf.spread()
        txt = f"trust={trust} n={loc.n_landmarks} spread={np.hypot(sx,sy):.2f}m"
        cv2.putText(img, txt, (self.pad, self.H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    pose_col, 1, cv2.LINE_AA)
        if real_landmarks or real_ball_xy is not None:
            real_txt = f"real lmk={len(real_landmarks or [])}/{n_real_used} ball={'Y' if real_ball_xy is not None else 'N'}"
            cv2.putText(img, real_txt, (self.pad, self.H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (180, 220, 220), 1, cv2.LINE_AA)
        return img


class TopViewManager:
    """위치추정 표시가 켜졌을 때만 필드 창을 열고 지정한 간격으로 갱신한다."""
    def __init__(self, win_name: str = "localization (top view)", every: int = 1):
        self.win = win_name
        # every를 늘리면 그리는 횟수를 줄일 수 있다.
        self.every = max(1, every)
        self._view: Optional[FieldTopView] = None
        self._open = False
        self._k = 0

    def update(self, bf):
        try:
            import cv2
        except Exception:
            return
        real_pos = getattr(bf, "real_pos", None)
        real_snap = getattr(real_pos, "last_snap", None) if real_pos is not None else None
        want = (bf.features["localization"].visible
                or (real_snap is not None and (real_snap.ball_xy is not None or real_snap.n_landmarks > 0)))
        if not want:
            if self._open:
                cv2.destroyWindow(self.win); self._open = False
            return
        self._k += 1
        if self._open and (self._k % self.every) != 0:
            return  # throttle: 창 유지하되 이번 루프는 렌더 skip
        if self._view is None:
            self._view = FieldTopView(bf.loc.fm)
        img = self._view.render(bf.loc,
                                show_landmark_comp=bf.features["localization"].visible,
                                landmark_obs=getattr(bf, "_last_landmark_obs", []),
                                real_ball_xy=getattr(real_snap, "ball_xy", None),
                                real_ball_method=getattr(real_snap, "ball_method", "none"),
                                real_landmarks=getattr(real_pos, "last_landmarks", []) if real_pos is not None else [])
        if not self._open:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            self._open = True
        cv2.imshow(self.win, img)

    def close(self):
        if self._open:
            try:
                import cv2; cv2.destroyWindow(self.win)
            except Exception:
                pass
            self._open = False
