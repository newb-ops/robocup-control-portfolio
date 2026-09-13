from __future__ import annotations
"""
localization.py — compact MCL (기준: v6 ParticleFilter + trust/spread).
v10.3 patch: center_mark / penalty_mark / corner_arc 직접 지원.

기본 OFF. 켜기 전 필수(아직 미해결):
  - 카메라 자세 입력: set_pose_matrices(head/IMU) 를 매 프레임 줘야 pixel_to_floor 정확 (주의 5).
  - 초기 시딩: 점대칭 필드라 initial_pose 를 킥오프 자리+IMU heading 으로 박아야 함 (주의 4).
신뢰 판단은 confidence(ESS) 가 아니라 spread + 관측 랜드마크 수 (trust). (주의 2/3)

[patch 변경점]
- SUPPORTED_OBS_TYPES: 기존 4종 + center_mark / penalty_mark / corner_arc
- YOLO_CLS_TO_OBS_TYPE: class name → obs_type 변환 dict (app._landmark_obs 에서 사용)
- FieldMap._default: penalty_mark / corner_arc 를 X/L 에서 분리해 전용 type 으로 등록
- FieldMap.__init__: _by_type 캐시를 7종으로 확장
- ParticleFilter.sigma / md: 신형 3종 기본값 추가
- ParticleFilter.update: md 계산을 obs_type 별로 확장 (L/T tight, 나머지 기본)
- Observation 필드명/인터페이스 변경 없음 (rx, ry 유지)
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# 지원 obs_type 목록 + YOLO class → obs_type 매핑

SUPPORTED_OBS_TYPES = ("L", "T", "X", "goalpost",
                        "center_mark", "penalty_mark", "corner_arc")

# app._landmark_obs 에서 이 dict 로 변환
YOLO_CLS_TO_OBS_TYPE: Dict[str, str] = {
    "line_l":       "L",
    "line_t":       "T",
    "line_x":       "X",
    "center_mark":  "center_mark",
    "penalty_mark": "penalty_mark",
    "corner_arc":   "corner_arc",
    "goal_bar":     "goalpost",
    "goalpost":     "goalpost",
}
# None 을 반환해야 하는 class (ball/robot/goal 등) 는 dict 에 넣지 않음


@dataclass
class Observation:
    obs_type: str   # SUPPORTED_OBS_TYPES 중 하나
    rx: float       # robot-relative x (forward)
    ry: float       # robot-relative y (left)


class CoordinateTransform:
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0, h_cam=1.25):
        self.fx, self.fy, self.cx, self.cy, self.h_cam = fx, fy, cx, cy, h_cam
        self.R_cam = np.eye(3); self.R_base = np.eye(3)
        self.depth_min, self.depth_max = 0.05, 15.0

    def set_pose_matrices(self, R_cam: Optional[np.ndarray], R_base: Optional[np.ndarray]):
        if R_cam is not None: self.R_cam = np.asarray(R_cam).reshape(3, 3)
        if R_base is not None: self.R_base = np.asarray(R_base).reshape(3, 3)

    def pixel_to_floor_direct(self, u, v, depth) -> Optional[Tuple[float, float]]:
        if depth is None or not (self.depth_min <= depth <= self.depth_max):
            return None
        x = (u - self.cx) / self.fx; y = (v - self.cy) / self.fy
        ray = np.array([x, -y, -1.0]); ray /= np.linalg.norm(ray)
        ray_base = (self.R_base.T @ self.R_cam) @ ray
        cam_pos = np.array([0.0, 0.0, self.h_cam])
        if ray_base[2] >= -1e-6:
            return None
        t = -cam_pos[2] / ray_base[2]
        hit = cam_pos + t * ray_base
        d = math.hypot(hit[0], hit[1])
        if d < 0.05 or d > 15.0:
            return None
        return float(hit[0]), float(hit[1])


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    ltype: str


class FieldMap:
    def __init__(self, half_len=7.0, half_wid=4.5, field_mode="full"):
        self.half_len, self.half_wid = half_len, half_wid
        self.field_mode = field_mode
        all_lm = self._default(half_len, half_wid)
        # 하프코트 제한: 물리적으로 존재하는 절반의 랜드마크만 사용 → 점대칭 모호성 제거.
        # x_pos: x>=0 (골 +x 쪽 절반)   x_neg: x<=0   full: 전체(대회용)
        if field_mode == "x_pos":
            self.landmarks = [l for l in all_lm if l.x >= -1e-6]
            self.x_lo, self.x_hi = 0.0, half_len
        elif field_mode == "x_neg":
            self.landmarks = [l for l in all_lm if l.x <= 1e-6]
            self.x_lo, self.x_hi = -half_len, 0.0
        else:
            self.landmarks = all_lm
            self.x_lo, self.x_hi = -half_len, half_len
        # 7종으로 캐시 확장
        self._by_type = {}
        for t in SUPPORTED_OBS_TYPES:
            arr = np.array([[l.x, l.y] for l in self.landmarks if l.ltype == t], dtype=float)
            self._by_type[t] = arr if arr.size else np.zeros((0, 2))

    @staticmethod
    def _default(L, W) -> List[Landmark]:
        """M-Field 랜드마크 — 검증된 params.yaml 과 동일 좌표로 통일.
        (이전 하드코드가 T/X/L 을 실제 필드와 다르게 만들어 위치추정이 틀렸음 → 수정.)
        원점=정중앙, +x=오른쪽 골, +y=위 사이드라인.
        corner_arc / penalty_mark 는 params.yaml 엔 없지만 신형 모델 클래스라 추가."""
        out: List[Landmark] = []

        # L (8) : 박스 안쪽 직각 코너만. 외곽 코너는 corner_arc, 골라인 교차는 T (MAP 기준).
        for sx in (-1, 1):
            out.append(Landmark(sx * 6.0,  2.0, "L"))   # 골에어리어 앞 코너
            out.append(Landmark(sx * 6.0, -2.0, "L"))
            out.append(Landmark(sx * 4.0,  3.0, "L"))   # 페널티에어리어 앞 코너
            out.append(Landmark(sx * 4.0, -3.0, "L"))

        # T (10) : 사이드라인 + 페널티/골에어리어 변이 골라인과 만나는 지점
        out.append(Landmark(0.0,  4.5, "T"))            # 센터라인-사이드라인
        out.append(Landmark(0.0, -4.5, "T"))
        for sx in (-1, 1):
            out.append(Landmark(sx * 7.0,  3.0, "T"))   # 페널티 골라인 T
            out.append(Landmark(sx * 7.0, -3.0, "T"))
            out.append(Landmark(sx * 7.0,  2.0, "T"))   # 골에어리어 골라인 T
            out.append(Landmark(sx * 7.0, -2.0, "T"))

        # X (2) : 센터서클이 센터라인과 만나는 위/아래 교차 (MAP 의 보라 박스 = (0,±1.5)).
        out.append(Landmark(0.0,  1.5, "X"))
        out.append(Landmark(0.0, -1.5, "X"))

        # goalpost (4)
        for sx in (-1, 1):
            out.append(Landmark(sx * 7.0,  1.25, "goalpost"))
            out.append(Landmark(sx * 7.0, -1.25, "goalpost"))

        # corner_arc (4) : 외곽 코너에 위치 (신형 모델 클래스)
        for sx in (-1, 1):
            out.append(Landmark(sx * 7.0,  4.5, "corner_arc"))
            out.append(Landmark(sx * 7.0, -4.5, "corner_arc"))

        # penalty_mark (2) : 골라인에서 2m (신형 모델 클래스)
        out.append(Landmark( 5.0, 0.0, "penalty_mark"))
        out.append(Landmark(-5.0, 0.0, "penalty_mark"))

        # center_mark 만 미설치(모델 클래스에도 없음) → 제외.
        return out

    def nearest_batch(self, xy: np.ndarray, ltype: str) -> np.ndarray:
        arr = self._by_type.get(ltype)
        if arr is None or len(arr) == 0:
            return np.full(len(xy), 1e9)
        return np.linalg.norm(xy[:, None, :] - arr[None, :, :], axis=2).min(axis=1)

    def attack_goal_xy(self, attack_positive_x=True) -> Tuple[float, float]:
        return (self.half_len if attack_positive_x else -self.half_len), 0.0


# PF sigma / match_dist 기본값 (type별)
_SIGMA: Dict[str, float] = {
    "L":            0.35,
    "T":            0.40,
    "X":            0.45,
    "goalpost":     0.35,
    "center_mark":  0.30,   # 위치 확실, tight
    "penalty_mark": 0.30,
    "corner_arc":   0.35,
}
_MATCH_DIST: Dict[str, float] = {
    "L":            1.50,   # params.yaml 과 통일 (0.5 는 투영노이즈에 너무 타이트 → 초기수렴 불가)
    "T":            1.00,
    "X":            1.00,
    "goalpost":     1.50,
    "center_mark":  0.80,
    "penalty_mark": 0.80,
    "corner_arc":   1.00,
}


class ParticleFilter:
    def __init__(self, field_map: FieldMap, n=600, init_pose=None,
                 lt_max_match_dist=None, init_mode="uniform"):
        self.fm = field_map; self.n = n
        self.motion_noise = np.array([0.05, 0.05, 0.02])
        # sigma / match_dist: 기본값(_MATCH_DIST, params.yaml 과 통일) 사용.
        self.sigma = dict(_SIGMA)
        self.match_dist = dict(_MATCH_DIST)
        if lt_max_match_dist is not None:      # 명시 시에만 L/T 오버라이드
            self.match_dist["L"] = lt_max_match_dist
            self.match_dist["T"] = lt_max_match_dist
        self.p = np.zeros((n, 3))
        self.w = np.ones(n) / n
        # 연속성/회복 안정화 상태 (검증된 독립 PF 에서 이식).
        # _tracking: spread 가 작아져 '수렴됨' 으로 판단된 뒤에만 연속성 패널티 적용
        # (uniform global 시작이라 수렴 전엔 패널티 끄고 자유 탐색 → 어디서든 수렴 가능).
        self.prev_est = None
        self._tracking = False
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._lowess_count = 0   # ESS<1% 가 몇 프레임 연속인지 (kidnap 회복 지속 트리거용)
        if init_pose is not None:
            self.seed(*init_pose)                      # 알고있는 위치(시딩)
        elif init_mode == "own_half":
            self.reset_own_half()                      # 우리 진영 절반 (대회 킥오프 전)
        else:
            self.reset_uniform()                       # 필드 전체 (global localization, 아무데서나 시작)

    def seed(self, x, y, th, sxy=0.3, sth_deg=10.0):
        """알고 있는 pose 근처로 파티클 집중."""
        self.p[:, 0] = np.clip(x + np.random.randn(self.n) * sxy, self.fm.x_lo, self.fm.x_hi)
        self.p[:, 1] = np.clip(y + np.random.randn(self.n) * sxy, -self.fm.half_wid, self.fm.half_wid)
        self.p[:, 2] = th + np.random.randn(self.n) * math.radians(sth_deg)
        self.w[:] = 1.0 / self.n

    def reset_uniform(self):
        """필드 전체 + 모든 방향으로 균일 분포 (어디서 시작하든 추정 — global localization)."""
        self.p[:, 0] = np.random.uniform(self.fm.x_lo, self.fm.x_hi, self.n)
        self.p[:, 1] = np.random.uniform(-self.fm.half_wid, self.fm.half_wid, self.n)
        self.p[:, 2] = np.random.uniform(-math.pi, math.pi, self.n)
        self.w[:] = 1.0 / self.n
        self._tracking = False; self.prev_est = None; self._lowess_count = 0   # 재탐색 시작

    def reset_own_half(self, attack_positive_x=True):
        """우리 진영 절반에만 균일 분포 (대회: 자기 진영에서 시작)."""
        if attack_positive_x:                          # 공격=+x → 우리 진영 x<0
            self.p[:, 0] = np.random.uniform(-self.fm.half_len, 0.0, self.n)
        else:
            self.p[:, 0] = np.random.uniform(0.0, self.fm.half_len, self.n)
        self.p[:, 1] = np.random.uniform(-self.fm.half_wid, self.fm.half_wid, self.n)
        self.p[:, 2] = np.random.uniform(-math.pi, math.pi, self.n)
        self.w[:] = 1.0 / self.n

    def predict(self, dx, dy, dth):
        scale = max(0.05, math.hypot(dx, dy) + abs(dth) * 0.05)
        noise = np.random.randn(self.n, 3) * self.motion_noise * scale
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        self.p[:, 0] += c * dx - s * dy + noise[:, 0]
        self.p[:, 1] += s * dx + c * dy + noise[:, 1]
        self.p[:, 2] += dth + noise[:, 2]
        self.p[:, 0] = np.clip(self.p[:, 0], self.fm.x_lo, self.fm.x_hi)
        self.p[:, 1] = np.clip(self.p[:, 1], -self.fm.half_wid, self.fm.half_wid)
        self._last_dx, self._last_dy = dx, dy

    def update(self, obs: List[Observation]):
        if not obs: return
        logw = np.log(self.w + 1e-300)
        c, s = np.cos(self.p[:, 2]), np.sin(self.p[:, 2])
        for o in obs:
            gx = self.p[:, 0] + c * o.rx - s * o.ry
            gy = self.p[:, 1] + s * o.rx + c * o.ry
            d = self.fm.nearest_batch(np.stack([gx, gy], 1), o.obs_type)
            sig = self.sigma.get(o.obs_type, 0.35)
            md  = self.match_dist.get(o.obs_type, 1.00)
            logw += np.where(d < md, -0.5 * (d / sig) ** 2, -30.0)
        # 연속성 안정화 (수렴 후에만; global 탐색 중엔 미적용)
        # 독립 PF 에서 이식: 대칭 뒤집힘·위치 점프 억제. _tracking=False 면 영향 없음.
        if self._tracking and self.prev_est is not None:
            px, py, pth = self.prev_est
            dth = (self.p[:, 2] - pth + np.pi) % (2 * np.pi) - np.pi
            logw += np.where(np.abs(dth) > math.radians(90),
                             -10.0 * (np.abs(dth) / math.pi), 0.0)     # heading 급변 패널티
            allowed = 2.0 + abs(self._last_dx) + abs(self._last_dy)    # 이동량만큼 점프 허용
            dpos = np.hypot(self.p[:, 0] - px, self.p[:, 1] - py)
            logw += np.where(dpos > allowed, -5.0 * (dpos - allowed), 0.0)  # 위치 점프 패널티
        logw -= logw.max()
        self.w = np.exp(logw); self.w /= self.w.sum() if self.w.sum() > 0 else 1.0

    def resample(self, random_ratio=0.03):
        ess = 1.0 / np.sum(self.w ** 2)
        if ess > self.n * 0.5:
            return
        pos = (np.arange(self.n) + np.random.uniform()) / self.n
        idx = np.clip(np.searchsorted(np.cumsum(self.w), pos), 0, self.n - 1)
        self.p = self.p[idx].copy()
        # 다양성 주입: 일부 파티클을 필드 전역 랜덤으로 교체 → 가짜 모드 조기 collapse 방지.
        # 수렴(_tracking) 후엔 비율을 step() 에서 크게 낮춰, 멀/적은 랜드마크일 때
        # 랜덤 파티클이 weight 잡아 spread 출렁(=GPS 점프 느낌)이는 것을 억제.
        k = int(self.n * random_ratio)
        if k > 0:
            ri = np.random.choice(self.n, k, replace=False)
            self.p[ri, 0] = np.random.uniform(self.fm.x_lo, self.fm.x_hi, k)
            self.p[ri, 1] = np.random.uniform(-self.fm.half_wid, self.fm.half_wid, k)
            self.p[ri, 2] = np.random.uniform(-math.pi, math.pi, k)
        # 납치/완전 발산 회복: ESS<1% 가 '지속'(연속 3프레임) 될 때만. 단발 노이즈로
        # 텔레포트하지 않도록. 회복도 80% 는 이전추정 근처(σ작게), 20%만 전역.
        if ess < self.n * 0.01:
            self._lowess_count += 1
        else:
            self._lowess_count = 0
        if self.prev_est is not None and self._lowess_count >= 3:
            self._lowess_count = 0
            nr = max(2, int(self.n * 0.1)); h = int(nr * 0.8)
            ri = np.random.choice(self.n, nr, replace=False)
            px, py, pth = self.prev_est
            self.p[ri[:h], 0] = np.clip(px + np.random.randn(h) * 0.6, self.fm.x_lo, self.fm.x_hi)
            self.p[ri[:h], 1] = np.clip(py + np.random.randn(h) * 0.6, -self.fm.half_wid, self.fm.half_wid)
            self.p[ri[:h], 2] = pth + np.random.randn(h) * math.radians(20)
            self.p[ri[h:], 0] = np.random.uniform(self.fm.x_lo, self.fm.x_hi, nr - h)
            self.p[ri[h:], 1] = np.random.uniform(-self.fm.half_wid, self.fm.half_wid, nr - h)
            self.p[ri[h:], 2] = np.random.uniform(-math.pi, math.pi, nr - h)
        self.w[:] = 1.0 / self.n

    def estimate(self):
        # dominant-cluster 가중평균: posterior 가 multimodal/넓을 때 '모드 사이 중앙'으로
        # 빠지는 것(center bias)을 막는다. MAP 파티클 주변 1.5m 클러스터만 평균.
        i_map = int(np.argmax(self.w))
        cx, cy = self.p[i_map, 0], self.p[i_map, 1]
        d = np.hypot(self.p[:, 0] - cx, self.p[:, 1] - cy)
        wm = self.w * (d < 1.5)
        s = wm.sum()
        if s < 1e-12:               # 클러스터 비면 전역 평균으로 폴백
            wm = self.w; s = wm.sum() or 1.0
        x = float(np.sum(self.p[:, 0] * wm) / s)
        y = float(np.sum(self.p[:, 1] * wm) / s)
        th = math.atan2(float(np.sum(np.sin(self.p[:, 2]) * wm) / s),
                        float(np.sum(np.cos(self.p[:, 2]) * wm) / s))
        return x, y, float(th)

    def global_mean(self):
        """전역 가중평균 (spread 계산 등 내부용)."""
        x = float(np.average(self.p[:, 0], weights=self.w))
        y = float(np.average(self.p[:, 1], weights=self.w))
        th = math.atan2(np.average(np.sin(self.p[:, 2]), weights=self.w),
                        np.average(np.cos(self.p[:, 2]), weights=self.w))
        return x, y, float(th)

    def spread(self):
        mx = np.average(self.p[:, 0], weights=self.w)
        my = np.average(self.p[:, 1], weights=self.w)
        sx = math.sqrt(max(0.0, np.average((self.p[:, 0] - mx) ** 2, weights=self.w)))
        sy = math.sqrt(max(0.0, np.average((self.p[:, 1] - my) ** 2, weights=self.w)))
        R  = min(1.0, math.hypot(np.average(np.sin(self.p[:, 2]), weights=self.w),
                                  np.average(np.cos(self.p[:, 2]), weights=self.w)))
        return sx, sy, math.sqrt(max(0.0, -2.0 * math.log(R + 1e-9)))

    def step(self, dx, dy, dth, obs):
        self.predict(dx, dy, dth); self.update(obs)
        # 수렴(_tracking) 후엔 전역 랜덤 주입을 0.5%로 줄여 안정화. 탐색 중엔 3% 유지.
        self.resample(random_ratio=0.005 if self._tracking else 0.03)
        x, y, th = self.estimate()
        sxy = math.hypot(*self.spread()[:2])
        if sxy < 0.6:                 # 수렴 → 연속성 안정화 on
            self._tracking = True
        elif sxy > 1.5:               # 다시 발산(납치 등) → off, 재탐색 허용
            self._tracking = False
        self.prev_est = (x, y, th)
        return x, y, float(th)


class LocalizationEngine:
    def __init__(self, cfg, half_len=7.0, half_wid=4.5):
        self.enabled = bool(getattr(cfg, "localization_enabled", False))
        self.attack_positive_x = bool(getattr(cfg, "attack_positive_x", True))
        self.transform = CoordinateTransform()
        # 하프코트: 물리적으로 존재하는 절반만 맵에 → 대칭 모호성 제거 (노트북+카메라 벤치).
        field_mode = str(getattr(cfg, "loc_field_mode", "x_pos"))
        self.fm = FieldMap(half_len, half_wid, field_mode=field_mode)
        # 기본 uniform = 그 절반 전체에 균일 분포 → 절반 안 어디서 시작하든 수렴.
        init_mode = str(getattr(cfg, "loc_init_mode", "uniform"))
        self.pf = ParticleFilter(self.fm, init_mode=init_mode)
        if init_mode == "own_half":
            self.pf.reset_own_half(self.attack_positive_x)
        self.pose = (0.0, 0.0, 0.0); self.n_landmarks = 0
        # 출력 안정화 상태
        self._smoothed = None              # EMA 적용된 출력 pose
        self._hist = deque(maxlen=5)       # 최근 추정 (시간 안정성 판단)
        self._trust_state = False          # trust 히스테리시스(Schmitt) 상태
        self._raw_pose = (0.0, 0.0, 0.0)   # 평활화 전 PF 원추정 (디버그/검증용)

    # 출력 평활화 파라미터 (핸드헬드 카메라: 텔레포트 억제하되 실제 이동은 몇 프레임 내 추종)
    _OUT_EMA = 0.5          # EMA 계수 (클수록 빠르고 덜 부드러움)
    _OUT_MAX_JUMP = 0.8     # 프레임당 허용 위치 변화(m). 초과분은 잘라 점진 추종.

    def _reset_output_state(self, pose):
        self._smoothed = None if pose is None else tuple(pose)
        self._hist.clear()
        self._trust_state = False

    def reset_uniform(self):
        """global 재추정: 필드 전체 분포 (드롭 위치 모름)."""
        self.pf.reset_uniform(); self.pose = (0.0, 0.0, 0.0); self.n_landmarks = 0
        self._reset_output_state(None)

    def reset_own_half(self):
        self.pf.reset_own_half(self.attack_positive_x); self.pose = (0.0, 0.0, 0.0); self.n_landmarks = 0
        self._reset_output_state(None)

    def seed_pose(self, x, y, th):
        """알고 있는 시작 위치로 시딩."""
        self.pf.seed(x, y, th); self.pose = (float(x), float(y), float(th)); self.n_landmarks = 0
        self._reset_output_state((float(x), float(y), float(th)))

    def update(self, obs: List[Observation], odom=(0.0, 0.0, 0.0)):
        if not self.enabled:
            return self.pose
        raw = self.pf.step(*odom, obs)
        self._raw_pose = raw
        self.n_landmarks = len(obs)
        self.pose = self._smooth_output(raw)   # 점프 클램프 + EMA
        self._update_trust()                    # 히스테리시스 + 시간 안정성
        return self.pose

    def _smooth_output(self, raw):
        """PF 원추정에 점프 클램프 + EMA → view 상 뚝뚝 끊김/텔레포트 억제."""
        if self._smoothed is None:
            self._smoothed = raw
            return raw
        sx, sy, sth = self._smoothed
        rx, ry, rth = raw
        dx, dy = rx - sx, ry - sy
        jump = math.hypot(dx, dy)
        if jump > self._OUT_MAX_JUMP:          # 한 프레임에 너무 멀리 → 잘라서 점진 이동
            sc = self._OUT_MAX_JUMP / jump
            rx, ry = sx + dx * sc, sy + dy * sc
        a = self._OUT_EMA
        nx = sx + a * (rx - sx)
        ny = sy + a * (ry - sy)
        dth = (rth - sth + math.pi) % (2 * math.pi) - math.pi
        nth = sth + a * dth
        self._smoothed = (nx, ny, float(nth))
        return self._smoothed

    def _update_trust(self):
        """trust = 랜드마크 충분 + spread 작음 + 시간적으로 안정. Schmitt 히스테리시스로 깜빡임 제거."""
        sx, sy, sth = self.pf.spread()
        sxy = math.hypot(sx, sy)
        self._hist.append((self.pose[0], self.pose[1]))
        stable = True
        if len(self._hist) >= self._hist.maxlen:
            xs = [p[0] for p in self._hist]; ys = [p[1] for p in self._hist]
            stable = (max(xs) - min(xs) < 0.5) and (max(ys) - min(ys) < 0.5)
        enough = self.n_landmarks >= 3         # 3개 미만이면 신뢰 안 함(과확신 방지)
        if not self._trust_state:
            # 진입: 엄격 (랜드마크 3+, spread<0.5, 안정 지속)
            self._trust_state = enough and sxy < 0.5 and sth < 0.30 and stable
        else:
            # 유지: 완화 (spread<0.9 까지 허용) → 경계에서 True↔False 깜빡임 방지
            self._trust_state = (self.n_landmarks >= 2) and sxy < 0.9 and sth < 0.5

    def trust(self, max_xy=0.6, max_theta=0.35, min_landmarks=2) -> bool:
        # 히스테리시스/시간안정성 반영된 상태 반환 (_update_trust 에서 갱신).
        if not self.enabled:
            return False
        return self._trust_state

    def goal_direction(self) -> Optional[float]:
        if not self.trust():
            return None
        x, y, th = self.pose
        gx, gy = self.fm.attack_goal_xy(self.attack_positive_x)
        return (math.atan2(gy - y, gx - x) - th + math.pi) % (2 * math.pi) - math.pi

    def reset_pose(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                   n: Optional[int] = None) -> None:
        """PF를 지정 pose로 수동 초기화한다.

        테스트용 — 로봇을 알고 있는 위치에 세운 뒤 호출해 PF 시딩.
        나중에 실대회에서는 킥오프 위치 + IMU heading 으로 교체 예정.

        Args:
            x, y, theta : 필드 좌표계 초기 위치 (m, m, rad). 기본값 (0,0,0) = 필드 중심/정면.
            n           : 파티클 수 유지. None 이면 현재 pf.n 그대로.
        """
        n_particles = n if n is not None else self.pf.n
        self.pf = ParticleFilter(self.fm, n=n_particles, init_pose=(x, y, theta))
        self.pose = (float(x), float(y), float(theta))
        self.n_landmarks = 0
        self._reset_output_state((float(x), float(y), float(theta)))
        print(f"[LOC] reset_pose → ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
