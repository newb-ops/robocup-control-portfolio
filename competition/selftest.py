#!/usr/bin/env python3
"""가상 영상과 탐지 결과로 관측값과 동작 상태 전이를 검사한다.
DryRunDriver를 사용하며 실제 카메라나 로봇을 연결하지 않는다."""
import math
import sys
import time

import numpy as np

from tronb_vision.app import BallFollower
from tronb_vision.common import OperationMode, RobotConfig, RobotState, TuningParams
from tronb_vision.detector import Detection, DummyDetector
from tronb_vision.drivers import DryRunDriver
from tronb_vision.sources import DummySource, FramePacket


class NoDepthSource(DummySource):
    """depth=None → 거리 추정이 area 기반(=가까워질수록 작아지는 dist)으로 동작."""
    def read(self):
        pkt = super().read()
        return FramePacket(pkt.rgb, None, pkt.capture_time, pkt.frame_id)


def square(cls_id, cls_name, conf, side, cx=320, cy=240):
    s = side / 2.0
    return Detection(cls_id, cls_name, conf, (cx - s, cy - s, cx + s, cy + s))


def build_script():
    """공: side 30→280 램프 후 유지(가까워짐). 골: 중앙 상시(정렬 가능)."""
    frames = []
    side = 30.0
    for _ in range(700):
        ball = square(0, "ball", 0.9, side)
        goal = square(3, "goal", 0.8, 120, cx=320, cy=180)   # 중앙 상단
        frames.append([ball, goal])
        if side < 280.0:
            side = min(280.0, side + 6.0)
    return frames


def main():
    cfg = RobotConfig(detector_backend="dummy", localization_enabled=True)  # loc 경로도 켜서 검증
    tuning = TuningParams()
    det = DummyDetector(cfg.class_names, scripted=build_script())
    src = NoDepthSource(cfg.img_width, cfg.img_height)
    drv = DryRunDriver(verbose=False)
    bf = BallFollower(cfg, tuning, src, drv, detector=det, bgr_input=False)
    bf.set_mode(OperationMode.AUTO)

    # 모든 기능을 켠 상태의 전이와 관측값도 확인한다.
    from tronb_vision.common import TriState
    for k in list(bf.features.keys()):
        bf.features[k] = TriState.ON
    bf.loc.enabled = True

    src.start(); drv.start()
    seen = []; obs_ok = True; last_state = None; err = None
    t0 = time.time()
    for i in range(700):
        try:
            obs = bf.tick()
        except Exception as e:
            import traceback; traceback.print_exc(); err = e; break
        if bf.state != last_state:
            seen.append(bf.state.name); last_state = bf.state
        if obs is not None and not (math.isfinite(obs.dist) and math.isfinite(obs.screen_x)):
            obs_ok = False
        time.sleep(0.005)
        if bf.state == RobotState.COOLDOWN:
            break
    drv.send_twist(0, 0); src.stop(); drv.stop()
    elapsed = time.time() - t0

    feat_keys = sorted((bf.feat_out or {}).keys())
    print(f"[states] {' -> '.join(seen)}")
    print(f"[final ] state={bf.state.name} twist={drv.last_twist}")
    print(f"[iters ] {i+1} in {elapsed:.2f}s")
    print(f"[feat_out keys (모든 기능 ON)] {feat_keys}")
    print(f"[no exception] {err is None}")

    must_reach = {"SEARCH", "TRACK", "APPROACH", "AIM"}
    reached = set(seen)
    expect_feats = {"ball_stop"}
    ok = (err is None and obs_ok and must_reach.issubset(reached)
          and expect_feats.issubset(set(feat_keys)))
    print(f"[contract finite] {obs_ok}")
    print(f"[reached {sorted(must_reach)}] {must_reach.issubset(reached)}")
    print(f"[reached KICK]    {'KICK' in reached}")
    print(f"[features active] {expect_feats.issubset(set(feat_keys))}")
    print("SELFTEST V10.3 OK" if ok else "SELFTEST V10.3 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
