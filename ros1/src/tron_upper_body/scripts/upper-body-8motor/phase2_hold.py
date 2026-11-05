#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""선택한 모터에 낮은 위치 게인을 인가해 hold 동작을 확인한다.
현재 ID와 영점을 확인하고 관절을 지지한 상태에서 실행한다."""

import rospy
import time
import signal
import threading
from tron_upper_body.msg import MotorCommand, MotorFeedback


# 테스트 순서 — 가벼운 것부터
TEST_ORDER = [5, 7, 8, 2, 3]
ALL_IDS    = [2, 3, 5, 7, 8]

ID_NAME = {
    1: "sho_L",   2: "sho_R",
    3: "arm_L",   4: "arm_R",
    5: "neck",    6: "head",
    7: "elbow_L", 8: "elbow_R",
}

# 모터별 hold 시험 Kp
TEST_KP = {
    1: 3.0, 2: 3.0,
    3: 3.0, 4: 3.0,
    5: 3.0, 6: 4.0,    # 목 pitch는 중력 부하를 고려해 Kp를 높임
    7: 3.0, 8: 3.0,
}
TEST_KD = 0.5

# 폭주 감지 — 시작점에서 이 값 넘게 벗어나면 즉시 중단
RUNAWAY = 0.3  # rad

RAMP_TIME = 2.0
HOLD_TIME = 3.0


class Phase2Hold:
    def __init__(self):
        rospy.init_node("phase2_hold", anonymous=True)
        self.pos = {}
        self.stop = False
        self.lock = threading.Lock()

        self.pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=10)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self._cb)
        signal.signal(signal.SIGINT, self._sigint)

    def _cb(self, msg):
        with self.lock:
            self.pos[msg.id] = msg.position

    def _sigint(self, sig, frame):
        print("\n[Stop] Ctrl+C — 모터 전체 해제")
        self.stop = True

    def _send(self, mid, p_des, kp, kd=TEST_KD):
        cmd = MotorCommand()
        cmd.id = mid
        cmd.p_des = float(p_des)
        cmd.v_des = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
        cmd.t_ff = 0.0
        self.pub.publish(cmd)

    def _release_all(self):
        """모든 모터의 위치 제어를 끄고 Kd=0.1의 감쇠만 남긴다."""
        for _ in range(30):
            for mid in ALL_IDS:
                self._send(mid, 0.0, kp=0.0, kd=0.1)
            time.sleep(0.02)

    def _wait_all_connected(self, timeout=10.0):
        print("[Init] 모터 연결 대기")
        rate = rospy.Rate(50)
        t0 = time.time()
        while not rospy.is_shutdown():
            for mid in ALL_IDS:
                self._send(mid, 0.0, kp=0.0, kd=0.1)
            with self.lock:
                got = sorted(self.pos.keys())
            if all(m in got for m in ALL_IDS):
                print(f"       응답: {got}")
                return True
            if time.time() - t0 > timeout:
                missing = [m for m in ALL_IDS if m not in got]
                print(f"       timeout. 응답 없는 ID: {missing}")
                return False
            rate.sleep()
        return False

    def _test_single(self, mid):
        """ID 하나만 hold 테스트."""
        with self.lock:
            start = self.pos[mid]
        kp_max = TEST_KP[mid]

        print()
        print(f"─── ID {mid} ({ID_NAME[mid]}) ───")
        print(f"    start = {start:+.4f} rad, target Kp = {kp_max:.1f}")
        print(f"    Enter 로 시작 / Ctrl+C 로 중단")
        try:
            input("    > ")
        except (EOFError, KeyboardInterrupt):
            self.stop = True
            return False

        rate = rospy.Rate(50)

        # 1) Ramp-up
        print(f"    [1/3] Kp ramp-up 0 → {kp_max:.1f} ({RAMP_TIME}s)")
        steps = int(RAMP_TIME * 50)
        for i in range(steps + 1):
            if self.stop or rospy.is_shutdown():
                return False

            ratio = i / steps
            kp_now = kp_max * ratio

            with self.lock:
                err = abs(self.pos.get(mid, start) - start)
            if err > RUNAWAY:
                print(f"    [Fail] 폭주 감지: err={err:.3f} > {RUNAWAY}")
                self.stop = True
                return False

            # 대상 ID 만 hold, 나머지는 Kp=0 (Kd 댐핑)
            for other in ALL_IDS:
                if other == mid:
                    self._send(mid, start, kp=kp_now)
                else:
                    self._send(other, 0.0, kp=0.0, kd=0.1)
            rate.sleep()

        # 2) Hold
        print(f"    [2/3] hold {HOLD_TIME}s — 손으로 살짝 밀어보세요")
        t0 = time.time()
        last_print = 0
        max_err = 0.0
        while time.time() - t0 < HOLD_TIME:
            if self.stop or rospy.is_shutdown():
                return False

            with self.lock:
                cur = self.pos.get(mid, start)
            err = cur - start
            max_err = max(max_err, abs(err))

            if abs(err) > RUNAWAY:
                print(f"    [Fail] hold 중 폭주: err={err:+.3f}")
                self.stop = True
                return False

            for other in ALL_IDS:
                if other == mid:
                    self._send(mid, start, kp=kp_max)
                else:
                    self._send(other, 0.0, kp=0.0, kd=0.1)

            now = time.time()
            if now - last_print > 0.5:
                print(f"          t={now - t0:.1f}s  cur={cur:+.4f}  err={err:+.4f}")
                last_print = now
            rate.sleep()

        # 3) Ramp-down
        print(f"    [3/3] Kp ramp-down")
        for i in range(steps + 1):
            ratio = 1.0 - (i / steps)
            kp_now = kp_max * ratio
            for other in ALL_IDS:
                if other == mid:
                    self._send(mid, start, kp=kp_now)
                else:
                    self._send(other, 0.0, kp=0.0, kd=0.1)
            rate.sleep()

        print(f"    PASS — 최대 에러 {max_err:.4f} rad")
        return True

    def run(self):
        print("=" * 60)
        print(" Phase 2: 모터별 단독 hold 테스트")
        print("=" * 60)
        print(" 각 ID 를 순서대로 켜고 Kp=3 으로 약하게 잡음.")
        print(" 손으로 살짝 밀었을 때 부드럽게 저항하면 정상.")
        print(" 휙 돌거나 떨면 즉시 Ctrl+C.")
        print()
        print(f" 테스트 순서: {TEST_ORDER}")
        print()

        try:
            if not self._wait_all_connected():
                return

            results = {}
            for mid in TEST_ORDER:
                if self.stop:
                    break
                ok = self._test_single(mid)
                results[mid] = ok
                if not ok:
                    break

            print()
            print("=" * 60)
            print(" 결과 요약")
            print("=" * 60)
            for mid in TEST_ORDER:
                if mid not in results:
                    status = "skip"
                elif results[mid]:
                    status = "PASS"
                else:
                    status = "FAIL"
                print(f"  ID {mid} {ID_NAME[mid]:8s} : {status}")

        finally:
            self._release_all()
            print("\n[End] 모터 전체 해제 완료")


if __name__ == "__main__":
    Phase2Hold().run()
