#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""상체 hold, 목 들어 올리기와 팔꿈치 사인파 동작을 시험한다.
각 관절의 방향과 가동 범위를 확인한 실물 시험용 코드다."""

import rospy
import time
import math
import signal
import threading
from tron_upper_body.msg import MotorCommand, MotorFeedback


# 모터 그룹
ARM_IDS    = [1, 2, 3, 4]
HEAD_IDS   = [5, 6]
ELBOW_IDS  = [7, 8]
HOLD_IDS   = ARM_IDS + HEAD_IDS
ALL_IDS    = ARM_IDS + HEAD_IDS + ELBOW_IDS

ID_NAME = {
    1: "sho_L",   2: "sho_R",
    3: "arm_L",   4: "arm_R",
    5: "neck",    6: "head",
    7: "elbow_L", 8: "elbow_R",
}

# 시작점 기준 위치 한계 (±rad)
POS_LIMIT = {
    1: 0.7, 2: 0.7,
    3: 1.0, 4: 1.0,
    5: 0.6,
    6: 1.8,    # lift 1.5 + 여유
    7: 0.5, 8: 0.5,
}
RUNAWAY_ERR = {mid: lim * 1.6 for mid, lim in POS_LIMIT.items()}

# hold 게인 (Kp, Kd)
HOLD_GAINS = {
    1: (15.0, 1.0), 2: (15.0, 1.0),
    3: (15.0, 1.0), 4: (15.0, 1.0),
    5: (15.0, 1.0), 6: (20.0, 1.2),
    7: (12.0, 1.5), 8: (12.0, 1.5),
}

# 목 들어 올리기 파라미터
HEAD_LIFT      = 1.5      # 엎드린 자세에서 정면까지
LIFT_TIME      = 2.0      # 부드럽게 들어올리는 시간
ARRIVAL_TOL    = 0.1      # lift 도착 판정 (rad, 약 5°)
ARRIVAL_WAIT   = 5.0      # 도착 timeout (sec)

RAMP_TIME      = 2.0
WAVE_AMP       = 0.2
WAVE_FREQ      = 0.5
WAVE_DUR       = 10.0
FB_TIMEOUT     = 1.0


class Phase3Integrated:
    def __init__(self):
        rospy.init_node("phase3_integrated", anonymous=True)

        self.pos = {}
        self.start_pos = {}
        self.last_fb = {}
        self.stop = False
        self.lock = threading.Lock()

        self.pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=10)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self._cb)
        signal.signal(signal.SIGINT, self._sigint)

    def _cb(self, msg):
        with self.lock:
            self.pos[msg.id] = msg.position
            self.last_fb[msg.id] = time.time()

    def _sigint(self, sig, frame):
        print("\n[Stop] Ctrl+C — ramp-down 후 종료")
        self.stop = True

    def _send(self, mid, p_des, kp, kd=1.0):
        # 시작점 기준 clamp
        if mid in self.start_pos:
            rel = p_des - self.start_pos[mid]
            lim = POS_LIMIT.get(mid, 0.5)
            if abs(rel) > lim:
                rel = math.copysign(lim, rel)
                p_des = self.start_pos[mid] + rel

        cmd = MotorCommand()
        cmd.id = mid
        cmd.p_des = float(p_des)
        cmd.v_des = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
        cmd.t_ff = 0.0
        self.pub.publish(cmd)

    def _health_ok(self):
        now = time.time()
        with self.lock:
            for mid in ALL_IDS:
                if mid not in self.last_fb:
                    print(f"[Fail] ID {mid} 피드백 없음")
                    return False
                if now - self.last_fb[mid] > FB_TIMEOUT:
                    print(f"[Fail] ID {mid} 피드백 끊김")
                    return False
                if mid in self.start_pos:
                    err = abs(self.pos[mid] - self.start_pos[mid])
                    if err > RUNAWAY_ERR[mid]:
                        print(f"[Fail] ID {mid} runaway: err={err:.3f}")
                        return False
        return True

    # 단계별 메서드
    def wait_connect(self, timeout=10.0):
        print("[Step 1] 모터 연결 대기")
        t0 = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            for mid in ALL_IDS:
                self._send(mid, 0.0, kp=0.0, kd=0.1)
            with self.lock:
                got = [m for m in ALL_IDS if m in self.pos]
            if len(got) == len(ALL_IDS):
                print(f"         응답: {got}")
                return True
            if time.time() - t0 > timeout:
                missing = [m for m in ALL_IDS if m not in got]
                print(f"         timeout. 응답 없는 ID: {missing}")
                return False
            rate.sleep()
        return False

    def initial_hold(self):
        """영점 설정 전에 현재 자세를 유지해 흔들림을 줄인다."""
        print("[Step 2] 초기 hold (1초) — 흔들림 진정")
        rate = rospy.Rate(50)
        # 현재 위치를 임시로 잡고 hold
        with self.lock:
            temp_pos = dict(self.pos)
        for _ in range(50):
            for mid in ALL_IDS:
                kp, kd = HOLD_GAINS[mid]
                self._send(mid, temp_pos[mid], kp=kp, kd=kd)
            rate.sleep()

    def calibrate_initial(self):
        print("[Step 3] 1차 캘리브 (현재 위치 = 0점)")
        time.sleep(0.3)
        with self.lock:
            for mid in ALL_IDS:
                self.start_pos[mid] = self.pos[mid]
                print(f"         ID {mid} {ID_NAME[mid]:8s} = {self.start_pos[mid]:+.4f}")

    def lift_head(self):
        """ID 6을 HEAD_LIFT만큼 움직이고 피드백으로 도착을 확인한다."""
        print(f"[Step 4] 목 lift +{HEAD_LIFT:.2f} rad ({LIFT_TIME}s)")
        steps = int(LIFT_TIME * 50)
        rate = rospy.Rate(50)
        base_pitch = self.start_pos[6]
        target_abs = base_pitch + HEAD_LIFT

        # (1) 궤적 생성 — 부드럽게 보간
        for i in range(steps + 1):
            if rospy.is_shutdown() or self.stop:
                return False
            if not self._health_ok():
                self.stop = True
                return False

            ratio = i / steps
            target_pitch = base_pitch + HEAD_LIFT * ratio

            # head pitch: lift
            self._send(6, target_pitch, kp=HOLD_GAINS[6][0], kd=HOLD_GAINS[6][1])
            # 나머지: ramp-up 하면서 시작점 hold
            for mid in ALL_IDS:
                if mid == 6:
                    continue
                kp, kd = HOLD_GAINS[mid]
                self._send(mid, self.start_pos[mid], kp=kp * ratio, kd=kd)
            rate.sleep()

        # (2) 도착 확인 — 실제로 target 에 도달했는지
        print(f"         궤적 송신 완료. 도착 확인 중 (tol={ARRIVAL_TOL} rad)")
        wait_start = time.time()
        while not rospy.is_shutdown() and not self.stop:
            with self.lock:
                cur_pitch = self.pos[6]
            err = abs(cur_pitch - target_abs)

            # 계속 명령 보내면서 도착 대기
            self._send(6, target_abs, kp=HOLD_GAINS[6][0], kd=HOLD_GAINS[6][1])
            for mid in ALL_IDS:
                if mid == 6:
                    continue
                kp, kd = HOLD_GAINS[mid]
                self._send(mid, self.start_pos[mid], kp=kp, kd=kd)

            if err < ARRIVAL_TOL:
                print(f"         도착 완료 (err={err:.4f})")
                break

            if time.time() - wait_start > ARRIVAL_WAIT:
                print(f"         timeout — 강제 진행 (err={err:.4f})")
                break

            rate.sleep()

        time.sleep(0.5)
        return True

    def manual_yaw_align(self):
        """머리 yaw 가 정중앙이 아닐 수 있으니 사용자가 손으로 정렬."""
        print()
        print("머리 yaw 가 정면이 아니면 손으로 살짝 돌려서 정렬해주세요.")
        print("(다른 관절은 건드리지 마세요)")
        print("정렬 후 Enter / Ctrl+C 로 중단")
        try:
            input("> ")
        except (EOFError, KeyboardInterrupt):
            self.stop = True
            return False
        return True

    def calibrate_after_lift(self):
        """정면 자세에서 목 0점 재설정."""
        print("[Step 6] 2차 캘리브 (정면 자세 = 목 0점)")
        time.sleep(0.3)
        with self.lock:
            for mid in HEAD_IDS:
                self.start_pos[mid] = self.pos[mid]
                print(f"         ID {mid} {ID_NAME[mid]:8s} = {self.start_pos[mid]:+.4f}")

    def motion_wave(self):
        print(f"[Step 7] 팔꿈치 사인파 ±{WAVE_AMP} rad ({WAVE_DUR}s)")
        t0 = time.time()
        rate = rospy.Rate(50)
        last_print = 0.0

        while not rospy.is_shutdown() and not self.stop:
            t = time.time() - t0
            if t > WAVE_DUR:
                break
            if not self._health_ok():
                self.stop = True
                break

            wave = WAVE_AMP * math.sin(2 * math.pi * WAVE_FREQ * t)

            # 팔꿈치 좌우 반대 방향 (박수 모양)
            self._send(7, self.start_pos[7] + wave,
                       kp=HOLD_GAINS[7][0], kd=HOLD_GAINS[7][1])
            self._send(8, self.start_pos[8] - wave,
                       kp=HOLD_GAINS[8][0], kd=HOLD_GAINS[8][1])
            # 나머지 hold
            for mid in HOLD_IDS:
                kp, kd = HOLD_GAINS[mid]
                self._send(mid, self.start_pos[mid], kp=kp, kd=kd)

            if t - last_print > 0.5:
                with self.lock:
                    p7 = self.pos.get(7, 0.0)
                    p8 = self.pos.get(8, 0.0)
                print(f"  t={t:4.1f}  "
                      f"ID7 tgt={self.start_pos[7] + wave:+.3f} cur={p7:+.3f}  "
                      f"ID8 tgt={self.start_pos[8] - wave:+.3f} cur={p8:+.3f}")
                last_print = t
            rate.sleep()

    def ramp_down(self):
        print("[Step 8] ramp-down")
        steps = 50
        rate = rospy.Rate(50)
        for i in range(steps + 1):
            ratio = 1.0 - (i / steps)
            for mid in ALL_IDS:
                if mid not in self.start_pos:
                    continue
                kp, kd = HOLD_GAINS[mid]
                self._send(mid, self.start_pos[mid], kp=kp * ratio, kd=kd)
            rate.sleep()

        # 위치 제어 해제. Kd 감쇠는 유지한다.
        for _ in range(30):
            for mid in ALL_IDS:
                self._send(mid, 0.0, kp=0.0, kd=0.1)
            time.sleep(0.02)
        print("         모터 해제 완료")

    # 메인
    def run(self):
        print("=" * 60)
        print(" Phase 3: 상체 통합 hold + 팔꿈치 사인파")
        print("=" * 60)
        print(" Phase 1, 2 통과 후에만 실행하세요.")
        print(" 시작 시 로봇이 엎드린 상태여야 합니다.")
        print()

        try:
            if not self.wait_connect():
                return
            self.initial_hold()
            self.calibrate_initial()

            print()
            print("다음 단계: 목 lift + 전체 ramp-up")
            print("사람과 거리 두고 Enter (Ctrl+C 로 중단)")
            try:
                input("> ")
            except (EOFError, KeyboardInterrupt):
                self.stop = True
                return

            if not self.lift_head():
                return
            if not self.manual_yaw_align():
                return
            self.calibrate_after_lift()
            self.motion_wave()

        except Exception as e:
            print(f"[Error] {e}")
            self.stop = True
        finally:
            self.ramp_down()


if __name__ == "__main__":
    Phase3Integrated().run()
