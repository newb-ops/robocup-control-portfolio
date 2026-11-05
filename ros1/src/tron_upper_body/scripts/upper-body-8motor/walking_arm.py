#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""제자리에서 양팔의 반대 위상 swing과 팔꿈치 추종 동작을 시험한다.
TRON1 하체의 보행 정책을 실행하는 코드는 아니다."""

import rospy
import time
import math
import signal
import threading
from collections import deque
from tron_upper_body.msg import MotorCommand, MotorFeedback

# 이 시험 구성의 모터 ID와 장착 방향
ID_R_SHO_R = 1   # 오 어깨 roll,  밖 +
ID_L_SHO_R = 2   # 왼 어깨 roll,  밖
ID_L_SHO_P = 3   # 왼 어깨 pitch, 앞 +
ID_R_SHO_P = 4   # 오 어깨 pitch, 앞
ID_NECK_P  = 5   # 목 pitch,      위 +
ID_NECK_Y  = 6   # 목 yaw,        오 +
ID_L_ELB   = 7   # 왼 팔꿈치,     앞 +
ID_R_ELB   = 8   # 오 팔꿈치,     앞

ALL_IDS = [ID_R_SHO_R, ID_L_SHO_R, ID_L_SHO_P, ID_R_SHO_P,
           ID_NECK_P, ID_NECK_Y, ID_L_ELB, ID_R_ELB]

ID_NAME = {
    1: "sho_R_roll", 2: "sho_L_roll", 3: "sho_L_pitch", 4: "sho_R_pitch",
    5: "neck_pitch", 6: "neck_yaw", 7: "elbow_L", 8: "elbow_R",
}

# 걷기 swing
LOOP_HZ        = 50
SWING_FREQ     = 0.7    # Hz
SWING_AMP_BASE = 0.08   # rad, 멈춰도 남는 최소 흔들림
SWING_AMP_GAIN = 0.20   # rad, gait=1일 때 추가 진폭. 늘리기 전에 관절 간섭을 확인한다.
ARM_BASE_OUT   = 0.10   # rad, 어깨 바깥쪽 roll. 간섭이 있으면 줄인다.

# 어깨 움직임을 지연해 따라가는 팔꿈치
ELBOW_BASE_BEND = 0.30  # rad, 기본 굽힘 (약 17도)
ELBOW_FOLLOW    = 0.40  # 어깨 swing 추종 비율
ELBOW_DELAY     = 0.12  # s, 어깨 대비 지연 시간

# 관절 게인. Kp가 너무 낮으면 중력 부하로 처질 수 있다.
SHOULDER_KP = 12.0
ELBOW_KP    = 8.0
NECK_KP     = 12.0
KD          = 0.8

# 데모에서는 gait를 0~1로 바꿔 팔 진폭을 조절한다. 실기 통합 시 속도 입력이 필요하다.
DEMO_GAIT  = True
GAIT_CYCLE = 12.0   # s, 한 번 느려졌다 빨라지는 주기

# 안전
RATE_LIMIT      = 0.04   # rad/step, 궤적 속도 제한
MAX_FOLLOW_ERR  = 0.40   # rad, 지령과 실제 위치의 차이가 이 값을 넘으면 궤적 진행 중단
EMERGENCY_POS   = math.radians(110)  # start 기준 이만큼 벗어나면 E-STOP
EMERGENCY_VEL   = 4.0    # rad/s
FB_TIMEOUT      = 1.0
INIT_HOLD_TIME  = 1.5
RAMP_TIME       = 2.0


class WalkingArmsDemo:
    def __init__(self):
        rospy.init_node("walking_arms_demo", anonymous=True)
        self.pos = {}
        self.vel = {}
        self.start_pos = {}
        self.last_fb = {}
        self.prev_cmd = {}
        self.stop = False
        self.lock = threading.RLock()

        self.pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=100)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self._cb)
        signal.signal(signal.SIGINT, self._sigint)

    def _cb(self, msg):
        with self.lock:
            if abs(msg.position) > 50.0:   # 깨진 프레임 필터
                return
            self.pos[msg.id] = msg.position
            self.vel[msg.id] = msg.velocity
            self.last_fb[msg.id] = time.time()

    def _sigint(self, sig, frame):
        print("\n[Stop] Ctrl+C — ramp-down 후 종료")
        self.stop = True

    def _send_raw(self, mid, p_des, kp, kd=KD):
        cmd = MotorCommand()
        cmd.id = mid
        cmd.p_des = float(p_des)
        cmd.v_des = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
        cmd.t_ff = 0.0
        self.pub.publish(cmd)

    def _send(self, mid, target_abs, kp, kd=KD):
        with self.lock:
            if mid not in self.prev_cmd:
                self.prev_cmd[mid] = self.pos.get(mid, self.start_pos.get(mid, 0.0))
            prev = self.prev_cmd[mid]
            actual = self.pos.get(mid, prev)

        # 1) 궤적 속도 제한
        diff = target_abs - prev
        safe = prev + math.copysign(RATE_LIMIT, diff) if abs(diff) > RATE_LIMIT else target_abs

        # 추종 오차가 크면 목표를 더 진행시키지 않는다.
        if abs(safe - actual) > MAX_FOLLOW_ERR:
            safe = prev

        with self.lock:
            self.prev_cmd[mid] = safe
        self._send_raw(mid, safe, kp, kd)

    def _health_ok(self):
        now = time.time()
        with self.lock:
            for mid in ALL_IDS:
                if mid not in self.last_fb:
                    return False
                if now - self.last_fb[mid] > FB_TIMEOUT:
                    print(f"[E-STOP] ID {mid} 피드백 끊김")
                    return False
                if mid in self.start_pos:
                    err = abs(self.pos[mid] - self.start_pos[mid])
                    if err > EMERGENCY_POS:
                        print(f"[E-STOP] ID {mid} 위치 초과: {err:.2f} > {EMERGENCY_POS:.2f} rad")
                        return False
                if abs(self.vel.get(mid, 0.0)) > EMERGENCY_VEL:
                    print(f"[E-STOP] ID {mid} 속도 초과: {self.vel[mid]:.2f} rad/s")
                    return False
        return True

    def wait_connect(self, timeout=10.0):
        print(f"[1] 모터 연결 대기 ({len(ALL_IDS)}개: {ALL_IDS})")
        t0 = time.time()
        rate = rospy.Rate(LOOP_HZ)
        while not rospy.is_shutdown():
            with self.lock:
                got = [m for m in ALL_IDS if m in self.pos]
            if len(got) == len(ALL_IDS):
                print(f"    응답 완료: {sorted(got)}")
                return True
            if time.time() - t0 > timeout:
                missing = [m for m in ALL_IDS if m not in got]
                print(f"    timeout. 응답 없는 ID: {missing}")
                return False
            rate.sleep()
        return False

    def initial_hold(self):
        print(f"[2] 초기 hold ({INIT_HOLD_TIME}s) — kp ramp-up")
        rate = rospy.Rate(LOOP_HZ)
        with self.lock:
            temp = dict(self.pos)
        for mid in ALL_IDS:
            self.prev_cmd[mid] = temp[mid]
        steps = int(INIT_HOLD_TIME * LOOP_HZ)
        for i in range(steps):
            if self.stop:
                return False
            ratio = min(1.0, (i / steps) * 2)
            for mid in ALL_IDS:
                kp = (ELBOW_KP if mid in (ID_L_ELB, ID_R_ELB) else SHOULDER_KP) * ratio
                self._send(mid, temp[mid], kp=kp)
            rate.sleep()
        return True

    def calibrate(self):
        print("[3] 차렷 자세 0점 캘리브레이션")
        time.sleep(0.3)
        with self.lock:
            for mid in ALL_IDS:
                self.start_pos[mid] = self.pos[mid]
                self.prev_cmd[mid] = self.start_pos[mid]
        for mid in ALL_IDS:
            print(f"    {ID_NAME[mid]:12s} (ID {mid}) start = {self.start_pos[mid]:+.4f} rad")

    def hold_until_enter(self):
        print("\n[4] 팔/머리 주변 1.5m 비웠는지 확인 후 Enter")
        ready = {"v": False}

        def waiter():
            try:
                input("> ")
            except Exception:
                pass
            ready["v"] = True

        threading.Thread(target=waiter, daemon=True).start()
        rate = rospy.Rate(LOOP_HZ)
        while not ready["v"] and not rospy.is_shutdown() and not self.stop:
            for mid in ALL_IDS:
                kp = ELBOW_KP if mid in (ID_L_ELB, ID_R_ELB) else SHOULDER_KP
                self._send(mid, self.start_pos[mid], kp=kp)
            rate.sleep()
        return not (self.stop or rospy.is_shutdown())

    @staticmethod
    def _delayed(hist, target_t):
        """hist=deque[(t, val)] 에서 target_t 시각 값을 선형보간."""
        if not hist:
            return 0.0
        if target_t <= hist[0][0]:
            return hist[0][1]
        for i in range(len(hist) - 1, 0, -1):
            t1, v1 = hist[i]
            t0, v0 = hist[i - 1]
            if t0 <= target_t <= t1:
                r = (target_t - t0) / max(t1 - t0, 1e-6)
                return v0 + (v1 - v0) * r
        return hist[-1][1]

    def walking_arms(self):
        print("\n[5] 걷기 팔 — anti-phase swing + 팔꿈치 secondary + compliance")
        print("    Ctrl+C 로 종료")
        rate = rospy.Rate(LOOP_HZ)
        t0 = time.time()
        swing_hist = deque(maxlen=int(LOOP_HZ * 0.5))
        b = dict(self.start_pos)

        while not self.stop and not rospy.is_shutdown():
            if not self._health_ok():
                self.stop = True
                return False
            t = time.time() - t0

            # gait 속도 (0~1). 통합 시: gait = min(1, |실제_선속도| / max_speed)
            gait = 0.5 + 0.5 * math.sin(2 * math.pi * t / GAIT_CYCLE) if DEMO_GAIT else 1.0

            amp = SWING_AMP_BASE + gait * SWING_AMP_GAIN
            swing = amp * math.sin(2 * math.pi * SWING_FREQ * t)
            swing_hist.append((t, swing))
            delayed = self._delayed(swing_hist, t - ELBOW_DELAY)

            # 어깨 pitch: 양쪽 +swing → 거울대칭(3=앞+, 4=앞-)이라 자동 anti-phase
            self._send(ID_L_SHO_P, b[ID_L_SHO_P] + swing, SHOULDER_KP)
            self._send(ID_R_SHO_P, b[ID_R_SHO_P] + swing, SHOULDER_KP)

            # 어깨 roll: 살짝 벌림 (1=오 밖+, 2=왼 밖-)
            self._send(ID_R_SHO_R, b[ID_R_SHO_R] + ARM_BASE_OUT, SHOULDER_KP)
            self._send(ID_L_SHO_R, b[ID_L_SHO_R] - ARM_BASE_OUT, SHOULDER_KP)

            # 팔꿈치: 기본굽힘 + 지연 swing (7=왼 앞+, 8=오 앞-)
            sec = ELBOW_FOLLOW * delayed
            self._send(ID_L_ELB, b[ID_L_ELB] + ELBOW_BASE_BEND + sec, ELBOW_KP)
            self._send(ID_R_ELB, b[ID_R_ELB] - ELBOW_BASE_BEND + sec, ELBOW_KP)

            # 목 고정 (통합 시 head controller 가 가져감)
            self._send(ID_NECK_P, b[ID_NECK_P], NECK_KP)
            self._send(ID_NECK_Y, b[ID_NECK_Y], NECK_KP)

            rate.sleep()
        return True

    def ramp_down(self):
        print("\n[6] ramp-down")
        steps = int(RAMP_TIME * LOOP_HZ)
        rate = rospy.Rate(LOOP_HZ)
        with self.lock:
            final = dict(self.prev_cmd)
        for i in range(steps + 1):
            ratio = 1.0 - (i / steps)
            for mid in ALL_IDS:
                if mid in final:
                    base_kp = ELBOW_KP if mid in (ID_L_ELB, ID_R_ELB) else SHOULDER_KP
                    self._send_raw(mid, final[mid], kp=base_kp * ratio)
            rate.sleep()
        # Kp와 Kd를 모두 0으로 내려 해제한다.
        for _ in range(30):
            with self.lock:
                for mid in ALL_IDS:
                    cur = self.pos.get(mid, final.get(mid, 0.0))
                    self._send_raw(mid, cur, kp=0.0, kd=0.0)
            time.sleep(0.02)
        print("    모터 해제 완료")

    def run(self):
        print("=" * 60)
        print(" 상체 walking-arm 데모 (TRON1 없이 제자리)")
        print("=" * 60)
        try:
            if not self.wait_connect():
                return
            if not self.initial_hold():
                return
            self.calibrate()
            if not self.hold_until_enter():
                return
            self.walking_arms()
            print("\n[Done]")
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
            self.stop = True
        finally:
            self.ramp_down()


if __name__ == "__main__":
    WalkingArmsDemo().run()
