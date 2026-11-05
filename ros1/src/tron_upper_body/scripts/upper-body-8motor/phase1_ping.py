#!/usr/bin/env python3

# coding: utf-8

"""상체 모터의 ID 응답과 현재 위치를 확인한다.
Kp=0, Kd=0.1을 사용하므로 감쇠 토크가 발생할 수 있다.
관절을 지지하고 모터 ID와 실제 장착 위치를 대조한다."""

import rospy
import time
import sys
import signal
from tron_upper_body.msg import MotorCommand, MotorFeedback


# 응답을 확인할 모터 ID

ALL_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

ID_NAME = {
    1: "sho_L  (어깨 왼)",
    2: "sho_R  (어깨 오)",
    3: "arm_L  (팔 왼)",
    4: "arm_R  (팔 오)",
    5: "neck   (목 yaw)",
    6: "head   (목 pitch)",
    7: "elbow_L (팔꿈치 왼)",
    8: "elbow_R (팔꿈치 오)",
}


class Phase1Ping:
    def __init__(self):
        rospy.init_node("phase1_ping", anonymous=True)
        self.pos = {}
        self.initial_pos = {}
        self.stop = False

        self.pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=10)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self._cb)
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'stop', True))

    def _cb(self, msg):
        self.pos[msg.id] = msg.position
        if msg.id not in self.initial_pos:
            self.initial_pos[msg.id] = msg.position

    def _ping(self, mid):
        """위치 제어를 끄고 Kd=0.1의 감쇠만 인가한다."""
        cmd = MotorCommand()
        cmd.id = mid
        cmd.p_des = 0.0
        cmd.v_des = 0.0
        cmd.kp = 0.0
        cmd.kd = 0.1
        cmd.t_ff = 0.0
        self.pub.publish(cmd)

    def run(self):
        print("=" * 60)
        print(" Phase 1: ping 테스트 (Kp=0, Kd=0.1)")
        print("=" * 60)
        print(f" 테스트 대상 ID: {ALL_IDS}")
        print(" 모터는 움직이지 않음. 손으로 까딱여서 매핑 확인.")
        print()

        # 1) 연결 확인 (3초)

        print("[1/2] 모터 응답 확인 (3초)")
        rate = rospy.Rate(50)
        t0 = time.time()
        while time.time() - t0 < 3.0 and not self.stop:
            for mid in ALL_IDS:
                self._ping(mid)
            rate.sleep()

        connected = sorted(self.pos.keys())
        missing = [m for m in ALL_IDS if m not in self.pos]

        if missing:
            print(f"      응답 없는 ID: {missing}")
            print()
            print("  중단. 점검 항목:")
            print("  - STM 펌웨어 Enter_Motor_Mode 가 1~8 다 도는지")
            print("  - Cubemars 보드의 motor ID 설정")
            print("  - CAN 케이블 연결, 종단 저항")
            print("  - 모터 전원")
            return

        print(f"      응답: {connected}")
        print()
        print("  현재 위치 (시작 자세):")
        for mid in ALL_IDS:
            if mid in self.pos:
                tag = ID_NAME.get(mid, f"ID {mid}")
                print(f"    ID {mid} {tag:22s} = {self.pos[mid]:+.4f} rad")

        # 2) 인터랙티브 매핑 검증

        print()
        print("[2/2] 매핑 검증 — 각 관절을 손으로 까딱여보세요")
        print("       값(rad / deg) 이 시작점 기준 상대값으로 표시됨")
        print("       Ctrl+C 로 종료")
        print()
        time.sleep(1.0)

        rate = rospy.Rate(10)
        last_status = ""

        while not self.stop and not rospy.is_shutdown():
            for mid in ALL_IDS:
                self._ping(mid)

            # 모든 ID 의 상대 위치 표시

            lines = []
            for mid in ALL_IDS:
                if mid in self.pos and mid in self.initial_pos:
                    diff = self.pos[mid] - self.initial_pos[mid]
                    marker = "  ←" if abs(diff) > 0.02 else ""
                    tag = ID_NAME.get(mid, f"ID {mid}")
                    lines.append(
                        f"  ID {mid} {tag:22s} {diff:+.3f} rad ({diff*57.3:+5.1f} deg){marker}"
                    )

            status = "\n".join(lines)
            if status != last_status:
                sys.stdout.write("\033[2J\033[H")  # 화면 클리어

                print("[Phase 1] 매핑 검증 — Ctrl+C 로 종료\n")
                print(status)
                last_status = status

            rate.sleep()

        print("\n[End] 종료")


if __name__ == "__main__":
    Phase1Ping().run()
