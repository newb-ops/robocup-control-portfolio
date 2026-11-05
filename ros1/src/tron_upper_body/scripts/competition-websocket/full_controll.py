#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""구형 6모터 상체와 TRON1 보행을 연결한 ROS1 제어 코드.
공 추적 결과를 받아 WebSocket 보행 명령과 ROS 모터 명령을 보낸다."""

import rospy
import cv2
import numpy as np
import json
import time
import threading
import math
import websocket
import uuid
import os
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# 의존성 더미 클래스
try:
    from tron_upper_body.msg import MotorCommand, MotorFeedback
except ImportError:
    rospy.logwarn("tron_upper_body 패키지 없음. 더미 클래스 사용.")
    class MotorCommand: pass
    class MotorFeedback: pass


# 공 속도 추정과 위치 예측

class BallVelocityEstimator:
    """EMA 필터를 사용한 공 속도 추정 및 위치 예측"""
    def __init__(self, history_size=5, alpha=0.3):
        self.history = deque(maxlen=history_size)
        self.alpha = alpha  # 평활화 계수
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def update(self, x, y, timestamp):
        """위치 업데이트 및 속도 계산"""
        self.history.append((x, y, timestamp))

        if len(self.history) >= 2:
            (x1, y1, t1) = self.history[-2]
            (x2, y2, t2) = self.history[-1]
            dt = t2 - t1

            if dt > 0.001:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                # 지수 이동 평균(EMA) 적용
                self.velocity_x = self.alpha * vx + (1 - self.alpha) * self.velocity_x
                self.velocity_y = self.alpha * vy + (1 - self.alpha) * self.velocity_y

    def predict_position(self, current_x, current_y, time_delta):
        """미래 위치 예측 (블라인드 킥용)"""
        pred_x = current_x + self.velocity_x * time_delta
        pred_y = current_y + self.velocity_y * time_delta
        return pred_x, pred_y

    def get_speed(self):
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)

    def reset(self):
        self.history.clear()
        self.velocity_x = 0.0
        self.velocity_y = 0.0



class BallFollowerDualArm_Master:
    def __init__(self):
        rospy.init_node("ball_follower_master", anonymous=True)

        print("\n" + "="*70)
        print("🏆 [MASTER INTEGRATED VERSION]")
        print("   1. Protocol: Universal GUID & SDK Boot")
        print("   2. Logic: Predictive Blind Kick & Velocity Est.")
        print("   3. Hardware: Corrected Signs (Roll: Out--, Pitch: L+ R+)")
        print("="*70 + "\n")


        # 통신 설정

        self.USE_DASHES_IN_GUID = False  # True: 대시 포함, False: 대시 제거(SDK)
        self.ENABLE_RECOVERY_ON_STARTUP = False # SDK는 기본적으로 False


        # 장치와 경로

        self.DEBUG = True
        self.ROBOT_SN = "PF_TRON1A_001"
        self.WS_URL = "ws://192.0.2.11:5000"
        self.model_path = "weights/best.pt"


        # 모터 ID

        self.HEAD_YAW_ID = 5; self.HEAD_PITCH_ID = 6
        self.L_ROLL_ID = 1; self.L_PITCH_ID = 3
        self.R_ROLL_ID = 2; self.R_PITCH_ID = 4

        self.ARM_IDS = [1, 2, 3, 4]
        self.HEAD_IDS = [5, 6]
        self.ALL_MOTOR_IDS = self.ARM_IDS + self.HEAD_IDS


        # 제어 파라미터

        # Vision
        self.CONF_THRESHOLD = 0.75
        self.MIN_BALL_AREA = 500
        self.DETECT_CONFIRM_FRAMES = 2
        self.BALL_AREA_THRESHOLD = 6000

        # Blind Kick
        self.BLIND_KICK_PITCH = -0.35
        self.BLIND_KICK_AREA = 4000
        self.BLIND_KICK_YAW_TOLERANCE = 0.20
        self.BLIND_KICK_WAIT = 0.15
        self.NORMAL_KICK_WAIT = 0.8

        # 이동하는 공의 접근 속도
        self.MOVING_BALL_THRESHOLD = 30.0
        self.APPROACH_SPEED_SLOW = 0.08
        self.APPROACH_SPEED_FAST = 0.12

        # Safety & Balance
        self.ROLL_WARNING_LIMIT = 0.25
        self.ROLL_FALL_LIMIT = 0.35
        self.KICK_ZMP_FACTOR = 0.35

        # Timing
        self.COOLDOWN_TIME = 2.5
        self.RECOVER_TIME = 1.0
        self.TRACK_LOSS_THRESHOLD = 15



        self.motor_pos = {}; self.motor_offsets = {}; self.motors_ready = False
        self.head_target_yaw = 0.0; self.head_target_pitch = 0.0
        self.prev_cmd_x = 0.0; self.prev_cmd_wz = 0.0
        self.prev_arm_cmd = {}

        # ZMP Variables
        self.prev_roll = 0.0
        self.prev_roll_time = time.time()
        self.filtered_roll_rate = 0.0

        # State Flags
        self.scan_timer = 0.0
        self.is_blind_kick_active = False
        self.stop_confirm_count = 0
        self.detect_confirm_count = 0
        self.stop_confirm_limit = 3

        self.ball_estimator = BallVelocityEstimator(history_size=5, alpha=0.3)
        self.last_detection_time = 0.0

        # Memory for Blind Kick
        self.last_known_area = 0.0
        self.last_known_pitch = 0.0
        self.last_known_yaw = 0.0
        self.last_known_x = 0.0
        self.last_known_y = 0.0

        # Gains
        self.HEAD_GAIN_X = 0.0008; self.HEAD_GAIN_Y = 0.0008
        self.HEAD_YAW_LIMIT = [-1.0, 1.0]
        self.HEAD_PITCH_LIMIT = [-0.6, 0.5]

        self.BODY_TURN_DEADZONE = 0.08
        self.BODY_TURN_GAIN = 1.0
        self.ACC_LIMIT = 0.04
        self.DEC_LIMIT = 0.08
        self.BODY_SLEW_RATE = 0.1

        # Profiles
        self.MAX_VX = {
            "SEARCH": 0.0, "TRACK": 0.0, "APPROACH": 0.10,
            "STOP": 0.0, "KICK": 0.05, "COOLDOWN": -0.05, "RECOVER": 0.0
        }
        self.KP_TABLE = {
            "SEARCH": 15.0, "TRACK": 18.0, "APPROACH": 18.0,
            "STOP": 20.0, "KICK": 15.0, "COOLDOWN": 15.0, "RECOVER": 20.0
        }

        self.lock = threading.Lock()
        self.img_lock = threading.Lock()
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.latest_rgb_img = None
        self.new_frame_available = False
        self.is_connected = False
        self.is_robot_ready = False
        self.stand_done_event = threading.Event()
        self.walk_done_event = threading.Event()
        self.state = "SEARCH"
        self.lost_frame_count = 0
        self.kick_start_time = 0.0
        self.cooldown_start_time = 0.0
        self.recover_start_time = 0.0
        self.current_roll = 0.0

        # ROS Setup
        self.motor_pub = rospy.Publisher('/upper_body/cmd', MotorCommand, queue_size=20)
        rospy.Subscriber('/upper_body/feedback', MotorFeedback, self.feedback_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)

        self.ws = websocket.WebSocketApp(
            self.WS_URL, on_open=self.on_open, on_message=self.on_message,
            on_error=self.on_error, on_close=self.on_close
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()
        threading.Thread(target=self.control_loop, daemon=True).start()
        threading.Thread(target=self.vision_loop, daemon=True).start()


    # 유틸리티 (GUID, State)

    def generate_guid(self):
        guid = str(uuid.uuid4())
        if not self.USE_DASHES_IN_GUID:
            guid = guid.replace("-", "")
        return guid

    def set_state(self, new_state):
        with self.lock:
            if self.state != new_state:
                if self.DEBUG: print(f"🔄 State: {self.state} -> {new_state}")
                self.state = new_state

    def get_state(self):
        with self.lock: return self.state

    def feedback_callback(self, msg): self.motor_pos[msg.id] = msg.position
    def rgb_callback(self, msg):
        try:
            with self.img_lock:
                self.latest_rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.new_frame_available = True
        except: pass


    # WebSocket 핸들러

    def on_open(self, ws):
        print(f"✅ WebSocket Connected: {self.WS_URL}")
        self.is_connected = True
        self.send_ws_cmd("request_enable_imu", {"enable": True})

    def on_error(self, ws, error): print(f"🔥 WS Error: {error}")
    def on_close(self, ws, c, m): print(f"❌ WS Closed: {m}"); self.is_connected = False

    def on_message(self, ws, msg):
        try:
            data = json.loads(msg)
            title = data.get('title', '')
            result = data.get('data', {}).get('result', '')

            if 'response' in title:
                if result != 'success' and result != '' and self.DEBUG:
                    print(f"⚠️ Command Rejected [{title}]: {result}")

            elif title == 'notify_imu':
                imu = data.get('data', {})
                roll = 0.0
                if 'euler' in imu:
                    e = imu['euler']
                    roll = e[0] if isinstance(e, list) and len(e)>0 else e.get('roll', 0.0)
                elif 'roll' in imu: roll = imu['roll']
                with self.lock: self.current_roll = math.radians(roll)

            elif title == 'notify_stand_mode':
                if result == 'success': print("✅ Stand Mode OK"); self.stand_done_event.set()
            elif title == 'notify_walk_mode':
                if result == 'success': print("✅ Walk Mode OK"); self.walk_done_event.set()
        except: pass

    def send_ws_cmd(self, title, data):
        if self.is_connected:
            msg = {
                "accid": self.ROBOT_SN, "title": title,
                "timestamp": int(time.time()*1000),
                "guid": self.generate_guid(), "data": data
            }
            self.ws.send(json.dumps(msg))


    # 공 검출과 추적

    def vision_loop(self):
        rate = rospy.Rate(30)
        print("👁️ Vision Loop Started (Advanced Mode)")

        while not rospy.is_shutdown():
            try:
                curr = self.get_state()
                if curr in ["RECOVER", "COOLDOWN"]: rate.sleep(); continue
                if not self.motors_ready: rate.sleep(); continue

                frame = None
                with self.img_lock:
                    if self.new_frame_available:
                        frame = self.latest_rgb_img.copy(); self.new_frame_available = False

                if frame is None:
                    if curr == "SEARCH":
                        with self.lock:
                            self.scan_timer += 0.04
                            self.head_target_yaw = 0.6 * math.sin(self.scan_timer)
                            self.head_target_pitch = -0.2 + (0.15 * math.sin(2 * self.scan_timer))
                    rate.sleep(); continue

                results = self.model.predict(frame, conf=self.CONF_THRESHOLD, verbose=False)

                raw_det = False; temp_area = 0.0; temp_x = 0.0; temp_y = 0.0
                if len(results[0].boxes) > 0:
                    h, w, _ = frame.shape
                    best = max(results[0].boxes, key=lambda b: b.conf[0])
                    bx, by, bw, bh = best.xywh.cpu().numpy()[0]
                    temp_area = bw * bh
                    temp_x = bx; temp_y = by

                    if temp_area > self.MIN_BALL_AREA:
                        raw_det = True; self.detect_confirm_count += 1
                    else: self.detect_confirm_count = 0
                else: self.detect_confirm_count = 0

                detected = (self.detect_confirm_count >= self.DETECT_CONFIRM_FRAMES)
                area = 0.0; err_x = 0.0; err_y = 0.0

                if detected:
                    err_x = (w/2) - temp_x; err_y = (h/2) - temp_y; area = temp_area

                with self.lock:
                    self.ball_area = area

                    if detected:
                        self.lost_frame_count = 0
                        self.last_detection_time = time.time()

                        self.ball_estimator.update(temp_x, temp_y, time.time())

                        self.head_target_yaw += np.clip(err_x, -100, 100) * self.HEAD_GAIN_X
                        self.head_target_pitch += np.clip(err_y, -100, 100) * self.HEAD_GAIN_Y

                        self.head_target_yaw = np.clip(self.head_target_yaw, *self.HEAD_YAW_LIMIT)
                        self.head_target_pitch = np.clip(self.head_target_pitch, *self.HEAD_PITCH_LIMIT)

                        self.last_known_area = area; self.last_known_pitch = self.head_target_pitch
                        self.last_known_yaw = self.head_target_yaw
                        self.last_known_x = temp_x; self.last_known_y = temp_y

                        if self.state == "SEARCH": self.set_state("TRACK")

                    else:
                        self.lost_frame_count += 1
                        time_since_last = time.time() - self.last_detection_time

                        if self.state == "APPROACH":
                            is_blind = (self.last_known_pitch < self.BLIND_KICK_PITCH) and \
                                       (self.last_known_area > self.BLIND_KICK_AREA)

                            if is_blind:
                                # 위치 예측
                                px, py = self.ball_estimator.predict_position(
                                    self.last_known_x, self.last_known_y, time_since_last)

                                pred_err_x = (w/2) - px
                                pred_yaw = self.last_known_yaw + (pred_err_x * self.HEAD_GAIN_X)
                                is_aligned = abs(pred_yaw) < self.BLIND_KICK_YAW_TOLERANCE

                                if is_aligned:
                                    print(f"👻 [Blind Kick] 예측 Yaw: {pred_yaw:.2f} -> KICK!")
                                    self.kick_start_time = time.time()
                                    self.stop_confirm_count = 0
                                    self.is_blind_kick_active = True
                                    self.head_target_yaw = pred_yaw # 예측 방향 주시
                                    self.set_state("STOP")
                                else:
                                    print(f"⚠️ [Blind Fail] 정렬 안됨: {pred_yaw:.2f}")
                                    self.set_state("SEARCH")

                            elif self.lost_frame_count > self.TRACK_LOSS_THRESHOLD:
                                self.ball_estimator.reset()
                                self.set_state("SEARCH")

                        elif self.state == "SEARCH":
                            self.scan_timer += 0.04
                            self.head_target_yaw = 0.6 * math.sin(self.scan_timer)
                            self.head_target_pitch = -0.2 + (0.15 * math.sin(2 * self.scan_timer))

                        elif self.state == "TRACK" and self.lost_frame_count > self.TRACK_LOSS_THRESHOLD:
                             self.ball_estimator.reset()
                             self.set_state("SEARCH")
                rate.sleep()
            except Exception as e:
                print(f"🔥 비전 오류: {e}"); time.sleep(1.0)


    # 보행과 상체 제어

    def control_loop(self):
        rate = rospy.Rate(50)

        print("[부팅] 모터 연결 중...")
        while not rospy.is_shutdown():
             conn = [m for m in self.ALL_MOTOR_IDS if m in self.motor_pos]
             if len(conn) == len(self.ALL_MOTOR_IDS): break
             for m in self.ALL_MOTOR_IDS:
                 if m not in self.motor_pos: self.send_motor_raw(m, 0, 0)
             time.sleep(0.1)

        print("[부팅] 웹소켓 연결 확인...")
        while not self.is_connected and not rospy.is_shutdown(): time.sleep(1.0)

        if self.ENABLE_RECOVERY_ON_STARTUP:
            self.send_ws_cmd("request_recovery", {}); time.sleep(3.0)

        print("🔹 기립 요청...")
        self.send_ws_cmd("request_stand_mode", {})
        while not self.stand_done_event.is_set():
             if rospy.is_shutdown(): return
             for mid in self.ARM_IDS: self.send_motor_raw(mid, 0.0, kp=0.0)
             rate.sleep()
        time.sleep(1.0)

        with self.lock:
            for mid in self.ARM_IDS: self.motor_offsets[mid] = self.motor_pos[mid]
            self.motor_offsets[self.HEAD_PITCH_ID] = self.motor_pos[self.HEAD_PITCH_ID]
            self.motor_offsets[self.HEAD_YAW_ID] = self.motor_pos.get(self.HEAD_YAW_ID, 0.0)

        print("🔹 머리 정렬...")
        for i in range(50):
            self.send_motor_raw(self.HEAD_PITCH_ID, self.motor_offsets[self.HEAD_PITCH_ID] + (1.5 * i/50.0), kp=20.0)
            rate.sleep()

        print("🔹 보행 모드 요청...")
        self.send_ws_cmd("request_walk_mode", {})
        self.walk_done_event.wait(10); self.is_robot_ready = True; self.motors_ready = True

        with self.lock:
             for m in self.ALL_MOTOR_IDS:
                 if m in self.motor_pos: self.motor_offsets[m] = self.motor_pos[m]

        self.prev_roll = self.current_roll
        self.prev_roll_time = time.time()
        start_time = time.time(); print("🤖 로봇 준비 완료! (Master Version)")
        last_debug_time = time.time()

        while not rospy.is_shutdown():
            if time.time() - last_debug_time > 3.0:
                bs = self.ball_estimator.get_speed()
                print(f"⚡ {self.state} | 공속도: {bs:.1f} | Roll: {self.current_roll:.2f}")
                last_debug_time = time.time()
            if not self.is_robot_ready: rate.sleep(); continue

            with self.lock:
                curr = self.state; yaw = self.head_target_yaw; pitch = self.head_target_pitch; ba = self.ball_area

            t = time.time() - start_time
            cvx, cwz = 0.0, 0.0
            kp = self.KP_TABLE.get(curr, 15.0)

            if curr == "SEARCH":
                cwz = 0.15; self.stop_confirm_count = 0; self.is_blind_kick_active = False

            elif curr == "TRACK":
                self.stop_confirm_count = 0; self.is_blind_kick_active = False
                cwz = 0.0 if abs(yaw) < self.BODY_TURN_DEADZONE else np.clip(yaw * self.BODY_TURN_GAIN, -0.6, 0.6)
                if abs(yaw) < 0.15: self.set_state("APPROACH")

            elif curr == "APPROACH":
                cwz = yaw * 0.8
                # Dynamic Speed
                bs = self.ball_estimator.get_speed()
                cvx = self.APPROACH_SPEED_SLOW if bs > self.MOVING_BALL_THRESHOLD else self.APPROACH_SPEED_FAST

                is_close = (ba > self.BALL_AREA_THRESHOLD) or (pitch < -0.4)
                # Tighter alignment for moving ball
                align_tol = 0.12 if bs > self.MOVING_BALL_THRESHOLD else 0.15

                if is_close and abs(yaw) < align_tol: self.stop_confirm_count += 1
                else: self.stop_confirm_count = 0

                if self.stop_confirm_count > self.stop_confirm_limit:
                    self.kick_start_time = time.time(); self.stop_confirm_count = 0
                    self.is_blind_kick_active = False; self.set_state("STOP")

            elif curr == "STOP":
                wait = self.BLIND_KICK_WAIT if self.is_blind_kick_active else self.NORMAL_KICK_WAIT
                if time.time() - self.kick_start_time > wait:
                    self.kick_start_time = time.time(); self.set_state("KICK")

            elif curr == "KICK":
                with self.lock: self.head_target_yaw = 0.0; self.head_target_pitch = -0.3
                cvx = self.MAX_VX["KICK"]
                if abs(self.current_roll) > self.ROLL_WARNING_LIMIT: cvx = 0.0
                if abs(self.current_roll) > self.ROLL_FALL_LIMIT:
                    print("🚨 넘어짐 감지!"); self.recover_start_time = time.time(); self.set_state("RECOVER")
                elif time.time() - self.kick_start_time > 0.5:
                    self.cooldown_start_time = time.time(); self.set_state("COOLDOWN")

            elif curr == "COOLDOWN":
                cvx = self.MAX_VX["COOLDOWN"]
                with self.lock: self.head_target_yaw = 0.0; self.head_target_pitch = 0.0
                if time.time() - self.cooldown_start_time > self.COOLDOWN_TIME:
                    self.ball_estimator.reset(); self.set_state("SEARCH")

            elif curr == "RECOVER":
                with self.lock: self.head_target_yaw = 0.0; self.head_target_pitch = 0.0
                if time.time() - self.recover_start_time > self.RECOVER_TIME:
                    self.ball_estimator.reset(); self.set_state("SEARCH")

            self.send_twist(cvx, cwz, curr)
            self.send_motor_offset(self.HEAD_YAW_ID, yaw, kp=20.0)
            self.send_motor_offset(self.HEAD_PITCH_ID, pitch, kp=20.0)

            # 몸체 기울기에 따른 팔 보상
            now = time.time(); dt = max(now - self.prev_roll_time, 1e-3)
            raw_rate = (self.current_roll - self.prev_roll) / dt
            self.filtered_roll_rate = (0.2 * raw_rate) + (0.8 * self.filtered_roll_rate)
            self.prev_roll = self.current_roll; self.prev_roll_time = now

            bal = 0.0
            if abs(self.current_roll) > 0.05:
                bal = -np.clip(((1.0 * self.current_roll) + (0.1 * self.filtered_roll_rate)) * 0.5, -0.15, 0.15)

            if curr == "KICK": bal *= self.KICK_ZMP_FACTOR

            swing = 0.0; base_roll = 0.25 # 음수 처리됨 (바깥쪽)
            if curr not in ["STOP", "KICK", "RECOVER", "COOLDOWN"]:
                swing = np.clip((abs(cvx) + abs(cwz)) * 2.0, 0.0, 0.8) * math.sin(t * 5.0)

            if curr == "KICK": swing = 1.0; base_roll = 0.3
            elif curr == "RECOVER": base_roll = 0.6; swing = 0.0

            # 장착 방향에 따른 부호: Roll은 바깥쪽(-), Pitch는 좌우 반대.
            # Left(1) Out(-): base(-) + bal(-) = More Out
            left_roll_cmd = np.clip(-base_roll + bal, -0.35, 0.35)
            # Right(2) Out(-): base(-) - bal(-) = Less Out (In)
            right_roll_cmd = np.clip(-base_roll - bal, -0.35, 0.35)

            # Left(3) Fwd(+), Right(4) Fwd(-) -> Both +swing creates cross gait
            self.send_motor_offset(self.L_ROLL_ID, left_roll_cmd, kp=kp)
            self.send_motor_offset(self.L_PITCH_ID, swing, kp=kp)

            self.send_motor_offset(self.R_ROLL_ID, right_roll_cmd, kp=kp)
            self.send_motor_offset(self.R_PITCH_ID, swing, kp=kp)

            rate.sleep()

    def send_motor_offset(self, mid, rel_pos, kp=15.0):
        if mid not in self.motor_offsets: return
        key = f"m_{mid}"
        prev = self.prev_arm_cmd.get(key, self.motor_offsets[mid])
        tgt = self.motor_offsets[mid] + rel_pos
        limit = 0.02 if self.state == "KICK" else 0.06
        safe = prev + np.clip(tgt - prev, -limit, limit)
        self.prev_arm_cmd[key] = safe
        self.send_motor_raw(mid, safe, kp)

    def send_motor_raw(self, mid, abs_pos, kp):
        cmd = MotorCommand(); cmd.id = mid; cmd.p_des = float(abs_pos); cmd.kp = float(kp)
        cmd.kd = 1.0; cmd.v_des = 0.0; cmd.t_ff = 0.0
        self.motor_pub.publish(cmd)

    def send_twist(self, x, z, state):
        if not self.is_robot_ready: return
        tvx = x if state in ["STOP", "RECOVER", "KICK", "COOLDOWN"] else np.clip(x, -0.12, 0.12)
        dx = tvx - self.prev_cmd_x
        if dx > 0: dx = min(dx, self.ACC_LIMIT)
        else: dx = max(dx, -self.DEC_LIMIT)
        fx = self.prev_cmd_x + dx
        fz = self.prev_cmd_wz + np.clip(z - self.prev_cmd_wz, -self.BODY_SLEW_RATE, self.BODY_SLEW_RATE)
        self.prev_cmd_x = fx; self.prev_cmd_wz = fz

        if self.is_connected:
            msg = {
                "accid": self.ROBOT_SN, "title": "request_twist",
                "timestamp": int(time.time() * 1000),
                "guid": self.generate_guid(),
                "data": {"x": float(fx), "y": 0.0, "z": float(fz)}
            }
            self.ws.send(json.dumps(msg))

if __name__ == "__main__":
    try:
        BallFollowerDualArm_Master()
        rospy.spin()
    except rospy.ROSInterruptException: pass
    except KeyboardInterrupt: print("\n👋 프로그램 종료")
