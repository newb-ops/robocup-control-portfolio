#!/usr/bin/env python3
import rospy
import serial
import struct
from tron_upper_body.msg import MotorCommand, MotorFeedback

SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 921600

try:
   stm32 = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.001)
   print(f"\n[연결 성공] STM32 ({SERIAL_PORT}) @ {BAUDRATE}")
except Exception as e:
   print(f"\n[연결 실패] {e}")
   stm32 = None


def cmd_callback(msg):
   if stm32 is None:
       return
   try:
       packet_body = struct.pack('<Bfffff', msg.id, msg.p_des, msg.v_des, msg.kp, msg.kd, msg.t_ff)
       checksum = sum(packet_body) & 0xFF
       final_packet = b'\xFF\xFF' + packet_body + struct.pack('B', checksum)
       stm32.write(final_packet)
   except Exception as e:
       rospy.logerr(f"송신 에러: {e}")


_rx_buf = bytearray()   # 시리얼 누적 버퍼 (프레임 경계 복구용)


def read_and_publish(pub):
   if stm32 is None:
       return

   # 한 번의 read가 프레임 경계와 일치하지 않을 수 있다.
   n = stm32.in_waiting
   if n:
       _rx_buf.extend(stm32.read(n))

   # 버퍼에서 완전한 프레임(16B)을 찾아 처리. 헤더 0xEE 0xEE 동기화 + 체크섬 검증.
   while len(_rx_buf) >= 16:
       # 헤더 정렬: 0xEE 0xEE 로 시작 안 하면 1바이트씩 버림
       if not (_rx_buf[0] == 0xEE and _rx_buf[1] == 0xEE):
           del _rx_buf[0]
           continue

       frame = _rx_buf[:16]                 # EE EE | id p v t (13B) | checksum
       body = frame[2:15]                   # id + pos + vel + torque (STM 이 체크섬 계산한 범위)
       checksum = frame[15]
       if (sum(body) & 0xFF) != checksum:
           # 체크섬 불일치 = 깨진/어긋난 프레임 → 헤더 1바이트만 버리고 재동기화
           del _rx_buf[0]
           continue

       try:
           uid, pos, vel, tor = struct.unpack('<Bfff', bytes(body))
           if 1 <= uid <= 8:
               fb = MotorFeedback()
               fb.id = uid
               fb.position = pos
               fb.velocity = vel
               fb.torque = tor
               fb.temp = 0.0
               pub.publish(fb)
       except Exception:
           pass
       del _rx_buf[:16]


def main():
   rospy.init_node('mit_bridge_node')
   rospy.Subscriber('/upper_body/cmd', MotorCommand, cmd_callback)
   pub_feedback = rospy.Publisher('/upper_body/feedback', MotorFeedback, queue_size=10)

   print("심플 브리지 시작 — Ctrl+C 로 종료")
   rate = rospy.Rate(1000)

   while not rospy.is_shutdown():
       read_and_publish(pub_feedback)
       rate.sleep()


if __name__ == '__main__':
   main()
