# ROS1 상체 제어

`tron_upper_body` 패키지의 메시지와 STM32 Serial 브리지, 상체 시험 코드입니다. 한국대회의 6모터 구성과 이후 팔꿈치를 추가한 8모터 구성이 함께 들어 있습니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `msg/MotorCommand.msg`, `msg/MotorFeedback.msg` | 모터 명령과 피드백 메시지 |
| `scripts/bridges/mit_bridge_node.py` | ROS 메시지와 STM32 Serial 패킷 변환 |
| `scripts/competition-websocket/full_controll.py` | 6모터 구성의 공 추적·킥 및 상체 제어 |
| `scripts/upper-body-8motor/phase1_ping.py` | 모터 ID 응답과 현재 위치 확인 |
| `scripts/upper-body-8motor/phase2_hold.py` | 선택한 관절을 낮은 게인으로 유지 |
| `scripts/upper-body-8motor/phase3_integrated.py` | 상체 동작 통합 시험 |
| `scripts/upper-body-8motor/walking_arm.py` | 보행에 맞춘 팔 동작 시험 |

## 빌드

ROS Noetic과 catkin 환경에서 메시지 패키지를 빌드합니다.

```bash
cd ros1
catkin_make
source devel/setup.bash
```

스크립트 설치 규칙은 없어 파일 경로로 실행합니다. 브리지 진입점은 다음과 같습니다.

```bash
python3 src/tron_upper_body/scripts/bridges/mit_bridge_node.py
```

스크립트별로 rospy, pyserial, NumPy, OpenCV, sensor_msgs, cv_bridge, tf, Ultralytics, websocket-client를 사용합니다. `package.xml`에 빠진 실행 의존성은 별도로 설치해야 합니다.

## 연결 전에 확인할 점

8모터 코드는 서로 다른 시기의 시험본입니다. `phase1/2`와 `phase3/walking_arm`의 어깨 ID가 다르고, `walking_arm`의 목 yaw/pitch ID도 다른 파일과 다릅니다. 번호 순서대로 실행하기 전에 각 파일의 매핑을 실물과 대조해야 합니다.

`phase1_ping.py`에도 감쇠 게인이 설정돼 있어 모터에 토크가 걸릴 수 있습니다. 브리지는 import 시점에 Serial 장치를 엽니다.

6모터 코드인 `full_controll.py`는 `weights/best.pt`가 필요하지만 이 저장소에는 포함하지 않았습니다. 당시 구조를 읽을 수 있도록 남긴 파일이며, `competition`의 ONNX 모델을 그대로 대신 넣는 구성은 아닙니다.
