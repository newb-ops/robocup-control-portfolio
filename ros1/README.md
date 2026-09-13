# ROS1 상체 제어

`src/tron_upper_body`에는 모터 명령과 피드백 메시지, STM32 브리지, 실물 시험 코드가 있습니다.

| 파일 | 내용 |
|---|---|
| `msg/MotorCommand.msg`, `msg/MotorFeedback.msg` | ROS 메시지 정의 |
| `scripts/bridges/mit_bridge_node.py` | ROS 명령과 STM32 Serial 패킷 연결 |
| `scripts/competition-websocket/full_controll.py` | 구형 6모터 구성에서 사용한 WebSocket 통합 코드 |
| `scripts/upper-body-8motor/phase1_ping.py` | ID 응답과 현재 위치 확인 |
| `scripts/upper-body-8motor/phase2_hold.py` | 선택 관절의 낮은 게인 hold |
| `scripts/upper-body-8motor/phase3_integrated.py` | 상체 통합 동작 시험 |
| `scripts/upper-body-8motor/walking_arm.py` | 보행에 맞춘 팔 동작 시험 |

## 파일별 하드웨어 구성

6모터 코드에는 팔꿈치가 없습니다. 8모터 코드도 서로 다른 시기의 시험본이라 좌우 어깨와 목 ID가 동일하지 않습니다. `phase1/2`의 어깨 ID와 `phase3/walking_arm`의 어깨 ID가 다르고, `walking_arm`의 목 yaw/pitch도 다른 파일과 다릅니다. 파일명을 따라 연속 실행하지 말고 각 코드의 매핑을 실물과 대조해야 합니다.

ping 코드에도 감쇠 게인이 있어 완전한 무토크 시험은 아닙니다. 브리지는 파일을 import할 때 Serial 장치를 열기 때문에 일반 PC에서 import 검사로 실행하지 않습니다.

## 빌드와 실행

ROS Noetic과 catkin 환경에서 다음과 같이 메시지 패키지를 빌드합니다.

```bash
cd ros1
catkin_make
source devel/setup.bash
```

스크립트 설치 규칙이 없어 파일 경로로 실행합니다. 장비를 연결하고 설정을 확인한 후 사용할 브리지 진입점은 다음과 같습니다.

```bash
python3 src/tron_upper_body/scripts/bridges/mit_bridge_node.py
```

스크립트에 따라 rospy, pyserial, NumPy, OpenCV, sensor_msgs, cv_bridge, tf, Ultralytics와 websocket-client가 필요합니다. 원본 package.xml은 모든 실행 의존성을 선언하지 않습니다.

구형 `full_controll.py`가 요구하는 `weights/best.pt`는 포함돼 있지 않습니다. 이 파일은 당시 제어 구조를 보여 주는 소스로 보존했습니다. 경기 v10.4.2의 ONNX 모델을 그대로 대체 입력으로 사용할 수 있다고 보장하지 않습니다.
