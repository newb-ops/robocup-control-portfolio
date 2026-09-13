# 경기 운용 코드

v10.4.2 계열의 경기 코드입니다. `tronb_vision/`이 인식과 위치 추정, 경기 상태와 동작 결정을 담당하고 `ball_follower_integrated.py`가 ROS 상체 제어와 TRON1 WebSocket 명령에 연결합니다.

| 파일 | 역할 |
|---|---|
| `ball_follower_integrated.py` | 인식 결과와 GameController 상태를 받아 통합제어 |
| `tronb_vision/control_fsm.py` | 동작 상태와 전이 조건 |
| `tronb_realrobot.py` | 통합 과정에서 보존한 실기 제어 베이스 |
| `run_camera.py` | RealSense와 ONNX 모델을 이용한 인식 확인 |
| `selftest.py` | 가상 입력으로 실행하는 기존 검사 |

## 로봇 없는 검사

```bash
python -m pip install numpy
python selftest.py
```

`selftest.py`는 가상 입력과 `DryRunDriver`를 사용합니다. 실제 통합 진입점의 하드웨어 검증을 대신하지 않습니다. 출력에 남은 `V10.3` 표기는 원본 검사 파일의 문자열입니다.

## 카메라와 실제 로봇

카메라 예제에는 NumPy, OpenCV, PyYAML, ONNX Runtime과 librealsense가 필요합니다. `requirements_laptop.txt`와 `requirements_jetson.txt`는 원본의 참고 목록입니다. Jetson의 ONNX Runtime은 JetPack/CUDA 환경에 맞춰 설치합니다.

```bash
python run_camera.py --device laptop --config configs/laptop_realsense.yaml --weights weights/real_best.onnx
```

실기 통합 코드에는 추가로 ROS1의 `rospy`, `sensor_msgs`, `cv_bridge`, 빌드된 `tron_upper_body` 메시지와 Ultralytics, websocket-client가 필요합니다. 기존 requirements 파일은 이 통합 의존성을 모두 나열하지 않습니다.

`ball_follower_integrated.py`의 실행 명령은 `python ball_follower_integrated.py`입니다. 이 파일은 CLI 설정 파일을 받지 않으므로 내부 설정과 실제 장비 구성을 먼저 대조해야 합니다. 로봇을 지지하고 모터 ID, 영점, 방향, 가동 범위와 정지 경로를 확인한 환경에서만 실행합니다.

## 포함한 인식 모델

`weights/real_best.onnx`는 원본 경기 번들의 실제 카메라용 모델입니다. 10,610,390바이트이며 SHA-256은 `da40467615797e657e33026ce2d41b8338d8c437717499583965509dcdcc673e`입니다. 학습 데이터와 영상, TensorRT 캐시는 포함하지 않았습니다.
