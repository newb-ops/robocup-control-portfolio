# 세계대회 준비 코드

한국대회 이후 비전 모듈을 분리하고 GameController 처리를 붙이던 때의 코드입니다. 원본의 v10.4.2 계열을 담았으며, 세계대회 자율 경기 전체를 완료한 버전은 아닙니다.

`tronb_vision/`에서 인식 결과와 경기 상태를 처리하고, `ball_follower_integrated.py`에서 ROS 상체 제어와 TRON1 WebSocket 명령으로 연결합니다.

| 파일 | 역할 |
|---|---|
| [ball_follower_integrated.py](ball_follower_integrated.py) | 카메라·모터 피드백·경기 상태를 받아 상하체 명령 출력 |
| [tronb_vision/control_fsm.py](tronb_vision/control_fsm.py) | 동작 상태와 전이 조건 |
| [tronb_realrobot.py](tronb_realrobot.py) | 실기 제어 베이스 |
| [run_camera.py](run_camera.py) | RealSense 영상과 ONNX 검출 결과 확인 |
| [selftest.py](selftest.py) | 가상 입력을 이용한 상태 전이 검사 |

## 가상 입력으로 확인

이 폴더에서 실행합니다.

```bash
python -m pip install numpy
python selftest.py
```

`DummyDetector`와 `DryRunDriver`를 사용합니다. 관측값이 유효한지, SEARCH·TRACK·APPROACH·AIM 상태를 거치는지 검사하고 KICK 도달 여부를 출력합니다. 실제 장비의 동작 시험은 포함하지 않습니다. 출력의 `V10.3`은 원본 검사 파일에 남아 있는 버전 표기입니다.

## 카메라 확인

NumPy, OpenCV, PyYAML, ONNX Runtime, librealsense가 필요합니다. `requirements_laptop.txt`와 `requirements_jetson.txt`를 참고하되, Jetson의 ONNX Runtime은 설치된 JetPack/CUDA 버전에 맞춰 준비합니다.

```bash
python run_camera.py --device laptop --config configs/laptop_realsense.yaml --weights weights/real_best.onnx
```

`weights/real_best.onnx`는 팀에서 사용한 실제 카메라용 검출 모델입니다. 학습 데이터, 영상, TensorRT 캐시는 포함하지 않았습니다.

## 로봇 연결

통합 코드에는 ROS1의 `rospy`, `sensor_msgs`, `cv_bridge`, 빌드된 `tron_upper_body` 메시지, Ultralytics, websocket-client가 추가로 필요합니다. 기존 requirements 파일에는 이 의존성이 모두 들어 있지 않습니다.

```bash
python ball_follower_integrated.py
```

이 진입점은 위 카메라 예제의 YAML을 읽지 않습니다. 코드 안의 연결 주소, 모터 ID, 영점과 관절 방향을 실제 구성에 맞춰 확인해야 합니다. 로봇을 지지하고 가동 범위와 정지 경로를 확인한 뒤 실행합니다.
