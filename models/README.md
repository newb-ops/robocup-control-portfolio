# 상하체 로봇 모델

TRON1 하체와 상체를 결합한 `SF_TRON1B` 모델 정의를 추렸습니다.

| 파일 | 내용 |
|---|---|
| `SF_TRON1B/urdf/robot.urdf` | 하체 8관절과 상체 8관절을 포함한 링크와 관절 구조 |
| `SF_TRON1B/xml/robot.xml` | 상하체 결합 모델의 MuJoCo 정의 |
| `gazebo/controller.yaml` | Gazebo 상체 위치 제어 설정 스냅샷 |

URDF와 MuJoCo XML은 같은 시기의 하드웨어와 모든 설정이 일치한다고 가정하지 않습니다. Gazebo 제어 설정도 파일에 나열된 관절 기준으로 확인해야 합니다.

참조하는 STL/OBJ 메시가 포함되지 않아 이 파일만으로 시각화와 물리 시뮬레이션을 완전히 실행할 수는 없습니다. 정책 모델도 포함하지 않았으며, 강화학습 보행 완료나 실물 적용의 증거로 제시하지 않습니다.

TRON1 기본 모델의 공식 출처는 [LimX Dynamics robot description](https://github.com/limxdynamics/tron1-robot-description)입니다. 현재 공식 모델은 참고 자료이며 이 프로젝트의 누락 메시와 같은 파일이라고 보장하지 않습니다. 기존 기본 모델의 권리와 출처는 [NOTICE](../NOTICE.md)에 적었습니다.
