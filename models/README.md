# 로봇 모델

TRON1 하체와 팀에서 제작한 상체를 결합한 `SF_TRON1B`의 링크·관절 정의입니다. 기구 담당 팀원이 작업한 모델을 제어 구조와 함께 볼 수 있도록 담았습니다.

| 파일 | 내용 |
|---|---|
| [robot.urdf](SF_TRON1B/urdf/robot.urdf) | 하체 8관절과 상체 8관절의 링크·관절 정의 |
| [robot.xml](SF_TRON1B/xml/robot.xml) | MuJoCo 모델 |
| [controller.yaml](gazebo/controller.yaml) | Gazebo 상체 위치 제어 설정 |

파일마다 작업 시기가 달라 관절 이름과 설정을 각각 확인해야 합니다. 참조하는 STL/OBJ 메시와 강화학습 정책은 포함하지 않았으므로, 이 폴더만으로 시각화나 보행 시뮬레이션을 실행할 수는 없습니다.

TRON1 기본 모델의 출처는 [LimX Dynamics](https://github.com/limxdynamics/tron1-robot-description)입니다. 현재 공식 배포본은 참고할 수 있지만, 이 프로젝트가 참조하는 메시와 동일한 버전인지는 확인이 필요합니다. 라이선스와 수정 범위는 [NOTICE.md](../NOTICE.md)에 적었습니다.
