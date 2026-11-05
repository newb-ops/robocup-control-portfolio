#!/usr/bin/env python3
"""RealSense 영상에 객체 탐지를 적용하는 카메라 실행 진입점.
장치별 설정과 모델 경로는 실행 인자로 지정한다."""
import argparse
import sys

from tronb_vision.app import BallFollower
from tronb_vision.common import OperationMode, load_config
from tronb_vision.debug_view import draw_overlay, handle_key
from tronb_vision.detector import create_detector
from tronb_vision.drivers import RealStubDriver
from tronb_vision.localization_view import TopViewManager
from tronb_vision.sources import RealSenseD435Source, ThreadedCameraSource


DEFAULT_CONFIGS = {
    "laptop": "configs/laptop_realsense.yaml",
    "jetson": "configs/jetson_realsense.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["laptop", "jetson"], default="laptop")
    ap.add_argument("--config", default=None, help="생략하면 --device 에 맞는 RealSense config 사용")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--auto", action="store_true", help="시작부터 AUTO (기본 LOCKED)")
    ap.add_argument("--external-control", action="store_true",
                    help="제어 FSM(control_fsm)으로 구동 + robot_state→제어 흐름 로깅 (노트북 단독 검증). 자동 AUTO 진입.")
    ap.add_argument("--control-log", default=None,
                    help="external-control 시 매 프레임 robot_state/명령을 JSONL 로 기록할 경로")
    ap.add_argument("--headless", action="store_true", help="미리보기 창 없이 (배포)")
    ap.add_argument("--control-hz", type=float, default=50.0)
    args = ap.parse_args()

    config_path = args.config or DEFAULT_CONFIGS[args.device]
    cfg, tuning = load_config(config_path)
    if args.weights:
        cfg.model_path = args.weights
    if args.external_control:
        cfg.external_control = True
        cfg.control_log_path = args.control_log
    from tronb_vision.common import check_model_and_backend
    check_model_and_backend(cfg, "realsense")
    print(
        f"[run_camera] device={args.device} config={config_path} "
        f"backend={cfg.detector_backend} model={cfg.model_path}"
    )
    tuning.print_summary()

    src = ThreadedCameraSource(RealSenseD435Source(
        cfg.img_width,
        cfg.img_height,
        cfg.realsense_fps,
        serial=cfg.realsense_serial,
        use_imu_pitch=cfg.use_imu_pitch,
        use_sysfs_imu_pitch=cfg.use_sysfs_imu_pitch,
        pitch_down_deg=cfg.camera_pitch_down_deg,
        imu_pitch_alpha=cfg.imu_pitch_alpha,
        imu_pitch_offset_deg=cfg.imu_pitch_offset_deg,
        imu_accel_fps=cfg.imu_accel_fps,
        emitter=cfg.realsense_emitter,
        laser_power=cfg.realsense_laser_power,
    ))
    det = create_detector(cfg)
    drv = RealStubDriver(cfg)
    bf = BallFollower(cfg, tuning, src, drv, detector=det, bgr_input=False)  # RealSense=bgr8
    if args.auto or args.external_control:
        bf.set_mode(OperationMode.AUTO)
        if args.external_control:
            print("[run_camera] external-control: 제어 FSM 구동 + 흐름 로깅. AUTO 진입."
                  + (f" 로그={args.control_log}" if args.control_log else " (콘솔 상태전이만)"))

    if args.headless:
        print("[run_camera] headless. Ctrl+C 종료")
        bf.run_threaded(control_hz=args.control_hz)
        return 0

    # OpenCV 창과 키 입력은 메인 스레드에서 처리한다.
    import cv2
    bf.start_threads(control_hz=args.control_hz)
    win = "TRON1 Vision v10.3 (camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[run_camera] P=AUTO M=MANUAL X=LOCKED 1~9=toggle H=help ESC=quit")
    ui = {"toast": None, "show_help": False, "running": True}
    topview = TopViewManager()
    try:
        while ui["running"] and bf._run:
            view = bf._last_bgr
            if view is not None:
                cv2.imshow(win, draw_overlay(view, bf, ui["toast"], ui["show_help"]))
            topview.update(bf)               # localization/landmark_comp 켜졌을 때만
            key = cv2.waitKey(15) & 0xFF
            ui = handle_key(bf, key, ui, allow_manual_drive=False)  # 실카메라는 수동주행 비활성
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        topview.close()
        bf.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
