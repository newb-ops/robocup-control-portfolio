from __future__ import annotations
"""실제 카메라, MuJoCo와 테스트용 영상 소스를 제공한다.

read()는 영상, 깊이, 촬영 시각과 프레임 ID를 담은 FramePacket을 반환한다.
ThreadedCameraSource는 별도 스레드에서 캡처하고 최신 프레임만 보관한다."""

import importlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int


@dataclass
class FramePacket:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    capture_time: float
    frame_id: int
    intrinsics: Optional[CameraIntrinsics] = None
    imu_pitch_deg: Optional[float] = None
    pitch_source: str = "manual"


class CameraSource:
    def start(self): ...
    def read(self) -> Optional[FramePacket]: ...
    def stop(self): ...
    # 시뮬 소스만 의미있음 (실 카메라는 None)
    def get_sim_handles(self):
        return None


class SysfsAccelReader:
    """Fallback for D435i accel exposed through Linux IIO sysfs."""
    DEVICE_NAMES = ("accel_3d",)

    def __init__(self, device_dir: Path):
        self.device_dir = device_dir
        self.x_raw = device_dir / "in_accel_x_raw"
        self.y_raw = device_dir / "in_accel_y_raw"
        self.z_raw = device_dir / "in_accel_z_raw"
        self.scale_file = device_dir / "in_accel_scale"
        self.offset_file = device_dir / "in_accel_offset"

    @staticmethod
    def discover() -> Optional["SysfsAccelReader"]:
        for device_dir in sorted(Path("/sys/bus/iio/devices").glob("iio:device*")):
            name_file = device_dir / "name"
            try:
                name = name_file.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if name not in SysfsAccelReader.DEVICE_NAMES:
                continue
            required = [device_dir / f"in_accel_{axis}_raw" for axis in ("x", "y", "z")]
            if all(path.exists() for path in required):
                return SysfsAccelReader(device_dir)
        return None

    @staticmethod
    def _read_float(path: Path, default: float) -> float:
        try:
            return float(path.read_text(encoding="utf-8").strip())
        except OSError:
            return float(default)

    def read_accel(self) -> np.ndarray:
        scale = self._read_float(self.scale_file, 1.0)
        offset = self._read_float(self.offset_file, 0.0)
        raw = np.array(
            [
                self._read_float(self.x_raw, 0.0),
                self._read_float(self.y_raw, 0.0),
                self._read_float(self.z_raw, 0.0),
            ],
            dtype=float,
        )
        return (raw + offset) * scale


class DummySource(CameraSource):
    """검정 프레임 + capture_time. headless/selftest 용."""
    def __init__(self, width=640, height=480):
        self.w, self.h, self._fid = width, height, 0
    def read(self) -> Optional[FramePacket]:
        self._fid += 1
        rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        depth = np.full((self.h, self.w), 1.5, dtype=np.float32)
        return FramePacket(rgb, depth, time.time(), self._fid)


class RealSenseD435Source(CameraSource):
    """Jetson Orin + RealSense D435. (pyrealsense2 필요)"""
    def __init__(self, width=640, height=480, fps=30, align_depth=True,
                 serial: str = "", use_imu_pitch: bool = False,
                 use_sysfs_imu_pitch: bool = True,
                 pitch_down_deg: float = 20.0, imu_pitch_alpha: float = 0.15,
                 imu_pitch_offset_deg: float = 0.0, imu_accel_fps: int = 0,
                 emitter: float = -1.0, laser_power: float = -1.0):
        self.w, self.h, self.fps, self.align_depth = width, height, fps, align_depth
        self.serial = str(serial or "")
        self.use_imu_pitch = bool(use_imu_pitch)
        self.use_sysfs_imu_pitch = bool(use_sysfs_imu_pitch)
        self.pitch_down_deg = float(pitch_down_deg)
        self.imu_pitch_alpha = float(np.clip(imu_pitch_alpha, 0.0, 1.0))
        self.imu_pitch_offset_deg = float(imu_pitch_offset_deg)
        self.imu_accel_fps = int(imu_accel_fps)
        self.emitter = float(emitter)
        self.laser_power = float(laser_power)
        self.rs = None; self.pipeline = None; self.align = None; self._fid = 0
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.depth_scale = 0.001
        self.imu_enabled = False
        self._imu_has_measurement = False
        self._accel_to_color = None
        self._pitch_source = "manual"
        self._sysfs_accel: Optional[SysfsAccelReader] = None
        self._no_accel_frame_warned = False

    @staticmethod
    def _fold_pitch_deg(angle_deg: float) -> float:
        angle = float(angle_deg)
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return angle

    def _enable_streams(self, cfg, accel_fps: Optional[int]) -> None:
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(self.rs.stream.color, self.w, self.h, self.rs.format.bgr8, self.fps)
        cfg.enable_stream(self.rs.stream.depth, self.w, self.h, self.rs.format.z16, self.fps)
        if accel_fps is not None:
            if int(accel_fps) > 0:
                cfg.enable_stream(self.rs.stream.accel, self.rs.format.motion_xyz32f, int(accel_fps))
            else:
                cfg.enable_stream(self.rs.stream.accel, self.rs.format.motion_xyz32f)

    def _start_pipeline(self):
        accel_candidates = []
        if self.use_imu_pitch:
            if self.imu_accel_fps > 0:
                accel_candidates.append(self.imu_accel_fps)
            accel_candidates.extend([0, 100, 200])
        accel_candidates.append(None)
        seen = set()
        last_exc = None
        for accel_fps in accel_candidates:
            if accel_fps in seen:
                continue
            seen.add(accel_fps)
            self.pipeline = self.rs.pipeline()
            cfg = self.rs.config()
            self._enable_streams(cfg, accel_fps)
            try:
                profile = self.pipeline.start(cfg)
                self.imu_enabled = accel_fps is not None
                if self.use_imu_pitch and not self.imu_enabled:
                    print("[RealSense] IMU pitch disabled; accel stream unavailable")
                return profile
            except RuntimeError as exc:
                last_exc = exc
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None
                if accel_fps is not None:
                    label = f"{accel_fps}Hz" if int(accel_fps) > 0 else "auto"
                    print(f"[RealSense] accel {label} failed: {exc}")
        raise last_exc or RuntimeError("RealSense pipeline start failed")

    def _apply_accel_pitch(self, accel: np.ndarray, source: str) -> bool:
        accel = np.asarray(accel, dtype=float)
        norm = float(np.linalg.norm(accel))
        if not np.isfinite(norm) or norm < 1e-6:
            return False
        measured = self._fold_pitch_deg(math.degrees(math.atan2(float(accel[2]), float(accel[1]))))
        measured += self.imu_pitch_offset_deg
        if self._imu_has_measurement:
            self.pitch_down_deg = (1.0 - self.imu_pitch_alpha) * self.pitch_down_deg + self.imu_pitch_alpha * measured
        else:
            self.pitch_down_deg = measured
            self._imu_has_measurement = True
        self._pitch_source = source
        return True

    def _update_pitch_from_sysfs(self) -> bool:
        if self._sysfs_accel is None:
            return False
        try:
            return self._apply_accel_pitch(self._sysfs_accel.read_accel(), "sysfs_imu")
        except OSError:
            return False

    def _update_pitch_from_frames(self, frames) -> None:
        if not self.imu_enabled:
            self._update_pitch_from_sysfs()
            return
        accel_frame = frames.first_or_default(self.rs.stream.accel)
        if not accel_frame:
            if self._update_pitch_from_sysfs():
                return
            if self.use_imu_pitch and not self._no_accel_frame_warned:
                print("[RealSense] accel stream started, but no accel frame received yet")
                self._no_accel_frame_warned = True
            return
        motion = accel_frame.as_motion_frame().get_motion_data()
        accel = np.array([motion.x, motion.y, motion.z], dtype=float)
        if self._accel_to_color is not None:
            try:
                origin = np.array(self.rs.rs2_transform_point_to_point(self._accel_to_color, [0.0, 0.0, 0.0]), dtype=float)
                point = np.array(self.rs.rs2_transform_point_to_point(self._accel_to_color, accel.tolist()), dtype=float)
                accel = point - origin
            except Exception:
                pass
        self._apply_accel_pitch(accel, "imu")

    def start(self):
        self.rs = importlib.import_module("pyrealsense2")
        print(f"[RealSense] request imu_pitch={self.use_imu_pitch} sysfs_fallback={self.use_sysfs_imu_pitch}")
        profile = self._start_pipeline()
        if self.use_imu_pitch and self.use_sysfs_imu_pitch:
            self._sysfs_accel = SysfsAccelReader.discover()
            if self._sysfs_accel is not None:
                print(f"[RealSense] sysfs accel fallback: {self._sysfs_accel.device_dir}")
        if self.align_depth:
            self.align = self.rs.align(self.rs.stream.color)
        device = profile.get_device()
        try:
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = float(depth_sensor.get_depth_scale())
            if self.emitter >= 0:
                depth_sensor.set_option(self.rs.option.emitter_enabled, self.emitter)
            if self.laser_power >= 0:
                depth_sensor.set_option(self.rs.option.laser_power, self.laser_power)
        except Exception as exc:
            print(f"[RealSense] depth option warning: {exc}")
        color_profile = profile.get_stream(self.rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(float(intr.fx), float(intr.fy), float(intr.ppx),
                                           float(intr.ppy), int(intr.width), int(intr.height))
        if self.imu_enabled:
            try:
                accel_profile = profile.get_stream(self.rs.stream.accel)
                self._accel_to_color = accel_profile.get_extrinsics_to(color_profile)
            except Exception as exc:
                print(f"[RealSense] accel-to-color extrinsics unavailable: {exc}")
        print(f"[RealSense] {self.w}x{self.h}@{self.fps} depth_scale={self.depth_scale:.8f} imu={self.imu_enabled}")

    def read(self) -> Optional[FramePacket]:
        frames = self.pipeline.wait_for_frames()
        self._update_pitch_from_frames(frames)
        if self.align is not None:
            frames = self.align.process(frames)
        c = frames.get_color_frame(); d = frames.get_depth_frame()
        if not c:
            return None
        rgb = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale if d else None
        # 실 카메라는 HW 타임스탬프 권장 (ms→s). 없으면 now.
        ts = c.get_timestamp() / 1000.0 if hasattr(c, "get_timestamp") else time.time()
        self._fid += 1
        if self._imu_has_measurement:
            pitch_source = self._pitch_source
        elif self.use_imu_pitch:
            pitch_source = "wait_imu"
        else:
            pitch_source = "manual"
        return FramePacket(rgb, depth, ts, self._fid, intrinsics=self.intrinsics,
                           imu_pitch_deg=self.pitch_down_deg, pitch_source=pitch_source)
    def stop(self):
        if self.pipeline:
            self.pipeline.stop()


class MuJoCoCameraSource(CameraSource):
    """MuJoCo scene 에서 로봇 카메라 RGB+depth 렌더. scene xml/mesh 가 있어야 함."""
    def __init__(self, xml_path: str, camera_name: str = "realsense", width=640, height=480):
        self.xml_path = xml_path
        self.camera_name = camera_name
        self.w, self.h, self._fid = width, height, 0
        self.mujoco = None; self.model = None; self.data = None
        self.renderer = None; self.renderer_depth = None
    def start(self):
        self.mujoco = importlib.import_module("mujoco")
        self.model = self.mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = self.mujoco.MjData(self.model)
        self.mujoco.mj_forward(self.model, self.data)
        if self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name) < 0:
            raise RuntimeError(f"MuJoCo camera '{self.camera_name}' 못 찾음. --camera 로 scene 카메라 이름을 줘라.")
        self.renderer = self.mujoco.Renderer(self.model, height=self.h, width=self.w)
        self.renderer_depth = self.mujoco.Renderer(self.model, height=self.h, width=self.w)
        self.renderer_depth.enable_depth_rendering()
        print(f"[MuJoCo] xml={self.xml_path} cam={self.camera_name}")
    def read(self) -> Optional[FramePacket]:
        # 물리 step 은 driver 가 담당 (qpos teleport). 여기선 렌더만.
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render().copy()              # RGB
        self.renderer_depth.update_scene(self.data, camera=self.camera_name)
        depth = self.renderer_depth.render().copy()       # meters
        # vision_core/onnx 는 BGR 가정 → 변환은 app 에서 cv2 로. 여기선 RGB 그대로 두되 flag 없음.
        self._fid += 1
        return FramePacket(rgb, depth, time.time(), self._fid)
    def get_sim_handles(self):
        return (self.mujoco, self.model, self.data)
    def stop(self):
        pass


class ThreadedCameraSource(CameraSource):
    """내부 소스를 백그라운드 스레드로 읽어 최신 프레임만 보관 (처리 루프가 안 막힘)."""
    def __init__(self, inner: CameraSource):
        self.inner = inner
        self._latest: Optional[FramePacket] = None
        self._lock = threading.Lock()
        self._run = False
        self._t: Optional[threading.Thread] = None
    def start(self):
        self.inner.start()
        self._run = True
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
    def _worker(self):
        while self._run:
            try:
                pkt = self.inner.read()
                if pkt is not None:
                    with self._lock:
                        self._latest = pkt
            except Exception as e:
                print(f"[ThreadedCameraSource] {e}")
                time.sleep(0.05)
    def read(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest
    def get_sim_handles(self):
        return self.inner.get_sim_handles()
    def stop(self):
        self._run = False
        self.inner.stop()


def create_source(cfg, kind: str, xml_path: Optional[str] = None) -> CameraSource:
    kind = kind.lower()
    if kind == "dummy":
        return DummySource(cfg.img_width, cfg.img_height)
    if kind == "realsense":
        return RealSenseD435Source(
            cfg.img_width, cfg.img_height, getattr(cfg, "realsense_fps", 30),
            serial=getattr(cfg, "realsense_serial", ""),
            use_imu_pitch=getattr(cfg, "use_imu_pitch", False),
            use_sysfs_imu_pitch=getattr(cfg, "use_sysfs_imu_pitch", True),
            pitch_down_deg=getattr(cfg, "camera_pitch_down_deg", 20.0),
            imu_pitch_alpha=getattr(cfg, "imu_pitch_alpha", 0.15),
            imu_pitch_offset_deg=getattr(cfg, "imu_pitch_offset_deg", 0.0),
            imu_accel_fps=getattr(cfg, "imu_accel_fps", 0),
            emitter=getattr(cfg, "realsense_emitter", -1.0),
            laser_power=getattr(cfg, "realsense_laser_power", -1.0),
        )
    if kind == "mujoco":
        if not xml_path:
            raise ValueError("mujoco source 는 --xml 경로 필요")
        return MuJoCoCameraSource(xml_path, cfg.sim_camera_name, cfg.img_width, cfg.img_height)
    raise ValueError(f"unknown source kind: {kind}")
