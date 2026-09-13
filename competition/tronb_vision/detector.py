from __future__ import annotations
"""
detector.py — 검출기 추상화 (기준: v8.2 OnnxYoloDetector).

핵심:
- BaseDetector.predict_boxes(rgb) -> DetectionBoxes  (box.cls/.conf/.xywh/.xyxy 제공)
  → vision_core 가 ultralytics/onnx 구분 없이 같은 코드로 처리.
- 클래스 id는 '이름' 으로 해석 (resolve_class_ids). 모델마다 학습 순서가 달라도 동작.
- DummyDetector: mujoco/onnx 없이 selftest/headless 실행용.
"""

import importlib
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import (BALL_ALIASES, BAR_ALIASES, GOAL_ALIASES, LINE_L_ALIASES,
                     LINE_T_ALIASES, LINE_X_ALIASES, POINT_ALIASES, ROBOT_ALIASES,
                     normalize_class_name)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: Tuple[float, float, float, float]

    def xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


class _Shim:
    """ultralytics box.cls.cpu().numpy() 패턴을 흉내내는 어댑터."""
    def __init__(self, values):
        self._v = np.asarray(values, dtype=float)
    def cpu(self): return self
    def numpy(self): return self._v
    def __getitem__(self, i): return self._v[i]


class DetectionBox:
    """vision_core 가 기대하는 box 인터페이스 (.cls/.conf/.xywh/.xyxy)."""
    def __init__(self, det: Detection):
        self._d = det
        self.cls = _Shim([det.cls_id])
        self.conf = _Shim([det.conf])
        self.xywh = _Shim([list(det.xywh())])
        self.xyxy = _Shim([list(det.xyxy)])
        self.cls_name = det.cls_name


class DetectionBoxes(list):
    @classmethod
    def from_detections(cls, dets: List[Detection]) -> "DetectionBoxes":
        return cls(DetectionBox(d) for d in dets)


# 클래스 이름 해석
def _ids_by_aliases(names: dict, aliases: Sequence[str]) -> Set[int]:
    norm = {normalize_class_name(a) for a in aliases}
    out: Set[int] = set()
    for k, v in (names or {}).items():
        if normalize_class_name(v) in norm:
            out.add(int(k))
    return out


def resolve_class_ids(names: dict) -> dict:
    """모델 names({id:name}) 에서 의미별 id 집합을 이름으로 해석."""
    ball = _ids_by_aliases(names, BALL_ALIASES) or ({32} if 32 in (names or {}) else set())
    goal = _ids_by_aliases(names, GOAL_ALIASES) | _ids_by_aliases(names, BAR_ALIASES)
    return {
        "ball": ball,
        "goal": goal,
        "robot": _ids_by_aliases(names, ROBOT_ALIASES),
        "line_l": _ids_by_aliases(names, LINE_L_ALIASES),
        "line_t": _ids_by_aliases(names, LINE_T_ALIASES),
        "line_x": _ids_by_aliases(names, LINE_X_ALIASES),
        "point": _ids_by_aliases(names, POINT_ALIASES),
    }


# letterbox / NMS
def _letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    import cv2  # 이 경로는 onnx 검출 시에만
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_x, pad_y = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: List[Detection], iou_thres: float) -> List[Detection]:
    out: List[Detection] = []
    for cid in {d.cls_id for d in dets}:
        group = sorted([d for d in dets if d.cls_id == cid], key=lambda d: d.conf, reverse=True)
        keep: List[Detection] = []
        while group:
            best = group.pop(0)
            keep.append(best)
            group = [d for d in group if _iou(best.xyxy, d.xyxy) < iou_thres]
        out.extend(keep)
    return out


# 검출기 구현
class BaseDetector:
    names: dict = {}
    last_infer_ms: float = 0.0
    def predict(self, rgb: np.ndarray) -> List[Detection]:
        raise NotImplementedError
    def predict_boxes(self, rgb: np.ndarray) -> DetectionBoxes:
        return DetectionBoxes.from_detections(self.predict(rgb))


class DummyDetector(BaseDetector):
    """mujoco/onnx 없이 도는 합성 검출기. selftest/headless 전용.
    선택적으로 scripted 박스 시퀀스를 주입해 FSM 전이를 결정적으로 테스트."""
    def __init__(self, class_names: Sequence[str] = (), scripted=None):
        self.names = {i: n for i, n in enumerate(class_names)}
        self._scripted = list(scripted) if scripted else None
        self._i = 0
    def predict(self, rgb: np.ndarray) -> List[Detection]:
        if self._scripted is not None:
            dets = self._scripted[min(self._i, len(self._scripted) - 1)]
            self._i += 1
            return list(dets)
        return []


class OnnxYoloDetector(BaseDetector):
    """YOLO ONNX (v8.2 구현 기준). v8/v5 raw 및 NMS-포함 출력 모두 파싱."""
    def __init__(self, model_path: str, class_names: Sequence[str], imgsz: int = 640,
                 conf_thres: float = 0.35, iou_thres: float = 0.45,
                 providers: Optional[List[str]] = None):
        ort = importlib.import_module("onnxruntime")
        if providers is None:
            avail = ort.get_available_providers()
            preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in avail] or avail
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # 클래스명: onnx 메타데이터(names)를 최우선 사용
        # Ultralytics export 는 custom_metadata_map['names'] 에 {id:name} 을 저장한다.
        # 이걸 무시하고 cfg.class_names 를 쓰면 모델과 어긋나 라벨/obs_type 이 통째로 밀린다.
        meta_names = self._read_meta_names()
        if meta_names:
            self._names_list = [meta_names[i] for i in sorted(meta_names)]
            self.names = dict(meta_names)
            if list(class_names) and list(class_names) != self._names_list:
                print(f"[OnnxYoloDetector] ⚠ cfg.class_names 무시하고 onnx 내장 names 사용: {self._names_list}")
        else:
            self._names_list = list(class_names)
            self.names = {i: n for i, n in enumerate(self._names_list)}
        self.imgsz, self.conf_thres, self.iou_thres = int(imgsz), float(conf_thres), float(iou_thres)
        self.providers = providers
        self.last_infer_ms = 0.0
        self._warned = False
        print(f"[OnnxYoloDetector] providers={providers} names={self.names}")

    def _read_meta_names(self):
        """onnx 메타데이터에서 {id:name} 파싱. 없으면 None."""
        try:
            import ast
            m = self.session.get_modelmeta().custom_metadata_map
            raw = m.get("names")
            if not raw:
                return None
            d = ast.literal_eval(raw)
            return {int(k): str(v) for k, v in d.items()} if isinstance(d, dict) else None
        except Exception:
            return None

    def _name(self, cid: int) -> str:
        return self._names_list[cid] if 0 <= cid < len(self._names_list) else str(cid)

    def _warn(self, m):
        if not self._warned:
            print(f"[WARN][onnx] {m}"); self._warned = True

    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        import cv2
        h0, w0 = frame_bgr.shape[:2]
        img, r, pad_x, pad_y = _letterbox(frame_bgr, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        t0 = time.time()
        out = self.session.run(None, {self.input_name: x})[0]
        self.last_infer_ms = (time.time() - t0) * 1000.0
        pred = np.squeeze(out)
        if pred.ndim != 2:
            self._warn(f"bad shape {out.shape}"); return []
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= 256:
            pred = pred.T
        if pred.shape[1] < 6:
            self._warn(f"last dim<6 {out.shape}"); return []
        cols, nc = pred.shape[1], len(self._names_list)
        dets: List[Detection] = []

        def _unpad(x1, y1, x2, y2):
            x1 = np.clip((x1 - pad_x) / r, 0, w0 - 1); x2 = np.clip((x2 - pad_x) / r, 0, w0 - 1)
            y1 = np.clip((y1 - pad_y) / r, 0, h0 - 1); y2 = np.clip((y2 - pad_y) / r, 0, h0 - 1)
            return float(x1), float(y1), float(x2), float(y2)

        if cols == 6:  # NMS 포함 / 일반 6열
            for row in pred:
                x1, y1, x2, y2, conf, cid = row.astype(float)
                if conf < self.conf_thres:
                    continue
                if x2 <= x1 or y2 <= y1:  # xywh 로 보고 변환
                    cx, cy, bw, bh = x1, y1, x2, y2
                    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                dets.append(Detection(int(cid), self._name(int(cid)), float(conf), _unpad(x1, y1, x2, y2)))
            return _nms(dets, self.iou_thres)

        if cols == 4 + nc:
            boxes, scores = pred[:, :4], pred[:, 4:]
        elif cols == 5 + nc:
            boxes, scores = pred[:, :4], pred[:, 5:] * pred[:, 4:5]
        else:
            self._warn(f"class dim {cols-4} != {nc}; raw 4+C parse")
            boxes, scores = pred[:, :4], pred[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        keep = confs >= self.conf_thres
        for box, cid, conf in zip(boxes[keep], cls_ids[keep], confs[keep]):
            cx, cy, bw, bh = box.astype(float)
            dets.append(Detection(int(cid), self._name(int(cid)), float(conf),
                                  _unpad(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
        return _nms(dets, self.iou_thres)


class UltralyticsDetector(BaseDetector):
    """PT 모델 폴백 (ultralytics 설치 시)."""
    def __init__(self, model_path: str, conf_thres: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = dict(getattr(self.model, "names", {}) or {})
        self.conf_thres = float(conf_thres)
        self.last_infer_ms = 0.0
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(frame_bgr, conf=self.conf_thres, verbose=False)[0]
        self.last_infer_ms = float(sum(res.speed.values())) if getattr(res, "speed", None) else 0.0
        dets = []
        for b in res.boxes:
            cid = int(b.cls.cpu().numpy()[0]); conf = float(b.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0].tolist()
            dets.append(Detection(cid, self.names.get(cid, str(cid)), conf, (x1, y1, x2, y2)))
        return dets


def create_detector(cfg) -> BaseDetector:
    backend = getattr(cfg, "detector_backend", "onnx").lower()
    if backend == "dummy":
        return DummyDetector(cfg.class_names)
    if backend in ("ultralytics", "pt"):   # 'pt' = v7 config 표기 호환
        return UltralyticsDetector(cfg.model_path, cfg.conf_thres)
    return OnnxYoloDetector(cfg.model_path, cfg.class_names, cfg.imgsz, cfg.conf_thres, cfg.iou_thres)
