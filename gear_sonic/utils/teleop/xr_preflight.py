"""Live, input-only XR tracking readiness checks for headset workflows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


def knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """Return the interior knee angle, or NaN for degenerate tracking geometry."""
    upper = np.asarray(hip, dtype=float) - np.asarray(knee, dtype=float)
    lower = np.asarray(ankle, dtype=float) - np.asarray(knee, dtype=float)
    denominator = float(np.linalg.norm(upper) * np.linalg.norm(lower))
    if denominator <= 1e-12:
        return math.nan
    cosine = float(np.dot(upper, lower)) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class PreflightStatus:
    ready: bool
    message: str
    message_zh: str
    left_knee_deg: float | None = None
    right_knee_deg: float | None = None
    max_device_gap_ms: float | None = None
    pose_component_range: float | None = None


class XRPreflightMonitor:
    """Maintain a short live window and describe tracking readiness for speech."""

    def __init__(
        self,
        *,
        min_duration_s: float = 5.0,
        max_device_gap_ms: float = 100.0,
        min_knee_angle_deg: float = 150.0,
        min_pose_component_range: float = 1e-4,
        disconnect_after_s: float = 1.0,
        clock=time.monotonic,
    ) -> None:
        self.min_duration_s = min_duration_s
        self.max_device_gap_ms = max_device_gap_ms
        self.min_knee_angle_deg = min_knee_angle_deg
        self.min_pose_component_range = min_pose_component_range
        self.disconnect_after_s = disconnect_after_s
        self._clock = clock
        self._samples: deque[tuple[int, np.ndarray, float, float]] = deque()
        self._last_device_timestamp_ns: int | None = None
        self._last_receive_monotonic: float | None = None

    def update(
        self,
        body_poses: np.ndarray,
        *,
        device_timestamp_ns: int,
        receive_monotonic: float | None = None,
    ) -> None:
        """Add a new device frame. Repeated reader snapshots are ignored."""
        if device_timestamp_ns <= 0 or device_timestamp_ns == self._last_device_timestamp_ns:
            return
        now = self._clock() if receive_monotonic is None else receive_monotonic
        array = np.asarray(body_poses, dtype=float)
        self._last_device_timestamp_ns = device_timestamp_ns
        self._last_receive_monotonic = now

        if array.shape != (24, 7) or not np.isfinite(array).all():
            self._samples.clear()
            return

        left_knee = knee_angle_deg(array[1, :3], array[4, :3], array[7, :3])
        right_knee = knee_angle_deg(array[2, :3], array[5, :3], array[8, :3])
        self._samples.append((device_timestamp_ns, array.copy(), left_knee, right_knee))

        keep_after_ns = device_timestamp_ns - int((self.min_duration_s + 2.0) * 1e9)
        while len(self._samples) > 2 and self._samples[1][0] < keep_after_ns:
            self._samples.popleft()

    @staticmethod
    def _stance_message(left_knee: float, right_knee: float) -> str:
        if not math.isfinite(left_knee) or not math.isfinite(right_knee):
            return "PICO tracking reports invalid body geometry"
        if left_knee >= 150.0 and right_knee >= 150.0:
            return "PICO tracking reports an upright stance"
        if left_knee < 90.0 and right_knee < 90.0:
            return "PICO tracking reports both legs deeply bent or retracted"
        if left_knee < 150.0 and right_knee < 150.0:
            return "PICO tracking reports both knees bent"
        if left_knee < 150.0:
            return "PICO tracking reports the left knee bent"
        return "PICO tracking reports the right knee bent"

    def status(self, *, now_monotonic: float | None = None) -> PreflightStatus:
        now = self._clock() if now_monotonic is None else now_monotonic
        if self._last_receive_monotonic is None or not self._samples:
            return PreflightStatus(False, "Waiting for PICO body tracking", "正在等待PICO全身追踪")
        if now - self._last_receive_monotonic > self.disconnect_after_s:
            return PreflightStatus(
                False,
                "PICO body tracking is disconnected or stalled",
                "PICO全身追踪已断开或停止更新",
            )

        device_ns = np.asarray([sample[0] for sample in self._samples], dtype=np.int64)
        arrays = np.stack([sample[1] for sample in self._samples])
        left_knees = np.asarray([sample[2] for sample in self._samples], dtype=float)
        right_knees = np.asarray([sample[3] for sample in self._samples], dtype=float)
        duration_s = float((device_ns[-1] - device_ns[0]) / 1e9)
        gaps_ms = np.diff(device_ns) / 1e6
        max_gap_ms = float(gaps_ms.max()) if gaps_ms.size else 0.0
        pose_range = float(np.max(np.ptp(arrays, axis=0)))
        left = float(left_knees[-1])
        right = float(right_knees[-1])
        stance = self._stance_message(left, right)

        common = dict(
            left_knee_deg=left,
            right_knee_deg=right,
            max_device_gap_ms=max_gap_ms,
            pose_component_range=pose_range,
        )
        if duration_s < self.min_duration_s:
            return PreflightStatus(
                False,
                f"{stance}. Collecting calibration evidence",
                "正在收集校准数据，请保持姿势并缓慢弯曲再伸直双肘",
                **common,
            )
        if max_gap_ms > self.max_device_gap_ms:
            return PreflightStatus(False, f"{stance}. Tracking is intermittent", "追踪不连续", **common)
        if pose_range < self.min_pose_component_range:
            return PreflightStatus(
                False,
                (
                    f"{stance}. Tracking appears frozen. If your physical legs are straight, "
                    "recalibrate the ankle trackers, then slowly bend and restore both elbows"
                ),
                "追踪画面可能冻结，请缓慢弯曲再伸直双肘",
                **common,
            )
        if (
            not np.isfinite(left_knees).all()
            or not np.isfinite(right_knees).all()
            or float(left_knees.min()) < self.min_knee_angle_deg
            or float(right_knees.min()) < self.min_knee_angle_deg
        ):
            return PreflightStatus(
                False,
                (
                    f"{stance}. If that does not match your physical stance, recalibrate the "
                    "ankle trackers before arming"
                ),
                "站姿不正确，请重新校准后再解锁",
                **common,
            )
        return PreflightStatus(
            True,
            (
                "PICO tracking reports an upright stance. Tracking is fresh and moving. "
                "Press A and B together to arm"
            ),
            "站姿直立，追踪新鲜且身体数据正在变化。请同时按下A键和B键解锁",
            **common,
        )
