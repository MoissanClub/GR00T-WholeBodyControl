"""Validate continuous raw XR/body/controller capture before robot actuation."""

from __future__ import annotations

import argparse
import math

import numpy as np

from gear_sonic.utils.teleop.zmq.packet_capture import iter_input_samples


def knee_angle_deg(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    upper = hip - knee
    lower = ankle - knee
    cosine = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--max-device-gap-ms", type=float, default=100.0)
    parser.add_argument("--min-knee-angle-deg", type=float, default=150.0)
    parser.add_argument("--min-pose-component-range", type=float, default=1e-4)
    args = parser.parse_args()

    records = list(iter_input_samples(args.capture_dir))
    if len(records) < 2:
        raise SystemExit("insufficient continuous XR input samples")
    entries = [entry for entry, _ in records]
    arrays = [array for _, array in records]
    if any(array.shape != (24, 7) or not np.isfinite(array).all() for array in arrays):
        raise SystemExit("raw XR body pose is not finite 24x7 data")

    device_ns = np.asarray([entry["device_timestamp_ns"] for entry in entries], dtype=np.int64)
    gaps_ms = np.diff(device_ns) / 1e6
    duration_s = (device_ns[-1] - device_ns[0]) / 1e9
    left_knees = [knee_angle_deg(a[1, :3], a[4, :3], a[7, :3]) for a in arrays]
    right_knees = [knee_angle_deg(a[2, :3], a[5, :3], a[8, :3]) for a in arrays]
    pressed = sum(
        any(entry["controls"].get(key, False) for key in ("a", "b", "x", "y"))
        for entry in entries
    )
    pose_component_range = float(np.max(np.ptp(np.stack(arrays), axis=0)))
    print(
        f"validated {len(records)} continuous XR samples over {duration_s:.3f}s; "
        f"max_device_gap_ms={gaps_ms.max():.3f}; "
        f"left_knee_deg={min(left_knees):.2f}..{max(left_knees):.2f}; "
        f"right_knee_deg={min(right_knees):.2f}..{max(right_knees):.2f}; "
        f"pose_component_range={pose_component_range:.6g}; pressed_samples={pressed}"
    )
    if not math.isfinite(duration_s) or duration_s < args.min_duration_s:
        raise SystemExit(f"capture duration {duration_s:.3f}s is below required minimum")
    if gaps_ms.max() > args.max_device_gap_ms:
        raise SystemExit(f"device gap {gaps_ms.max():.3f}ms exceeds limit")
    if min(left_knees) < args.min_knee_angle_deg or min(right_knees) < args.min_knee_angle_deg:
        raise SystemExit("tracked knee geometry is not upright")
    if pose_component_range < args.min_pose_component_range:
        raise SystemExit("body-pose content is frozen or did not pass the required motion check")


if __name__ == "__main__":
    main()
