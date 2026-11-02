import numpy as np

from gear_sonic.utils.teleop.xr_preflight import XRPreflightMonitor, knee_angle_deg


def test_knee_angle_straight_and_right_angle():
    knee = np.array([0.0, 0.0, 0.0])
    assert knee_angle_deg(np.array([0.0, 1.0, 0.0]), knee, np.array([0.0, -1.0, 0.0])) == 180.0
    assert knee_angle_deg(np.array([0.0, 1.0, 0.0]), knee, np.array([1.0, 0.0, 0.0])) == 90.0


def _upright_pose(elbow_offset: float = 0.0) -> np.ndarray:
    pose = np.zeros((24, 7), dtype=float)
    pose[:, 6] = 1.0
    pose[1, :3] = [-0.1, 1.0, 0.0]
    pose[4, :3] = [-0.1, 0.5, 0.0]
    pose[7, :3] = [-0.1, 0.0, 0.0]
    pose[2, :3] = [0.1, 1.0, 0.0]
    pose[5, :3] = [0.1, 0.5, 0.0]
    pose[8, :3] = [0.1, 0.0, 0.0]
    pose[18, 0] = elbow_offset
    return pose


def test_voice_preflight_requires_motion_then_reports_ready():
    monitor = XRPreflightMonitor(clock=lambda: 6.0)
    for index in range(61):
        monitor.update(
            _upright_pose(elbow_offset=index * 1e-5),
            device_timestamp_ns=index * 100_000_000 + 1,
            receive_monotonic=index * 0.1,
        )

    status = monitor.status(now_monotonic=6.0)
    assert status.ready
    assert "upright stance" in status.message


def test_voice_preflight_rejects_frozen_bent_tracking():
    monitor = XRPreflightMonitor(clock=lambda: 6.0)
    bent = _upright_pose()
    bent[7, :3] = [0.4, 0.5, 0.0]
    bent[8, :3] = [-0.2, 0.5, 0.0]
    for index in range(61):
        monitor.update(
            bent,
            device_timestamp_ns=index * 100_000_000 + 1,
            receive_monotonic=index * 0.1,
        )

    status = monitor.status(now_monotonic=6.0)
    assert not status.ready
    assert "frozen" in status.message.lower()
