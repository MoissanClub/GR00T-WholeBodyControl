import numpy as np

from gear_sonic.scripts.inspect_xr_input_capture import knee_angle_deg


def test_knee_angle_straight_and_right_angle():
    knee = np.array([0.0, 0.0, 0.0])
    assert knee_angle_deg(np.array([0.0, 1.0, 0.0]), knee, np.array([0.0, -1.0, 0.0])) == 180.0
    assert knee_angle_deg(np.array([0.0, 1.0, 0.0]), knee, np.array([1.0, 0.0, 0.0])) == 90.0
