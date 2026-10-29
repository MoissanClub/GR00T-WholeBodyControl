import json

import numpy as np

from gear_sonic.utils.teleop.zmq.packet_capture import (
    RecordingSocket,
    iter_capture,
    iter_input_samples,
    read_raw_body_pose,
)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, payload, *args, **kwargs):
        self.sent.append(bytes(payload))
        return "sent"

    def close(self, *args, **kwargs):
        self.closed = True


def test_exact_packet_and_raw_pose_round_trip(tmp_path):
    fake = FakeSocket()
    socket = RecordingSocket(fake, tmp_path)
    raw_pose = np.arange(24 * 7, dtype=np.float64).reshape(24, 7)
    socket.record_input_sample(
        raw_pose,
        device_timestamp_ns=100,
        sample_realtime_s=4.0,
        sample_monotonic_s=6.0,
        controls={"a": False, "b": False, "x": False, "y": False},
    )
    socket.record_input_sample(
        raw_pose,
        device_timestamp_ns=100,
        sample_realtime_s=4.0,
        sample_monotonic_s=6.0,
        controls={"a": True, "b": True, "x": True, "y": True},
    )
    socket.set_mode("PLANNER")
    assert socket.send(b"planner-packet") == "sent"
    socket.send(b"manager_state-ignored")

    socket.set_mode("POSE")
    socket.set_pose_context(
        raw_pose,
        device_timestamp_ns=123,
        sample_realtime_s=4.5,
        sample_monotonic_s=6.7,
    )
    socket.send(b"pose-packet")
    socket.close()

    records = list(iter_capture(tmp_path))
    assert [payload for _, payload in records] == [b"planner-packet", b"pose-packet"]
    assert [entry["mode"] for entry, _ in records] == ["PLANNER", "POSE"]
    assert [entry["topic"] for entry, _ in records] == ["planner", "pose"]
    assert read_raw_body_pose(tmp_path, records[0][0]) is None
    np.testing.assert_array_equal(read_raw_body_pose(tmp_path, records[1][0]), raw_pose)
    input_records = list(iter_input_samples(tmp_path))
    assert len(input_records) == 1
    np.testing.assert_array_equal(input_records[0][1], raw_pose)
    assert input_records[0][0]["controls"]["a"] is False
    assert json.loads((tmp_path / "metadata.json").read_text())["format"] == (
        "sonic-zmq-packet-capture-v1"
    )
