"""Replay-grade capture for exact ZMQ manager output and aligned XR body poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np


class RecordingSocket:
    """Proxy a ZMQ socket while preserving every successfully sent payload."""

    REPLAY_TOPICS = ("command", "planner", "pose")

    def __init__(self, socket: Any, record_dir: str | Path):
        self._socket = socket
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._packet_file = (self.record_dir / "packets.bin").open("wb")
        self._raw_file = (self.record_dir / "raw_body_poses.bin").open("wb")
        self._index_file = (self.record_dir / "packets.jsonl").open("w", encoding="utf-8")
        self._input_body_file = (self.record_dir / "input_body_poses.bin").open("wb")
        self._input_index_file = (self.record_dir / "input_samples.jsonl").open(
            "w", encoding="utf-8"
        )
        self._sequence = 0
        self._input_sequence = 0
        self._last_input_device_timestamp_ns: int | None = None
        self._mode = "OFF"
        self._pose_context: dict[str, Any] | None = None
        metadata = {
            "format": "sonic-zmq-packet-capture-v1",
            "packet_file": "packets.bin",
            "index_file": "packets.jsonl",
            "raw_body_pose_file": "raw_body_poses.bin",
            "input_body_pose_file": "input_body_poses.bin",
            "input_index_file": "input_samples.jsonl",
            "created_wall_ns": time.time_ns(),
        }
        (self.record_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._socket, name)

    def set_mode(self, mode: str) -> None:
        mode = str(mode)
        if mode == self._mode:
            return
        self.flush()
        self._mode = mode

    def set_pose_context(
        self,
        body_poses: np.ndarray,
        *,
        device_timestamp_ns: int,
        sample_realtime_s: float,
        sample_monotonic_s: float,
    ) -> None:
        raw = np.ascontiguousarray(body_poses)
        self._pose_context = {
            "array": raw,
            "device_timestamp_ns": int(device_timestamp_ns),
            "sample_realtime_s": float(sample_realtime_s),
            "sample_monotonic_s": float(sample_monotonic_s),
        }

    def record_input_sample(
        self,
        body_poses: np.ndarray,
        *,
        device_timestamp_ns: int,
        sample_realtime_s: float,
        sample_monotonic_s: float,
        controls: dict[str, Any],
    ) -> None:
        """Record each fresh XR input frame once, independent of stream mode."""

        device_timestamp_ns = int(device_timestamp_ns)
        if device_timestamp_ns == self._last_input_device_timestamp_ns:
            return
        raw = np.ascontiguousarray(body_poses)
        raw_bytes = raw.tobytes(order="C")
        entry = {
            "sequence": self._input_sequence,
            "mode": self._mode,
            "record_wall_ns": time.time_ns(),
            "record_monotonic_ns": time.monotonic_ns(),
            "device_timestamp_ns": device_timestamp_ns,
            "sample_realtime_s": float(sample_realtime_s),
            "sample_monotonic_s": float(sample_monotonic_s),
            "body_pose_offset": self._input_body_file.tell(),
            "body_pose_size": len(raw_bytes),
            "body_pose_dtype": raw.dtype.str,
            "body_pose_shape": list(raw.shape),
            "controls": controls,
        }
        self._input_body_file.write(raw_bytes)
        self._input_index_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._input_sequence += 1
        self._last_input_device_timestamp_ns = device_timestamp_ns
        self.flush()

    def send(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        payload_bytes = bytes(payload)
        result = self._socket.send(payload, *args, **kwargs)
        topic = next(
            (name for name in self.REPLAY_TOPICS if payload_bytes.startswith(name.encode())),
            None,
        )
        if topic is None:
            self._pose_context = None
            return result
        wall_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        packet_offset = self._packet_file.tell()
        self._packet_file.write(payload_bytes)

        entry: dict[str, Any] = {
            "sequence": self._sequence,
            "topic": topic,
            "mode": self._mode,
            "wall_ns": wall_ns,
            "monotonic_ns": monotonic_ns,
            "packet_offset": packet_offset,
            "packet_size": len(payload_bytes),
            "packet_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        if self._pose_context is not None:
            context = self._pose_context
            raw = context["array"]
            raw_bytes = raw.tobytes(order="C")
            entry.update(
                {
                    "raw_body_pose_offset": self._raw_file.tell(),
                    "raw_body_pose_size": len(raw_bytes),
                    "raw_body_pose_dtype": raw.dtype.str,
                    "raw_body_pose_shape": list(raw.shape),
                    "device_timestamp_ns": context["device_timestamp_ns"],
                    "sample_realtime_s": context["sample_realtime_s"],
                    "sample_monotonic_s": context["sample_monotonic_s"],
                }
            )
            self._raw_file.write(raw_bytes)
            self._pose_context = None

        self._index_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._sequence += 1
        self.flush()
        return result

    def flush(self) -> None:
        self._packet_file.flush()
        self._raw_file.flush()
        self._index_file.flush()
        self._input_body_file.flush()
        self._input_index_file.flush()

    def close(self, *args: Any, **kwargs: Any) -> Any:
        self.flush()
        self._packet_file.close()
        self._raw_file.close()
        self._index_file.close()
        self._input_body_file.close()
        self._input_index_file.close()
        return self._socket.close(*args, **kwargs)


def iter_capture(record_dir: str | Path, *, verify: bool = True) -> Iterator[tuple[dict, bytes]]:
    """Yield index entries and exact packet bytes from a capture directory."""

    root = Path(record_dir)
    with (root / "packets.bin").open("rb") as packet_file, (root / "packets.jsonl").open(
        encoding="utf-8"
    ) as index_file:
        for line in index_file:
            entry = json.loads(line)
            packet_file.seek(entry["packet_offset"])
            payload = packet_file.read(entry["packet_size"])
            if len(payload) != entry["packet_size"]:
                raise ValueError(f"truncated packet at sequence {entry['sequence']}")
            if verify and hashlib.sha256(payload).hexdigest() != entry["packet_sha256"]:
                raise ValueError(f"packet hash mismatch at sequence {entry['sequence']}")
            yield entry, payload


def read_raw_body_pose(record_dir: str | Path, entry: dict) -> np.ndarray | None:
    """Load the raw XR body-pose array aligned to one indexed POSE packet."""

    if "raw_body_pose_offset" not in entry:
        return None
    root = Path(record_dir)
    with (root / "raw_body_poses.bin").open("rb") as raw_file:
        raw_file.seek(entry["raw_body_pose_offset"])
        raw = raw_file.read(entry["raw_body_pose_size"])
    if len(raw) != entry["raw_body_pose_size"]:
        raise ValueError(f"truncated raw body pose at sequence {entry['sequence']}")
    return np.frombuffer(raw, dtype=np.dtype(entry["raw_body_pose_dtype"])).reshape(
        entry["raw_body_pose_shape"]
    )


def iter_input_samples(record_dir: str | Path) -> Iterator[tuple[dict, np.ndarray]]:
    """Yield continuous fresh XR frames and controller samples from all modes."""

    root = Path(record_dir)
    if not (root / "input_body_poses.bin").exists() or not (
        root / "input_samples.jsonl"
    ).exists():
        return
    with (root / "input_body_poses.bin").open("rb") as body_file, (
        root / "input_samples.jsonl"
    ).open(encoding="utf-8") as index_file:
        for line in index_file:
            entry = json.loads(line)
            body_file.seek(entry["body_pose_offset"])
            raw = body_file.read(entry["body_pose_size"])
            if len(raw) != entry["body_pose_size"]:
                raise ValueError(f"truncated input body pose at sequence {entry['sequence']}")
            array = np.frombuffer(raw, dtype=np.dtype(entry["body_pose_dtype"])).reshape(
                entry["body_pose_shape"]
            )
            yield entry, array
