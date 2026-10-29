"""Inspect or replay an exact PICO-manager ZMQ packet capture."""

from __future__ import annotations

import argparse
from collections import Counter
import time

import zmq

from gear_sonic.utils.teleop.zmq.packet_capture import iter_capture, iter_input_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    parser.add_argument(
        "--publish-endpoint",
        help="Bind a PUB socket here and replay exact packets; omit for read-only validation.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    records = list(iter_capture(args.capture_dir, verify=True))
    if not records:
        raise SystemExit("capture contains no packets")

    modes = Counter(entry["mode"] for entry, _ in records)
    topics = Counter(entry.get("topic", "unknown") for entry, _ in records)
    raw_count = sum("raw_body_pose_offset" in entry for entry, _ in records)
    input_records = list(iter_input_samples(args.capture_dir))
    pressed_samples = sum(
        any(entry["controls"][key] for key in ("a", "b", "x", "y"))
        for entry, _ in input_records
    )
    duration_s = (records[-1][0]["monotonic_ns"] - records[0][0]["monotonic_ns"]) * 1e-9
    print(
        f"validated {len(records)} packets over {duration_s:.3f}s; "
        f"modes={dict(modes)}; topics={dict(topics)}; raw_body_pose_packets={raw_count}; "
        f"input_samples={len(input_records)}; pressed_input_samples={pressed_samples}"
    )
    if not args.publish_endpoint:
        return
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(args.publish_endpoint)
    time.sleep(0.2)
    try:
        previous_ns = records[0][0]["monotonic_ns"]
        for entry, payload in records:
            delay_s = (entry["monotonic_ns"] - previous_ns) * 1e-9 / args.speed
            if delay_s > 0:
                time.sleep(delay_s)
            socket.send(payload)
            previous_ns = entry["monotonic_ns"]
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
