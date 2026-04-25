#!/usr/bin/env python3
"""Run a small DRTC observation transport sweep and write CSV results."""

from __future__ import annotations

import argparse
import csv
import pickle  # nosec B403: benchmark-compatible payload format.
import socket
import time
from pathlib import Path

import grpc
import numpy as np

from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.utils.compression import encode_images_for_transport
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def _parse_csv_qualities(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"none", "raw"}:
            values.append(None)
        else:
            value = int(item)
            if not (1 <= value <= 100):
                raise argparse.ArgumentTypeError("JPEG qualities must be in [1, 100]")
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one quality")
    return values


def _parse_size(raw: str) -> tuple[int, int]:
    width_raw, height_raw = raw.lower().split("x", 1)
    width = int(width_raw)
    height = int(height_raw)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image width and height must be positive")
    return width, height


def _tcp_connect(host: str, port: int, timeout_s: float) -> float:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    start = time.perf_counter()
    try:
        sock.connect((host, port))
    finally:
        sock.close()
    return time.perf_counter() - start


def _make_image(width: int, height: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "zeros":
        return np.zeros((height, width, 3), dtype=np.uint8)
    if mode == "random":
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    if mode == "gradient":
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx, yy, ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    raise ValueError(f"Unsupported image mode: {mode}")


def _make_timed_observation(
    *,
    control_step: int,
    state_dim: int,
    image_size: tuple[int, int],
    image_count: int,
    image_mode: str,
    jpeg_quality: int | None,
    rng: np.random.Generator,
) -> tuple[TimedObservation, int, int | None]:
    width, height = image_size
    observation: dict[str, object] = {f"state_{i}": 0.0 for i in range(state_dim)}
    for i in range(image_count):
        key = "image" if i == 0 else f"image_{i}"
        observation[key] = _make_image(width, height, image_mode, rng)
    observation["task"] = "pick up the orange cube"

    raw_image_bytes = sum(value.nbytes for value in observation.values() if isinstance(value, np.ndarray))
    encoded_image_bytes = None
    if jpeg_quality is not None:
        observation, stats = encode_images_for_transport(observation, jpeg_quality=jpeg_quality)
        encoded_image_bytes = stats["encoded_bytes_total"]

    timed = TimedObservation(
        timestamp=time.time(),
        control_step=control_step,
        observation=observation,
        chunk_start_step=control_step,
    )
    return timed, raw_image_bytes, encoded_image_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", default="results/drtc_transport_sweep.csv")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--image-size", type=_parse_size, default=(224, 224), metavar="WIDTHxHEIGHT")
    parser.add_argument("--image-counts", type=_parse_csv_ints, default=[0, 1, 2])
    parser.add_argument("--jpeg-qualities", type=_parse_csv_qualities, default=[None, 60, 40, 30])
    parser.add_argument("--image-modes", default="zeros,random")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.image_modes.split(",") if mode.strip()]
    invalid_modes = sorted(set(modes) - {"zeros", "random", "gradient"})
    if invalid_modes:
        raise SystemExit(f"Invalid --image-modes: {invalid_modes}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    connect_s = _tcp_connect(args.host, args.port, args.connect_timeout_s)
    print(f"tcp_connect_ms={connect_s * 1000:.1f} target={args.host}:{args.port}")

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        grpc_channel_options(max_receive_message_length=64 * 1024 * 1024, max_send_message_length=64 * 1024 * 1024),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    rng = np.random.default_rng(0)

    fieldnames = [
        "host",
        "port",
        "tcp_connect_ms",
        "image_mode",
        "image_count",
        "jpeg_quality",
        "repeat",
        "raw_image_bytes",
        "encoded_image_bytes",
        "payload_bytes",
        "send_ms",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        control_step = 0
        for image_mode in modes:
            for image_count in args.image_counts:
                for jpeg_quality in args.jpeg_qualities:
                    if image_count == 0 and jpeg_quality is not None:
                        continue
                    for repeat in range(args.repeats):
                        timed, raw_bytes, encoded_bytes = _make_timed_observation(
                            control_step=control_step,
                            state_dim=args.state_dim,
                            image_size=args.image_size,
                            image_count=image_count,
                            image_mode=image_mode,
                            jpeg_quality=jpeg_quality,
                            rng=rng,
                        )
                        payload = pickle.dumps(timed)
                        start = time.perf_counter()
                        stub.SendObservations(send_bytes_in_chunks(payload, services_pb2.Observation, silent=True))
                        send_ms = (time.perf_counter() - start) * 1000
                        row = {
                            "host": args.host,
                            "port": args.port,
                            "tcp_connect_ms": round(connect_s * 1000, 3),
                            "image_mode": image_mode,
                            "image_count": image_count,
                            "jpeg_quality": jpeg_quality if jpeg_quality is not None else "none",
                            "repeat": repeat,
                            "raw_image_bytes": raw_bytes,
                            "encoded_image_bytes": encoded_bytes if encoded_bytes is not None else "",
                            "payload_bytes": len(payload),
                            "send_ms": round(send_ms, 3),
                        }
                        writer.writerow(row)
                        print(
                            f"mode={image_mode} count={image_count} jpeg={row['jpeg_quality']} "
                            f"repeat={repeat} payload={len(payload)} send_ms={send_ms:.1f}"
                        )
                        control_step += 1

    channel.close()
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
