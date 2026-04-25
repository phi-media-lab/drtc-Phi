#!/usr/bin/env python3
"""Benchmark DRTC client-to-server observation transport.

This script measures the network/serialization path for sending DRTC observations.
It does not require a policy to be loaded on the server. The server only needs to
be running and exposing the AsyncInference gRPC service.
"""

from __future__ import annotations

import argparse
import pickle  # nosec B403: internal benchmark payload format matches current DRTC transport.
import socket
import statistics
import time
from dataclasses import dataclass

import grpc
import numpy as np

from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.utils.compression import encode_images_for_transport
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


@dataclass(frozen=True)
class Sample:
    payload_bytes: int
    send_s: float


def _parse_size(raw: str) -> tuple[int, int]:
    try:
        width_raw, height_raw = raw.lower().split("x", 1)
        width = int(width_raw)
        height = int(height_raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got: {raw}") from exc
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


def _make_observation(
    *,
    control_step: int,
    state_dim: int,
    image_size: tuple[int, int],
    image_count: int,
    image_mode: str,
    task: str,
    jpeg_quality: int | None,
    rng: np.random.Generator,
) -> tuple[TimedObservation, int | None, int | None]:
    width, height = image_size
    observation: dict[str, object] = {f"state_{i}": 0.0 for i in range(state_dim)}
    for i in range(image_count):
        key = "image" if i == 0 else f"image_{i}"
        observation[key] = _make_image(width, height, image_mode, rng)
    observation["task"] = task

    raw_bytes = sum(value.nbytes for value in observation.values() if isinstance(value, np.ndarray))
    encoded_bytes = None
    if jpeg_quality is not None:
        observation, stats = encode_images_for_transport(observation, jpeg_quality=jpeg_quality)
        encoded_bytes = stats["encoded_bytes_total"]

    timed = TimedObservation(
        timestamp=time.time(),
        control_step=control_step,
        observation=observation,
        chunk_start_step=control_step,
    )
    return timed, raw_bytes, encoded_bytes


def _send_observation(stub: services_pb2_grpc.AsyncInferenceStub, timed: TimedObservation) -> Sample:
    payload = pickle.dumps(timed)
    iterator = send_bytes_in_chunks(payload, services_pb2.Observation, silent=True)
    start = time.perf_counter()
    stub.SendObservations(iterator)
    return Sample(payload_bytes=len(payload), send_s=time.perf_counter() - start)


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _summary(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"min={_fmt_ms(min(values))} "
        f"mean={_fmt_ms(statistics.fmean(values))} "
        f"median={_fmt_ms(statistics.median(values))} "
        f"max={_fmt_ms(max(values))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=_parse_size, default=(224, 224), metavar="WIDTHxHEIGHT")
    parser.add_argument("--image-count", type=int, default=2)
    parser.add_argument("--image-mode", choices=["zeros", "random", "gradient"], default="zeros")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--task", default="pick up the orange cube")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.image_count < 0:
        raise SystemExit("--image-count must be non-negative")
    if args.jpeg_quality is not None and not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    connect_s = _tcp_connect(args.host, args.port, args.connect_timeout_s)
    print(f"tcp_connect={_fmt_ms(connect_s)} target={args.host}:{args.port}")

    channel = grpc.insecure_channel(
        f"{args.host}:{args.port}",
        grpc_channel_options(max_receive_message_length=64 * 1024 * 1024, max_send_message_length=64 * 1024 * 1024),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    rng = np.random.default_rng(0)

    samples: list[Sample] = []
    raw_bytes_seen: int | None = None
    encoded_bytes_seen: int | None = None
    for i in range(args.repeats):
        timed, raw_bytes, encoded_bytes = _make_observation(
            control_step=i,
            state_dim=args.state_dim,
            image_size=args.image_size,
            image_count=args.image_count,
            image_mode=args.image_mode,
            task=args.task,
            jpeg_quality=args.jpeg_quality,
            rng=rng,
        )
        raw_bytes_seen = raw_bytes
        encoded_bytes_seen = encoded_bytes
        sample = _send_observation(stub, timed)
        samples.append(sample)
        print(
            f"sample={i} payload_bytes={sample.payload_bytes} "
            f"send={_fmt_ms(sample.send_s)}"
        )

    channel.close()

    sends = [sample.send_s for sample in samples]
    payloads = [sample.payload_bytes for sample in samples]
    print("summary")
    print(f"  image_size={args.image_size[0]}x{args.image_size[1]} image_count={args.image_count} mode={args.image_mode}")
    print(f"  jpeg_quality={args.jpeg_quality if args.jpeg_quality is not None else 'none'}")
    print(f"  raw_image_bytes={raw_bytes_seen}")
    print(f"  encoded_image_bytes={encoded_bytes_seen if encoded_bytes_seen is not None else 'n/a'}")
    print(f"  payload_bytes_min={min(payloads)} payload_bytes_max={max(payloads)}")
    print(f"  send_summary={_summary(sends)}")


if __name__ == "__main__":
    main()
