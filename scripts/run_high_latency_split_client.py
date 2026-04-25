#!/usr/bin/env python3
"""Run a synthetic DRTC split-client schedule test against a remote server.

This script is intentionally smaller than RobotClientDrtc: it validates the
networked control schedule with synthetic observations while still using the
real DRTC gRPC server, policy setup RPC, observation transport, and dense action
stream. It is useful before wiring in a physical robot client.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle  # nosec B403: internal DRTC transport format.
import queue
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import grpc
import numpy as np

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.utils.compression import encode_images_for_transport
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


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


def _parse_image_feature_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--image-feature-names must include at least one image key")
    return names


def _lerobot_features(
    state_dim: int, image_size: tuple[int, int], image_feature_names: list[str]
) -> dict[str, dict[str, Any]]:
    width, height = image_size
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        }
    }
    for image_name in image_feature_names:
        features[f"observation.images.{image_name}"] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _make_observation(
    *,
    control_step: int,
    action_step: int,
    state_dim: int,
    image_size: tuple[int, int],
    image_mode: str,
    image_feature_names: list[str],
    task: str,
    jpeg_quality: int | None,
    rng: np.random.Generator,
    rtc_meta: dict[str, Any] | None,
) -> tuple[TimedObservation, int, int | None, int]:
    width, height = image_size
    observation: dict[str, Any] = {f"state_{i}": 0.0 for i in range(state_dim)}
    for image_name in image_feature_names:
        observation[image_name] = _make_image(width, height, image_mode, rng)
    observation["task"] = task
    if rtc_meta is not None:
        observation["__rtc__"] = rtc_meta

    raw_image_bytes = sum(observation[image_name].nbytes for image_name in image_feature_names)
    encoded_image_bytes = None
    if jpeg_quality is not None:
        observation, stats = encode_images_for_transport(observation, jpeg_quality=jpeg_quality)
        encoded_image_bytes = int(stats["encoded_bytes_total"])

    timed = TimedObservation(
        timestamp=time.time(),
        control_step=control_step,
        observation=observation,
        chunk_start_step=action_step,
    )
    payload_bytes = len(pickle.dumps(timed))
    return timed, raw_image_bytes, encoded_image_bytes, payload_bytes


def _send_observation(stub: services_pb2_grpc.AsyncInferenceStub, obs: TimedObservation) -> float:
    payload = pickle.dumps(obs)
    start = time.perf_counter()
    stub.SendObservations(send_bytes_in_chunks(payload, services_pb2.Observation, silent=True))
    return time.perf_counter() - start


def _decode_dense(dense: services_pb2.ActionsDense) -> np.ndarray:
    arr = np.frombuffer(dense.actions_f32, dtype=np.float32)
    expected = int(dense.num_actions) * int(dense.action_dim)
    if arr.size != expected:
        raise ValueError(f"ActionsDense size mismatch: {arr.size} != {expected}")
    return arr.reshape(int(dense.num_actions), int(dense.action_dim))


def _stream_actions(
    stub: services_pb2_grpc.AsyncInferenceStub,
    out: queue.Queue[services_pb2.ActionsDense],
    stop: threading.Event,
) -> None:
    try:
        for dense in stub.StreamActionsDense(services_pb2.Empty()):
            if stop.is_set():
                return
            out.put(dense)
    except grpc.RpcError as exc:
        if not stop.is_set():
            out.put(exc)


def _setup_policy(args: argparse.Namespace, stub: services_pb2_grpc.AsyncInferenceStub) -> tuple[float, float]:
    start = time.perf_counter()
    stub.Ready(services_pb2.Empty())
    ready_s = time.perf_counter() - start

    policy = RemotePolicyConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=args.pretrained_name_or_path,
        lerobot_features=_lerobot_features(args.state_dim, args.image_size, args.image_feature_names),
        actions_per_chunk=args.actions_per_chunk,
        device=args.device,
        rtc_enabled=args.rtc_enabled,
        rtc_full_trajectory_alignment=args.rtc_full_trajectory_alignment,
        num_flow_matching_steps=args.num_flow_matching_steps,
    )
    start = time.perf_counter()
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy)))
    setup_s = time.perf_counter() - start
    return ready_s, setup_s


def _latency_steps(latency_ms: float, fps: float) -> int:
    return max(1, int(math.ceil((latency_ms / 1000.0) * fps)))


def _drain_actions(
    action_q: queue.Queue[services_pb2.ActionsDense],
    schedule: dict[int, np.ndarray],
    *,
    latest_rtt_ms: float | None,
    min_dense_timestamp: float,
) -> tuple[int, float | None, int | None, int | None, int | None, int | None]:
    received = 0
    last_source_step = None
    last_chunk_start = None
    last_num_actions = None
    last_action_dim = None
    while True:
        try:
            item = action_q.get_nowait()
        except queue.Empty:
            break
        if isinstance(item, grpc.RpcError):
            raise item
        dense = item
        if float(dense.timestamp) < min_dense_timestamp:
            continue
        actions = _decode_dense(dense)
        for i, action in enumerate(actions):
            schedule[int(dense.chunk_start_step) + i] = action
        received += 1
        latest_rtt_ms = max(0.0, (time.time() - float(dense.timestamp)) * 1000.0)
        last_source_step = int(dense.source_control_step)
        last_chunk_start = int(dense.chunk_start_step)
        last_num_actions = int(dense.num_actions)
        last_action_dim = int(dense.action_dim)
    return received, latest_rtt_ms, last_source_step, last_chunk_start, last_num_actions, last_action_dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="165.245.129.215:18201")
    parser.add_argument("--pretrained-name-or-path", required=True)
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--s-min", type=int, default=25)
    parser.add_argument("--epsilon", type=int, default=2)
    parser.add_argument("--initial-latency-ms", type=float, default=350.0)
    parser.add_argument("--prefill-chunks", type=int, default=1)
    parser.add_argument("--prefill-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--control-step-offset",
        type=int,
        default=None,
        help="Logical control-step base. Defaults to a wall-clock-derived value to avoid stale LWW drops.",
    )
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=_parse_size, default=(224, 224), metavar="WIDTHxHEIGHT")
    parser.add_argument("--image-mode", choices=["zeros", "random", "gradient"], default="zeros")
    parser.add_argument(
        "--image-feature-names",
        type=_parse_image_feature_names,
        default=_parse_image_feature_names("image,wrist_image"),
        help="Comma-separated raw image keys mapped to observation.images.<key>.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=40)
    parser.add_argument("--task", default="pick up the orange cube")
    parser.add_argument("--num-flow-matching-steps", type=int, default=8)
    parser.add_argument("--rtc-enabled", action="store_true")
    parser.add_argument("--rtc-full-trajectory-alignment", action="store_true")
    parser.add_argument("--skip-policy-setup", action="store_true")
    parser.add_argument("--output", default="results/drtc_high_latency_split_client.csv")
    args = parser.parse_args()

    if args.actions_per_chunk <= 0:
        raise SystemExit("--actions-per-chunk must be positive")
    if args.s_min <= 0 or args.s_min >= args.actions_per_chunk:
        raise SystemExit("--s-min must be in (0, actions_per_chunk)")
    if args.jpeg_quality is not None and not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    host, port_raw = args.server_address.rsplit(":", 1)
    port = int(port_raw)
    connect_s = _tcp_connect(host, port, timeout_s=5.0)
    print(f"tcp_connect_ms={connect_s * 1000:.1f} target={args.server_address}")

    channel = grpc.insecure_channel(
        args.server_address,
        grpc_channel_options(max_receive_message_length=64 * 1024 * 1024, max_send_message_length=64 * 1024 * 1024),
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    if args.skip_policy_setup:
        ready_s = 0.0
        setup_s = 0.0
        print("policy_setup=skipped")
    else:
        ready_s, setup_s = _setup_policy(args, stub)
        print(f"ready_s={ready_s:.3f} policy_setup_s={setup_s:.3f}")

    action_q: queue.Queue[services_pb2.ActionsDense] = queue.Queue()
    stop_stream = threading.Event()
    stream_thread = threading.Thread(target=_stream_actions, args=(stub, action_q, stop_stream), daemon=True)
    stream_thread.start()

    rng = np.random.default_rng(0)
    run_start_ts = time.time()
    schedule: dict[int, np.ndarray] = {}
    latest_rtt_ms: float | None = args.initial_latency_ms
    control_step = args.control_step_offset if args.control_step_offset is not None else int(time.time() * 1000)
    action_step = 0
    last_send_ms: float | None = None
    raw_image_bytes = 0
    encoded_image_bytes: int | None = None
    payload_bytes = 0

    for _ in range(args.prefill_chunks):
        timed, raw_image_bytes, encoded_image_bytes, payload_bytes = _make_observation(
            control_step=control_step,
            action_step=action_step,
            state_dim=args.state_dim,
            image_size=args.image_size,
            image_mode=args.image_mode,
            image_feature_names=args.image_feature_names,
            task=args.task,
            jpeg_quality=args.jpeg_quality,
            rng=rng,
            rtc_meta=None,
        )
        last_send_ms = _send_observation(stub, timed) * 1000.0
        control_step += 1

    prefill_deadline = time.monotonic() + args.prefill_timeout_s
    while len(schedule) < args.actions_per_chunk * args.prefill_chunks:
        received, latest_rtt_ms, *_ = _drain_actions(
            action_q,
            schedule,
            latest_rtt_ms=latest_rtt_ms,
            min_dense_timestamp=run_start_ts,
        )
        if received:
            print(f"prefill_actions={len(schedule)} latest_rtt_ms={latest_rtt_ms:.1f}")
            continue
        if time.monotonic() > prefill_deadline:
            print(f"prefill_timeout actions={len(schedule)}")
            break
        time.sleep(0.05)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    period_s = 1.0 / args.fps
    end_time = time.monotonic() + args.duration_s
    next_tick = time.monotonic()
    cooldown = 0
    trigger_threshold = args.actions_per_chunk - args.s_min
    stalls = 0
    sent_observations = args.prefill_chunks
    chunks_received = len(schedule) // args.actions_per_chunk

    print(
        "loop_start "
        f"fps={args.fps:g} H={args.actions_per_chunk} s_min={args.s_min} "
        f"threshold={trigger_threshold} duration_s={args.duration_s:g}"
    )

    while time.monotonic() < end_time:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        tick_wall = time.time()

        received, latest_rtt_ms, last_source_step, last_chunk_start, last_num_actions, last_action_dim = _drain_actions(
            action_q,
            schedule,
            latest_rtt_ms=latest_rtt_ms,
            min_dense_timestamp=run_start_ts,
        )
        chunks_received += received

        action_received = action_step in schedule
        if action_received:
            schedule.pop(action_step, None)
        else:
            stalls += 1
        action_step += 1

        latency_steps = _latency_steps(latest_rtt_ms or args.initial_latency_ms, args.fps)
        schedule_size = sum(1 for step in schedule if step >= action_step)
        obs_triggered = False
        if schedule_size <= trigger_threshold and cooldown <= 0:
            rtc_meta = None
            if args.rtc_enabled:
                rtc_meta = {
                    "latency_steps": latency_steps,
                    "overlap_end": args.actions_per_chunk - max(args.s_min, latency_steps),
                    "action_schedule_spans": [],
                }
            timed, raw_image_bytes, encoded_image_bytes, payload_bytes = _make_observation(
                control_step=control_step,
                action_step=action_step,
                state_dim=args.state_dim,
                image_size=args.image_size,
                image_mode=args.image_mode,
                image_feature_names=args.image_feature_names,
                task=args.task,
                jpeg_quality=args.jpeg_quality,
                rng=rng,
                rtc_meta=rtc_meta,
            )
            last_send_ms = _send_observation(stub, timed) * 1000.0
            sent_observations += 1
            control_step += 1
            obs_triggered = True
            cooldown = latency_steps + args.epsilon
        else:
            cooldown = max(0, cooldown - 1)

        rows.append(
            {
                "wall_time": f"{tick_wall:.6f}",
                "control_step": control_step,
                "action_step": action_step,
                "schedule_size": schedule_size,
                "stall": int(not action_received),
                "obs_triggered": int(obs_triggered),
                "chunks_received_total": chunks_received,
                "chunks_received_tick": received,
                "latency_estimate_ms": f"{(latest_rtt_ms or 0.0):.3f}",
                "latency_steps": latency_steps,
                "cooldown": cooldown,
                "last_send_ms": f"{last_send_ms:.3f}" if last_send_ms is not None else "",
                "last_source_step": last_source_step if last_source_step is not None else "",
                "last_chunk_start": last_chunk_start if last_chunk_start is not None else "",
                "last_num_actions": last_num_actions if last_num_actions is not None else "",
                "last_action_dim": last_action_dim if last_action_dim is not None else "",
                "payload_bytes": payload_bytes,
                "raw_image_bytes": raw_image_bytes,
                "encoded_image_bytes": encoded_image_bytes if encoded_image_bytes is not None else "",
            }
        )

        next_tick += period_s

    stop_stream.set()
    channel.close()

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    schedule_sizes = [int(row["schedule_size"]) for row in rows]
    latency_ms = [float(row["latency_estimate_ms"]) for row in rows if float(row["latency_estimate_ms"]) > 0]
    print("summary")
    print(f"  output={output}")
    print(f"  ready_s={ready_s:.3f} policy_setup_s={setup_s:.3f}")
    print(f"  ticks={len(rows)} stalls={stalls} starvation_ratio={stalls / max(1, len(rows)):.4f}")
    print(f"  observations_sent={sent_observations} chunks_received={chunks_received}")
    print(f"  schedule_min={min(schedule_sizes) if schedule_sizes else 0} schedule_max={max(schedule_sizes) if schedule_sizes else 0}")
    if latency_ms:
        print(
            "  latency_ms="
            f"min={min(latency_ms):.1f} median={statistics.median(latency_ms):.1f} max={max(latency_ms):.1f}"
        )


if __name__ == "__main__":
    main()
