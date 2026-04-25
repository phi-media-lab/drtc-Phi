#!/usr/bin/env python3
"""Lightweight reComputer DRTC observation client.

This script is intended for Jetson/reComputer robot-client bring-up where we do
not want to install the full LeRobot/Torch stack. It creates pickle-compatible
`lerobot.async_inference.helpers` objects locally, sends synthetic observations,
and validates that a remote DRTC server returns dense action chunks.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import queue
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np
from PIL import Image

from lerobot.transport import services_pb2, services_pb2_grpc


def _install_pickle_compatible_helpers() -> None:
    """Install minimal classes under the module path expected by the server."""
    sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
    sys.modules.setdefault("lerobot.async_inference", types.ModuleType("lerobot.async_inference"))
    helpers_mod = types.ModuleType("lerobot.async_inference.helpers")

    @dataclass
    class TimedData:
        timestamp: float
        control_step: int

        def get_timestamp(self):
            return self.timestamp

        def get_control_step(self):
            return self.control_step

    @dataclass
    class TimedObservation(TimedData):
        observation: dict[str, Any] | None = None
        chunk_start_step: int = 0
        must_go: bool = False
        server_received_ts: float = 0.0

        def get_observation(self):
            return self.observation

    @dataclass
    class RemotePolicyConfig:
        policy_type: str
        pretrained_name_or_path: str
        lerobot_features: dict[str, dict[str, Any]]
        actions_per_chunk: int
        device: str = "cpu"
        rename_map: dict[str, str] = field(default_factory=dict)
        rtc_enabled: bool = False
        rtc_max_guidance_weight: float | None = None
        rtc_prefix_attention_schedule: str = "linear"
        rtc_sigma_d: float = 1.0
        rtc_full_trajectory_alignment: bool = False
        num_flow_matching_steps: int | None = None
        spikes: list[dict] = field(default_factory=list)
        diagnostics_verbose: bool = False

    TimedData.__module__ = helpers_mod.__name__
    TimedObservation.__module__ = helpers_mod.__name__
    RemotePolicyConfig.__module__ = helpers_mod.__name__
    TimedData.__qualname__ = "TimedData"
    TimedObservation.__qualname__ = "TimedObservation"
    RemotePolicyConfig.__qualname__ = "RemotePolicyConfig"
    helpers_mod.TimedData = TimedData
    helpers_mod.TimedObservation = TimedObservation
    helpers_mod.RemotePolicyConfig = RemotePolicyConfig
    sys.modules[helpers_mod.__name__] = helpers_mod


_install_pickle_compatible_helpers()
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402


CHUNK_SIZE = 2 * 1024 * 1024


def _send_bytes_in_chunks(buffer: bytes, message_class: Any):
    sent = 0
    total = len(buffer)
    while sent < total:
        if sent + CHUNK_SIZE >= total:
            state = services_pb2.TransferState.TRANSFER_END
        elif sent == 0:
            state = services_pb2.TransferState.TRANSFER_BEGIN
        else:
            state = services_pb2.TransferState.TRANSFER_MIDDLE
        chunk = buffer[sent : sent + CHUNK_SIZE]
        yield message_class(transfer_state=state, data=chunk)
        sent += len(chunk)


def _decode_dense(dense: services_pb2.ActionsDense) -> np.ndarray:
    arr = np.frombuffer(dense.actions_f32, dtype=np.float32)
    return arr.reshape(int(dense.num_actions), int(dense.action_dim))


def _latency_steps(latency_ms: float, fps: float) -> int:
    return max(1, int(math.ceil((latency_ms / 1000.0) * fps)))


def _adapt_action(
    action: np.ndarray,
    prev_action: np.ndarray | None,
    *,
    clip_abs: float,
    max_delta: float,
) -> np.ndarray:
    adapted = np.clip(action.astype(np.float32), -clip_abs, clip_abs)
    if prev_action is not None:
        delta = np.clip(adapted - prev_action, -max_delta, max_delta)
        adapted = prev_action + delta
    return adapted.astype(np.float32)


def _jpeg_encode_image(rgb: np.ndarray, quality: int) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def _stream_actions(stub: services_pb2_grpc.AsyncInferenceStub, out: queue.Queue, stop: threading.Event):
    try:
        for dense in stub.StreamActionsDense(services_pb2.Empty()):
            if stop.is_set():
                return
            out.put(dense)
    except grpc.RpcError as exc:
        if not stop.is_set():
            out.put(exc)


def _drain_actions(
    action_q: queue.Queue,
    schedule: dict[int, np.ndarray],
    *,
    latest_rtt_ms: float | None,
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
        actions = _decode_dense(item)
        for i, action in enumerate(actions):
            schedule[int(item.chunk_start_step) + i] = action
        received += 1
        latest_rtt_ms = max(0.0, (time.time() - float(item.timestamp)) * 1000.0)
        last_source_step = int(item.source_control_step)
        last_chunk_start = int(item.chunk_start_step)
        last_num_actions = int(item.num_actions)
        last_action_dim = int(item.action_dim)
    return received, latest_rtt_ms, last_source_step, last_chunk_start, last_num_actions, last_action_dim


def _features(state_dim: int, width: int, height: int, image_names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        }
    }
    for name in image_names:
        result[f"observation.images.{name}"] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return result


def _make_observation(
    *,
    control_step: int,
    action_step: int,
    state_dim: int,
    width: int,
    height: int,
    image_names: list[str],
    task: str,
    jpeg_quality: int | None,
) -> TimedObservation:
    obs: dict[str, Any] = {f"state_{i}": 0.0 for i in range(state_dim)}
    for name in image_names:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if jpeg_quality is None:
            obs[name] = image
        else:
            obs[name] = {
                "__lerobot_image_encoding__": "jpeg",
                "quality": int(jpeg_quality),
                "data": _jpeg_encode_image(image, jpeg_quality),
            }
    obs["task"] = task
    return TimedObservation(
        timestamp=time.time(),
        control_step=control_step,
        observation=obs,
        chunk_start_step=action_step,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default="127.0.0.1:18203")
    parser.add_argument("--pretrained-name-or-path", default="jackvial/so101_smolvla_pickplaceorangecube_e100")
    parser.add_argument("--policy-type", default="smolvla")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--image-size", default="800x600")
    parser.add_argument("--image-feature-names", default="camera1,camera2")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--task", default="Pick up the orange cube and place it in the target area.")
    parser.add_argument("--setup-policy", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--run-loop", action="store_true", help="Run a simulated 15Hz scheduler loop.")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--s-min", type=int, default=25)
    parser.add_argument("--epsilon", type=int, default=2)
    parser.add_argument("--initial-latency-ms", type=float, default=50.0)
    parser.add_argument("--clip-abs", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    width, height = [int(x) for x in args.image_size.lower().split("x", 1)]
    image_names = [x.strip() for x in args.image_feature_names.split(",") if x.strip()]

    channel = grpc.insecure_channel(
        args.server_address,
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ],
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    t0 = time.perf_counter()
    stub.Ready(services_pb2.Empty())
    print(f"ready_ms={(time.perf_counter() - t0) * 1000:.1f}")

    if args.setup_policy:
        policy = RemotePolicyConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            lerobot_features=_features(args.state_dim, width, height, image_names),
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
        )
        t0 = time.perf_counter()
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy)))
        print(f"policy_setup_ms={(time.perf_counter() - t0) * 1000:.1f}")

    action_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(target=_stream_actions, args=(stub, action_q, stop), daemon=True)
    thread.start()

    if args.run_loop:
        _run_scheduler_loop(args, stub, action_q, width=width, height=height, image_names=image_names)
        stop.set()
        channel.close()
        return

    obs = _make_observation(
        control_step=int(time.time() * 1000),
        action_step=0,
        state_dim=args.state_dim,
        width=width,
        height=height,
        image_names=image_names,
        task=args.task,
        jpeg_quality=args.jpeg_quality,
    )
    payload = pickle.dumps(obs)
    t0 = time.perf_counter()
    stub.SendObservations(_send_bytes_in_chunks(payload, services_pb2.Observation))
    send_ms = (time.perf_counter() - t0) * 1000
    print(f"observation_sent_ms={send_ms:.1f} payload_bytes={len(payload)}")

    deadline = time.monotonic() + args.timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                item = action_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, grpc.RpcError):
                raise item
            actions = _decode_dense(item)
            print(
                "chunk_received "
                f"num_actions={item.num_actions} action_dim={item.action_dim} "
                f"chunk_start_step={item.chunk_start_step} source_control_step={item.source_control_step}"
            )
            print(
                "actions "
                f"shape={actions.shape} min={actions.min():.6f} max={actions.max():.6f} "
                f"mean={actions.mean():.6f} std={actions.std():.6f}"
            )
            print("first_action", np.round(actions[0], 6).tolist())
            return
        raise TimeoutError(f"No action chunk received within {args.timeout_s}s")
    finally:
        stop.set()
        channel.close()


def _run_scheduler_loop(
    args: argparse.Namespace,
    stub: services_pb2_grpc.AsyncInferenceStub,
    action_q: queue.Queue,
    *,
    width: int,
    height: int,
    image_names: list[str],
) -> None:
    schedule: dict[int, np.ndarray] = {}
    latest_rtt_ms: float | None = args.initial_latency_ms
    rng_control_step = int(time.time() * 1000)
    control_step = rng_control_step
    action_step = 0
    period_s = 1.0 / args.fps
    cooldown = 0
    trigger_threshold = args.actions_per_chunk - args.s_min
    next_tick = time.monotonic()
    end_time = time.monotonic() + args.duration_s
    rows: list[dict[str, Any]] = []
    stalls = 0
    sent_observations = 0
    chunks_received = 0
    prev_adapted: np.ndarray | None = None

    def _send(action_start_step: int) -> tuple[float, int]:
        nonlocal control_step, sent_observations
        obs = _make_observation(
            control_step=control_step,
            action_step=action_start_step,
            state_dim=args.state_dim,
            width=width,
            height=height,
            image_names=image_names,
            task=args.task,
            jpeg_quality=args.jpeg_quality,
        )
        payload = pickle.dumps(obs)
        t0 = time.perf_counter()
        stub.SendObservations(_send_bytes_in_chunks(payload, services_pb2.Observation))
        send_ms = (time.perf_counter() - t0) * 1000
        control_step += 1
        sent_observations += 1
        return send_ms, len(payload)

    last_send_ms, last_payload_bytes = _send(action_step)
    print(
        f"loop_start fps={args.fps:g} duration_s={args.duration_s:g} "
        f"H={args.actions_per_chunk} s_min={args.s_min} jpeg_quality={args.jpeg_quality}"
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
        )
        chunks_received += received

        raw_action = schedule.pop(action_step, None)
        action_received = raw_action is not None
        if not action_received:
            stalls += 1
            raw_action = np.zeros(6, dtype=np.float32) if prev_adapted is None else prev_adapted

        adapted = _adapt_action(
            raw_action,
            prev_adapted,
            clip_abs=args.clip_abs,
            max_delta=args.max_delta,
        )
        prev_adapted = adapted
        raw_norm = float(np.linalg.norm(raw_action))
        adapted_norm = float(np.linalg.norm(adapted))

        action_step += 1
        latency_steps = _latency_steps(latest_rtt_ms or args.initial_latency_ms, args.fps)
        schedule_size = sum(1 for step in schedule if step >= action_step)
        obs_triggered = False
        if schedule_size <= trigger_threshold and cooldown <= 0:
            last_send_ms, last_payload_bytes = _send(action_step)
            obs_triggered = True
            cooldown = latency_steps + args.epsilon
        else:
            cooldown = max(0, cooldown - 1)

        rows.append(
            {
                "wall_time": f"{tick_wall:.6f}",
                "action_step": action_step,
                "schedule_size": schedule_size,
                "stall": int(not action_received),
                "obs_triggered": int(obs_triggered),
                "chunks_received_total": chunks_received,
                "chunks_received_tick": received,
                "latency_estimate_ms": f"{(latest_rtt_ms or 0.0):.3f}",
                "latency_steps": latency_steps,
                "cooldown": cooldown,
                "last_send_ms": f"{last_send_ms:.3f}",
                "last_payload_bytes": last_payload_bytes,
                "last_source_step": last_source_step if last_source_step is not None else "",
                "last_chunk_start": last_chunk_start if last_chunk_start is not None else "",
                "last_num_actions": last_num_actions if last_num_actions is not None else "",
                "last_action_dim": last_action_dim if last_action_dim is not None else "",
                "raw_action_norm": f"{raw_norm:.6f}",
                "adapted_action_norm": f"{adapted_norm:.6f}",
                "adapted_action": " ".join(f"{x:.6f}" for x in adapted.tolist()),
            }
        )

        next_tick += period_s

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    print("summary")
    print(f"  output={args.output or '<none>'}")
    print(f"  ticks={len(rows)} stalls={stalls} starvation_ratio={(stalls / len(rows)) if rows else 0:.4f}")
    print(f"  observations_sent={sent_observations} chunks_received={chunks_received}")
    if rows:
        schedule_sizes = [int(row["schedule_size"]) for row in rows]
        send_times = [float(row["last_send_ms"]) for row in rows]
        latencies = [float(row["latency_estimate_ms"]) for row in rows]
        print(f"  schedule_min={min(schedule_sizes)} schedule_max={max(schedule_sizes)}")
        print(f"  latency_ms_min={min(latencies):.1f} median={float(np.median(latencies)):.1f} max={max(latencies):.1f}")
        print(f"  send_ms_min={min(send_times):.1f} median={float(np.median(send_times)):.1f} max={max(send_times):.1f}")


if __name__ == "__main__":
    main()
