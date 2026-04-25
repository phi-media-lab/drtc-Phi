#!/usr/bin/env python3
"""Lightweight reComputer DRTC observation client.

This script is intended for Jetson/reComputer robot-client bring-up where we do
not want to install the full LeRobot/Torch stack. It creates pickle-compatible
`lerobot.async_inference.helpers` objects locally, sends synthetic observations,
and validates that a remote DRTC server returns dense action chunks.
"""

from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    main()
