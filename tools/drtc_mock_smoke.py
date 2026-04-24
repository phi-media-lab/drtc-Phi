"""Run a local DRTC mock client/server smoke test.

This bypasses real robot hardware and real policy loading:
- client uses RobotClientDrtc(use_mock_robot=True)
- server uses PolicyServerDrtc(mock_policy=True)

The script is intentionally small and operational. It exists to validate the
DRTC transport/control path before moving the server to a remote machine.
"""

from __future__ import annotations

import argparse
import json
import pickle  # nosec
import threading
import time
from concurrent import futures
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import grpc

from lerobot.async_inference.configs_drtc import PolicyServerDrtcConfig, RobotClientDrtcConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server_drtc import ActionChunkCache, PolicyServerDrtc
from lerobot.async_inference.robot_client_drtc import RobotClientDrtc
from lerobot.robots.config import RobotConfig
from lerobot.transport import services_pb2, services_pb2_grpc


@RobotConfig.register_subclass("drtc_mock")
@dataclass(kw_only=True)
class DrtcMockRobotConfig(RobotConfig):
    pass


class SmokePolicyServer(PolicyServerDrtc):
    """PolicyServerDrtc variant that starts mock inference after handshake."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be RemotePolicyConfig, got {type(policy_specs)}")

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._action_cache = ActionChunkCache(max_size=self.actions_per_chunk)

        self._policy_ready.set()
        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._producer_thread = threading.Thread(
                target=self._inference_producer_loop,
                name="drtc_mock_policy_producer",
                daemon=True,
            )
            self._producer_thread.start()

        return services_pb2.Empty()


class _DeviceOnlyProcessor:
    def __init__(self, device: str):
        self.device = device

    def __call__(self, data):
        import torch

        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {key: self(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self(value) for value in data)
        return data


class _IdentityProcessor:
    def __call__(self, data):
        return data


class RandomActPolicyServer(PolicyServerDrtc):
    """PolicyServerDrtc variant that builds a random ACT policy after handshake."""

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be RemotePolicyConfig, got {type(policy_specs)}")

        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.utils.constants import ACTION, OBS_STATE

        self.device = policy_specs.device
        self.policy_type = "act"
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._action_cache = ActionChunkCache(max_size=self.actions_per_chunk)

        state_feature = self.lerobot_features.get(OBS_STATE, {})
        state_shape = tuple(state_feature.get("shape", (6,)))
        action_feature = self.lerobot_features.get(ACTION, {})
        action_shape = tuple(action_feature.get("shape", (6,)))

        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        }
        for key, feature in self.lerobot_features.items():
            if not key.startswith("observation.images."):
                continue
            h, w, c = tuple(feature.get("shape", (240, 320, 3)))
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))

        config = ACTConfig(
            input_features=input_features,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=action_shape)},
            device=self.device,
            chunk_size=self.actions_per_chunk,
            n_action_steps=self.actions_per_chunk,
            pretrained_backbone_weights=None,
            use_vae=False,
            dropout=0.0,
        )
        self.policy = ACTPolicy(config).to(self.device)
        self.policy.eval()
        self.preprocessor = _DeviceOnlyProcessor(self.device)
        self.postprocessor = _IdentityProcessor()

        self._policy_ready.set()
        if self._producer_thread is None or not self._producer_thread.is_alive():
            self._producer_thread = threading.Thread(
                target=self._inference_producer_loop,
                name="drtc_random_act_policy_producer",
                daemon=True,
            )
            self._producer_thread.start()

        return services_pb2.Empty()


def make_server(
    host: str,
    port: int,
    fps: int,
    *,
    policy_server_mode: str,
    warmup_passes: int,
) -> tuple[PolicyServerDrtc, grpc.Server, int]:
    server_cfg = PolicyServerDrtcConfig(
        host=host,
        port=port,
        fps=fps,
        mock_policy=policy_server_mode == "mock",
        mock_action_dim=6,
        warmup_passes=warmup_passes,
        metrics_diagnostic_enabled=True,
        metrics_diagnostic_interval_s=2.0,
        metrics_diagnostic_window_s=10.0,
    )
    if policy_server_mode == "mock":
        policy_server_class = SmokePolicyServer
    elif policy_server_mode == "random-act":
        policy_server_class = RandomActPolicyServer
    else:
        policy_server_class = PolicyServerDrtc
    policy_server = policy_server_class(server_cfg)

    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bound_port = grpc_server.add_insecure_port(f"{server_cfg.host}:{server_cfg.port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")
    grpc_server.start()
    return policy_server, grpc_server, bound_port


def make_client(
    server_address: str,
    fps: int,
    *,
    policy_type: str,
    pretrained_name_or_path: str,
    policy_device: str,
    actions_per_chunk: int,
    num_flow_matching_steps: int | None,
    policy_vlm_model_name: str | None,
    policy_load_vlm_weights: bool | None,
    policy_no_act_pretrained_backbone_weights: bool,
    camera_count: int,
    camera_width: int,
    camera_height: int,
    camera_names: list[str] | None,
) -> RobotClientDrtc:
    client_cfg = RobotClientDrtcConfig(
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        robot=DrtcMockRobotConfig(),
        actions_per_chunk=actions_per_chunk,
        server_address=server_address,
        policy_device=policy_device,
        policy_vlm_model_name=policy_vlm_model_name,
        policy_load_vlm_weights=policy_load_vlm_weights,
        policy_no_act_pretrained_backbone_weights=policy_no_act_pretrained_backbone_weights,
        fps=fps,
        s_min=8,
        epsilon=1,
        use_mock_robot=True,
        rtc_enabled=False,
        num_flow_matching_steps=num_flow_matching_steps,
        action_filter_mode="none",
        action_filter_butterworth_cutoff=3.0,
        metrics_diagnostic_enabled=True,
        metrics_diagnostic_interval_s=2.0,
        metrics_diagnostic_window_s=10.0,
        trajectory_viz_enabled=False,
    )
    client = RobotClientDrtc(client_cfg)
    client.policy_config.lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(client.robot.state_features),),
            "names": list(client.robot.state_features),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(client.robot.action_features),),
            "names": list(client.robot.action_features),
        },
    }

    if camera_count > 0:
        if camera_names is None:
            camera_names = [f"camera{i + 1}" for i in range(camera_count)]
        if len(camera_names) != camera_count:
            raise ValueError(f"expected {camera_count} camera names, got {len(camera_names)}")
        for i in range(camera_count):
            client.policy_config.lerobot_features[f"observation.images.{camera_names[i]}"] = {
                "dtype": "image",
                "shape": (camera_height, camera_width, 3),
                "names": ["height", "width", "channels"],
            }

        original_get_observation = client.robot.get_observation

        def get_observation_with_cameras():
            obs = original_get_observation()
            import numpy as np

            for i in range(camera_count):
                obs[camera_names[i]] = np.random.randint(
                    0,
                    255,
                    (camera_height, camera_width, 3),
                    dtype=np.uint8,
                )
            return obs

        client.robot.get_observation = get_observation_with_cameras  # type: ignore[method-assign]

    return client


def run_server(host: str, port: int, fps: int, *, policy_server_mode: str, warmup_passes: int) -> None:
    policy_server, grpc_server, bound_port = make_server(
        host=host,
        port=port,
        fps=fps,
        policy_server_mode=policy_server_mode,
        warmup_passes=warmup_passes,
    )
    print(f"DRTC mock server listening on {host}:{bound_port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        policy_server.stop()
        grpc_server.stop(grace=2)


def run_client(
    duration_s: float,
    fps: int,
    server_address: str,
    *,
    policy_type: str,
    pretrained_name_or_path: str,
    policy_device: str,
    actions_per_chunk: int,
    num_flow_matching_steps: int | None,
    policy_vlm_model_name: str | None,
    policy_load_vlm_weights: bool | None,
    policy_no_act_pretrained_backbone_weights: bool,
    camera_count: int,
    camera_width: int,
    camera_height: int,
    camera_names: list[str] | None,
    json_output: Path | None,
) -> dict[str, Any]:
    client = make_client(
        server_address=server_address,
        fps=fps,
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_name_or_path,
        policy_device=policy_device,
        actions_per_chunk=actions_per_chunk,
        num_flow_matching_steps=num_flow_matching_steps,
        policy_vlm_model_name=policy_vlm_model_name,
        policy_load_vlm_weights=policy_load_vlm_weights,
        policy_no_act_pretrained_backbone_weights=policy_no_act_pretrained_backbone_weights,
        camera_count=camera_count,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_names=camera_names,
    )

    received_chunks = 0
    executed_actions = 0

    original_publish = client._publish_received_actions

    def counting_publish(*args, **kwargs):
        nonlocal received_chunks
        received_chunks += 1
        return original_publish(*args, **kwargs)

    client._publish_received_actions = counting_publish  # type: ignore[method-assign]

    original_send_action = client.robot.send_action

    def counting_send_action(action):
        nonlocal executed_actions
        executed_actions += 1
        return original_send_action(action)

    client.robot.send_action = counting_send_action  # type: ignore[method-assign]

    obs_thread: threading.Thread | None = None
    action_thread: threading.Thread | None = None
    control_thread: threading.Thread | None = None

    try:
        if not client.start():
            raise RuntimeError("client failed to start")

        obs_thread = threading.Thread(target=client.observation_sender, name="smoke_obs_sender", daemon=True)
        action_thread = threading.Thread(target=client.action_receiver, name="smoke_action_receiver", daemon=True)
        control_thread = threading.Thread(
            target=client.control_loop,
            kwargs={"task": "smoke test"},
            name="smoke_control_loop",
            daemon=True,
        )

        obs_thread.start()
        action_thread.start()
        control_thread.start()
        time.sleep(duration_s)
    finally:
        client.stop()
        for thread in (obs_thread, action_thread, control_thread):
            if thread is not None:
                thread.join(timeout=3.0)

    result = {
        "duration_s": duration_s,
        "fps": fps,
        "server": server_address,
        "camera_count": camera_count,
        "camera_width": camera_width,
        "camera_height": camera_height,
        "received_chunks": received_chunks,
        "executed_actions": executed_actions,
        "final_action_step": client.current_action_step,
        "final_schedule_size": client.action_schedule.get_size(),
        "latency_steps": client.latency_estimator.estimate_steps,
        "latency_ms": client.latency_estimator.estimate_seconds * 1000.0,
    }

    print("DRTC mock smoke result")
    for key, value in result.items():
        if key == "latency_ms":
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if received_chunks <= 0 or executed_actions <= 0:
        raise RuntimeError("smoke test did not exchange and execute actions")
    return result


def run_smoke(
    duration_s: float,
    fps: int,
    host: str,
    port: int,
    *,
    policy_server_mode: str,
    warmup_passes: int,
    policy_type: str,
    pretrained_name_or_path: str,
    policy_device: str,
    actions_per_chunk: int,
    num_flow_matching_steps: int | None,
    policy_vlm_model_name: str | None,
    policy_load_vlm_weights: bool | None,
    policy_no_act_pretrained_backbone_weights: bool,
    camera_count: int,
    camera_width: int,
    camera_height: int,
    camera_names: list[str] | None,
    json_output: Path | None,
) -> dict[str, Any]:
    policy_server, grpc_server, bound_port = make_server(
        host=host,
        port=port,
        fps=fps,
        policy_server_mode=policy_server_mode,
        warmup_passes=warmup_passes,
    )
    try:
        return run_client(
            duration_s=duration_s,
            fps=fps,
            server_address=f"127.0.0.1:{bound_port}",
            policy_type=policy_type,
            pretrained_name_or_path=pretrained_name_or_path,
            policy_device=policy_device,
            actions_per_chunk=actions_per_chunk,
            num_flow_matching_steps=num_flow_matching_steps,
            policy_vlm_model_name=policy_vlm_model_name,
            policy_load_vlm_weights=policy_load_vlm_weights,
            policy_no_act_pretrained_backbone_weights=policy_no_act_pretrained_backbone_weights,
            camera_count=camera_count,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_names=camera_names,
            json_output=json_output,
        )
    finally:
        policy_server.stop()
        grpc_server.stop(grace=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", "server", "client"), default="all")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-address", default=None)
    parser.add_argument("--policy-server-mode", choices=("mock", "random-act", "real"), default="mock")
    parser.add_argument("--warmup-passes", type=int, default=0)
    parser.add_argument("--policy-type", default="smoke")
    parser.add_argument("--pretrained-name-or-path", default="mock")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument("--num-flow-matching-steps", type=int, default=None)
    parser.add_argument("--policy-vlm-model-name", default=None)
    parser.add_argument("--policy-load-vlm-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--policy-no-act-pretrained-backbone-weights", action="store_true")
    parser.add_argument("--camera-count", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-names", default=None, help="Comma-separated raw camera names, e.g. front,wrist")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.camera_count < 0:
        raise ValueError("--camera-count must be non-negative")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("--camera-width and --camera-height must be positive")
    if args.actions_per_chunk <= 0:
        raise ValueError("--actions-per-chunk must be positive")
    camera_names = args.camera_names.split(",") if args.camera_names else None
    if args.mode == "server":
        run_server(
            host=args.host,
            port=args.port,
            fps=args.fps,
            policy_server_mode=args.policy_server_mode,
            warmup_passes=args.warmup_passes,
        )
    elif args.mode == "client":
        server_address = args.server_address or f"{args.host}:{args.port}"
        run_client(
            duration_s=args.duration_s,
            fps=args.fps,
            server_address=server_address,
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            policy_device=args.policy_device,
            actions_per_chunk=args.actions_per_chunk,
            num_flow_matching_steps=args.num_flow_matching_steps,
            policy_vlm_model_name=args.policy_vlm_model_name,
            policy_load_vlm_weights=args.policy_load_vlm_weights,
            policy_no_act_pretrained_backbone_weights=args.policy_no_act_pretrained_backbone_weights,
            camera_count=args.camera_count,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=camera_names,
            json_output=args.json_output,
        )
    else:
        run_smoke(
            duration_s=args.duration_s,
            fps=args.fps,
            host=args.host,
            port=args.port,
            policy_server_mode=args.policy_server_mode,
            warmup_passes=args.warmup_passes,
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            policy_device=args.policy_device,
            actions_per_chunk=args.actions_per_chunk,
            num_flow_matching_steps=args.num_flow_matching_steps,
            policy_vlm_model_name=args.policy_vlm_model_name,
            policy_load_vlm_weights=args.policy_load_vlm_weights,
            policy_no_act_pretrained_backbone_weights=args.policy_no_act_pretrained_backbone_weights,
            camera_count=args.camera_count,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=camera_names,
            json_output=args.json_output,
        )


if __name__ == "__main__":
    main()
