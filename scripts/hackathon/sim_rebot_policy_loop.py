#!/usr/bin/env python3
"""Run a minimal policy -> reBot kinematic simulator -> reward loop.

This is a plumbing test, not a physics simulator. It validates that we can:

1. Construct an observation.
2. Produce a 7D action vector.
3. Adapt it to 6 reBot joints + gripper.
4. Step a Pinocchio kinematic state.
5. Compute a task reward/success signal.
6. Record an episode CSV for later dataset/RL pipeline work.

Run from an environment where vectorBH6/reBotArm_control_py is importable:

    cd /Users/fbsh/projects/reBotArm_control_py
    PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py \
    uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/sim_rebot_policy_loop.py \
      --config /Users/fbsh/projects/drtc-Phi/config/rebot_sim_task_smoke.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle  # nosec B403: internal DRTC transport format.
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import yaml

from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik


@dataclass
class SimState:
    q: np.ndarray
    gripper: float


@dataclass
class AdapterConfig:
    joint_delta_scale_rad: float
    gripper_delta_scale: float
    max_abs_joint_rad: float
    min_gripper: float
    max_gripper: float


class PolicyBackend:
    def act(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class MockProportionalIKPolicy(PolicyBackend):
    def __init__(
        self,
        *,
        model: Any,
        end_frame_id: int,
        target_xyz: np.ndarray,
        adapter: AdapterConfig,
        action_clip: float,
    ) -> None:
        self.model = model
        self.end_frame_id = end_frame_id
        self.target_xyz = target_xyz
        self.adapter = adapter
        self.action_clip = action_clip

    def act(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        q = np.asarray(observation["q"], dtype=float)
        data = self.model.createData()
        _, current_rot, _ = compute_fk(self.model, q)
        target_pose = pin.SE3(current_rot, self.target_xyz)
        ik = solve_ik(
            self.model,
            data,
            self.end_frame_id,
            target_pose,
            q,
            IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6),
        )
        if ik.success:
            desired_delta = (ik.q - q) / self.adapter.joint_delta_scale_rad
        else:
            desired_delta = np.zeros_like(q)
        action = np.zeros(7, dtype=np.float32)
        action[:6] = np.clip(desired_delta, -self.action_clip, self.action_clip)
        return action, {"backend": "mock_proportional_ik", "ik_success": bool(ik.success), "ik_error": float(ik.error)}


class MockRandomPolicy(PolicyBackend):
    def __init__(self, *, action_clip: float, seed: int) -> None:
        self.action_clip = action_clip
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        action = self.rng.uniform(-self.action_clip, self.action_clip, size=7).astype(np.float32)
        return action, {"backend": "mock_random"}


class DRTCPolicy(PolicyBackend):
    def __init__(self, config: dict[str, Any], task_name: str) -> None:
        import grpc
        from lerobot.transport import services_pb2, services_pb2_grpc

        self.grpc = grpc
        self.services_pb2 = services_pb2
        self.server_address = str(config.get("server_address", "165.245.129.215:18201"))
        self.task = str(config.get("task", task_name))
        self.state_dim = int(config.get("state_dim", 8))
        self.image_size = tuple(config.get("image_size", [224, 224]))
        self.action_clip = float(config.get("action_clip", 1.0))
        self.wait_timeout_s = float(config.get("wait_timeout_s", 90.0))
        self.control_step = int(config.get("control_step_offset") or int(time.time() * 1000))
        self.action_buffer: queue.Queue[np.ndarray] = queue.Queue()
        self.dense_queue: queue.Queue[Any] = queue.Queue()
        self.stop_stream = threading.Event()
        self.run_start_ts = time.time()

        host, _ = self.server_address.rsplit(":", 1)
        no_proxy_hosts = {host, "127.0.0.1", "localhost"}
        for key in ("no_proxy", "NO_PROXY"):
            existing = {item for item in os.environ.get(key, "").split(",") if item}
            os.environ[key] = ",".join(sorted(existing | no_proxy_hosts))

        self._check_tcp()
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.stream_thread = threading.Thread(target=self._stream_actions, daemon=True)
        self.stream_thread.start()

        if not bool(config.get("skip_policy_setup", True)):
            self._setup_policy(config)

    def _check_tcp(self) -> None:
        host, port_raw = self.server_address.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect((host, int(port_raw)))
        finally:
            sock.close()

    def _stream_actions(self) -> None:
        try:
            for dense in self.stub.StreamActionsDense(self.services_pb2.Empty()):
                if self.stop_stream.is_set():
                    return
                self.dense_queue.put(dense)
        except self.grpc.RpcError as exc:
            if not self.stop_stream.is_set():
                self.dense_queue.put(exc)

    def _setup_policy(self, config: dict[str, Any]) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig

        checkpoint = config.get("pretrained_name_or_path")
        if not checkpoint:
            raise ValueError("policy.type=drtc with skip_policy_setup=false requires pretrained_name_or_path")

        self.stub.Ready(self.services_pb2.Empty())
        policy = RemotePolicyConfig(
            policy_type=str(config.get("policy_type", "pi05")),
            pretrained_name_or_path=str(checkpoint),
            lerobot_features=self._lerobot_features(),
            actions_per_chunk=int(config.get("actions_per_chunk", 50)),
            device=str(config.get("device", "cuda")),
            num_flow_matching_steps=int(config.get("num_flow_matching_steps", 8)),
        )
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy)))
        self.run_start_ts = time.time()

    def _lerobot_features(self) -> dict[str, dict[str, Any]]:
        width, height = self.image_size
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": [f"state_{i}" for i in range(self.state_dim)],
            },
            "observation.images.image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            },
        }

    def _make_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        width, height = self.image_size
        q = np.asarray(observation["q"], dtype=np.float32)
        ee = np.asarray(observation["ee_xyz"], dtype=np.float32)
        target = np.asarray(observation["target_xyz"], dtype=np.float32)
        distance = np.asarray([float(np.linalg.norm(ee - target))], dtype=np.float32)
        state = np.concatenate([q[:6], [float(observation["gripper"])], distance])[: self.state_dim]
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))

        obs: dict[str, Any] = {f"state_{i}": float(state[i]) for i in range(self.state_dim)}
        obs["image"] = np.zeros((height, width, 3), dtype=np.uint8)
        obs["wrist_image"] = np.zeros((height, width, 3), dtype=np.uint8)
        obs["task"] = self.task
        return obs

    def _chunked_observation(self, payload: bytes):
        chunk_size = 1024 * 1024
        if not payload:
            yield self.services_pb2.Observation(transfer_state=self.services_pb2.TRANSFER_END, data=b"")
            return
        for offset in range(0, len(payload), chunk_size):
            if offset + chunk_size >= len(payload):
                state = self.services_pb2.TRANSFER_END
            elif offset == 0:
                state = self.services_pb2.TRANSFER_BEGIN
            else:
                state = self.services_pb2.TRANSFER_MIDDLE
            yield self.services_pb2.Observation(transfer_state=state, data=payload[offset : offset + chunk_size])

    def _send_observation(self, observation: dict[str, Any]) -> None:
        from lerobot.async_inference.helpers import TimedObservation

        timed = TimedObservation(
            timestamp=time.time(),
            control_step=self.control_step,
            observation=self._make_policy_observation(observation),
            chunk_start_step=int(observation["step"]),
        )
        self.control_step += 1
        payload = pickle.dumps(timed)
        self.stub.SendObservations(self._chunked_observation(payload))

    def _fill_action_buffer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._send_observation(observation)
        deadline = time.monotonic() + self.wait_timeout_s
        while time.monotonic() < deadline:
            try:
                item = self.dense_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, self.grpc.RpcError):
                raise item
            dense = item
            if float(dense.timestamp) < self.run_start_ts:
                continue
            actions = np.frombuffer(dense.actions_f32, dtype=np.float32)
            actions = actions.reshape(int(dense.num_actions), int(dense.action_dim))
            for action in actions:
                clipped = np.clip(action[:7], -self.action_clip, self.action_clip).astype(np.float32)
                if clipped.shape[0] == 7:
                    self.action_buffer.put(clipped)
            return {
                "backend": "drtc",
                "received_actions": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]),
                "source_control_step": int(dense.source_control_step),
                "rtt_ms": max(0.0, (time.time() - float(dense.timestamp)) * 1000.0),
            }
        raise TimeoutError(f"Timed out waiting for DRTC actions from {self.server_address}")

    def act(self, observation: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        if self.action_buffer.empty():
            info = self._fill_action_buffer(observation)
        else:
            info = {"backend": "drtc", "received_actions": 0}
        return self.action_buffer.get(), info


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def _adapter_config(data: dict[str, Any]) -> AdapterConfig:
    raw = data["action_adapter"]
    return AdapterConfig(
        joint_delta_scale_rad=float(raw["joint_delta_scale_rad"]),
        gripper_delta_scale=float(raw["gripper_delta_scale"]),
        max_abs_joint_rad=float(raw["max_abs_joint_rad"]),
        min_gripper=float(raw["min_gripper"]),
        max_gripper=float(raw["max_gripper"]),
    )


def _make_observation(state: SimState, ee_xyz: np.ndarray, target_xyz: np.ndarray, step: int, task_name: str) -> dict[str, Any]:
    return {
        "step": step,
        "task": task_name,
        "q": state.q.astype(float).tolist(),
        "gripper": float(state.gripper),
        "ee_xyz": ee_xyz.astype(float).tolist(),
        "target_xyz": target_xyz.astype(float).tolist(),
    }


def _make_policy_backend(
    *,
    model: Any,
    end_frame_id: int,
    cfg: dict[str, Any],
    default_target_xyz: np.ndarray,
    adapter: AdapterConfig,
) -> PolicyBackend:
    policy_cfg = cfg.get("policy") or cfg.get("mock_policy")
    if not isinstance(policy_cfg, dict):
        raise ValueError("Config must contain a policy mapping")
    policy_type = str(policy_cfg["type"])
    action_clip = float(policy_cfg.get("action_clip", 1.0))
    if policy_type in {"mock_proportional_ik", "proportional_ik"}:
        target_xyz = np.asarray(policy_cfg.get("ik_target_xyz", default_target_xyz), dtype=float)
        return MockProportionalIKPolicy(
            model=model,
            end_frame_id=end_frame_id,
            target_xyz=target_xyz,
            adapter=adapter,
            action_clip=action_clip,
        )
    if policy_type == "mock_random":
        return MockRandomPolicy(action_clip=action_clip, seed=int(policy_cfg.get("random_seed", 0)))
    if policy_type == "drtc":
        return DRTCPolicy(policy_cfg, task_name=str(cfg["task"]["name"]))
    raise ValueError(f"Unsupported policy.type: {policy_type}")


def _apply_action(state: SimState, action: np.ndarray, adapter: AdapterConfig) -> SimState:
    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    q_next = state.q + action[:6].astype(float) * adapter.joint_delta_scale_rad
    q_next = np.clip(q_next, -adapter.max_abs_joint_rad, adapter.max_abs_joint_rad)
    gripper_next = state.gripper + float(action[6]) * adapter.gripper_delta_scale
    gripper_next = float(np.clip(gripper_next, adapter.min_gripper, adapter.max_gripper))
    return SimState(q=q_next, gripper=gripper_next)


def _reward(distance_m: float, success_distance_m: float) -> tuple[float, bool]:
    success = distance_m <= success_distance_m
    return -distance_m + (1.0 if success else 0.0), success


def _json_loads_or_none(raw: Any) -> Any:
    if not raw:
        return None
    if not isinstance(raw, str):
        return raw
    return json.loads(raw)


def _episode_steps_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in rows:
        steps.append(
            {
                "step": int(row["step"]),
                "observation": _json_loads_or_none(row["observation_json"]),
                "action": _json_loads_or_none(row.get("action")),
                "reward": float(row["reward"]),
                "success": bool(int(row["success"])),
                "distance_m": float(row["distance_m"]),
                "policy_info": _json_loads_or_none(row.get("policy_info_json")),
            }
        )
    return steps


def _append_episode_jsonl(
    *,
    path: Path,
    config_path: Path,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    policy_cfg = cfg.get("policy") or cfg.get("mock_policy") or {}
    episode = {
        "schema_version": "rebot_sim_episode_v1",
        "created_unix_s": time.time(),
        "config_path": str(config_path),
        "policy_type": policy_cfg.get("type"),
        "task": cfg["task"],
        "action_adapter": cfg["action_adapter"],
        "initial_state": cfg["initial_state"],
        "summary": summary,
        "steps": _episode_steps_from_rows(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(episode, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("results/rebot_sim_policy_loop.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/rebot_sim_policy_loop_summary.json"))
    parser.add_argument("--export-jsonl", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    task = cfg["task"]
    adapter = _adapter_config(cfg)
    target_xyz = np.asarray(task["target_xyz"], dtype=float)
    max_steps = int(task["max_steps"])
    success_distance_m = float(task["success_distance_m"])
    task_name = str(task["name"])

    model = load_robot_model()
    end_frame_id = get_end_effector_frame_id(model)
    initial = cfg["initial_state"]
    state = SimState(q=np.radians(np.asarray(initial["q_deg"], dtype=float)), gripper=float(initial["gripper"]))

    policy = _make_policy_backend(
        model=model,
        end_frame_id=end_frame_id,
        cfg=cfg,
        default_target_xyz=target_xyz,
        adapter=adapter,
    )

    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    success = False
    final_distance = float("inf")

    for step in range(max_steps):
        ee_xyz = compute_fk(model, state.q)[0]
        distance = float(np.linalg.norm(ee_xyz - target_xyz))
        reward, success = _reward(distance, success_distance_m)
        total_reward += reward
        observation = _make_observation(state, ee_xyz, target_xyz, step, task_name)

        rows.append(
            {
                "step": step,
                "task": task_name,
                "ee_x": f"{ee_xyz[0]:.8f}",
                "ee_y": f"{ee_xyz[1]:.8f}",
                "ee_z": f"{ee_xyz[2]:.8f}",
                "target_x": f"{target_xyz[0]:.8f}",
                "target_y": f"{target_xyz[1]:.8f}",
                "target_z": f"{target_xyz[2]:.8f}",
                "distance_m": f"{distance:.8f}",
                "reward": f"{reward:.8f}",
                "success": int(success),
                "q_rad": json.dumps(np.round(state.q, 8).tolist()),
                "q_deg": json.dumps(np.round(np.degrees(state.q), 5).tolist()),
                "gripper": f"{state.gripper:.6f}",
                "observation_json": json.dumps(observation, separators=(",", ":")),
            }
        )

        final_distance = distance
        if success:
            break

        action, policy_info = policy.act(observation)
        rows[-1]["action"] = json.dumps(np.round(action, 8).tolist())
        rows[-1]["policy_info_json"] = json.dumps(policy_info, separators=(",", ":"))
        state = _apply_action(state, action, adapter)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": task_name,
        "steps": len(rows),
        "success": success,
        "final_distance_m": final_distance,
        "total_reward": total_reward,
        "output_csv": str(args.output_csv),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))

    if args.export_jsonl is not None:
        _append_episode_jsonl(
            path=args.export_jsonl,
            config_path=args.config,
            cfg=cfg,
            summary=summary,
            rows=rows,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
