#!/usr/bin/env python3
"""Evaluate a trained SAC checkpoint in the reBot kinematic simulator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import torch

import lerobot
from lerobot.configs.types import FeatureType, PolicyFeature

# Avoid lerobot.policies.__init__ eager-importing unrelated heavy policy stacks
# such as GROOT/diffusers. This evaluation only needs the SAC subpackage.
_LEROBOT_ROOT = Path(lerobot.__file__).resolve().parent
_POLICIES_STUB = types.ModuleType("lerobot.policies")
_POLICIES_STUB.__path__ = [str(_LEROBOT_ROOT / "policies")]
sys.modules.setdefault("lerobot.policies", _POLICIES_STUB)

from lerobot.policies.sac.configuration_sac import ActorNetworkConfig, CriticNetworkConfig, SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik


STATE_NAMES = [
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
    "ee_x",
    "ee_y",
    "ee_z",
    "target_x",
    "target_y",
    "target_z",
]


def _sample_targets(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.asarray(
            [
                float(rng.uniform(0.24, 0.32)),
                float(rng.uniform(-0.08, 0.12)),
                float(rng.uniform(0.21, 0.29)),
            ],
            dtype=float,
        )
        for _ in range(count)
    ]


def _load_policy(dataset_root: Path, checkpoint: Path, device: str) -> SACPolicy:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    state_min = stats[OBS_STATE]["min"]
    state_max = stats[OBS_STATE]["max"]
    action_min = stats[ACTION]["min"]
    action_max = stats[ACTION]["max"]
    cfg = SACConfig(
        device=device,
        storage_device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(state_min),))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_min),))},
        dataset_stats={
            OBS_STATE: {"min": state_min, "max": state_max},
            ACTION: {"min": action_min, "max": action_max},
        },
        state_encoder_hidden_dim=32,
        latent_dim=32,
        critic_network_kwargs=CriticNetworkConfig(hidden_dims=[64, 64]),
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[64, 64]),
        use_torch_compile=False,
    )
    cfg.validate_features()
    policy = SACPolicy(config=cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _state_vector(q: np.ndarray, gripper: float, ee_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    return np.asarray([*q[:6], gripper, *ee_xyz, *target_xyz], dtype=np.float32)


def _apply_action(q: np.ndarray, gripper: float, action: np.ndarray) -> tuple[np.ndarray, float]:
    q_next = q + np.clip(action[:6], -1.0, 1.0).astype(float) * 0.035
    q_next = np.clip(q_next, -2.8, 2.8)
    gripper_next = float(np.clip(gripper + float(np.clip(action[6], -1.0, 1.0)) * 0.05, 0.0, 1.0))
    return q_next, gripper_next


def _ik_action(model: Any, end_frame_id: int, q: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    data = model.createData()
    _, current_rot, _ = compute_fk(model, q)
    ik = solve_ik(
        model,
        data,
        end_frame_id,
        pin.SE3(current_rot, target_xyz),
        q,
        IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6),
    )
    action = np.zeros(7, dtype=np.float32)
    if ik.success:
        action[:6] = np.clip((ik.q - q) / 0.035, -1.0, 1.0)
    return action


def _sac_action(policy: SACPolicy, q: np.ndarray, gripper: float, ee_xyz: np.ndarray, target_xyz: np.ndarray, device: str) -> np.ndarray:
    state = torch.as_tensor(_state_vector(q, gripper, ee_xyz, target_xyz), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy.select_action({OBS_STATE: state})
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _rollout(
    *,
    policy_name: str,
    target_xyz: np.ndarray,
    model: Any,
    end_frame_id: int,
    sac_policy: SACPolicy | None,
    rng: np.random.Generator,
    device: str,
    max_steps: int,
    success_distance_m: float,
) -> dict[str, Any]:
    q = np.zeros(6, dtype=float)
    gripper = 0.0
    total_reward = 0.0
    min_distance = float("inf")
    final_distance = float("inf")

    for step in range(max_steps):
        ee_xyz = compute_fk(model, q)[0]
        distance = float(np.linalg.norm(ee_xyz - target_xyz))
        min_distance = min(min_distance, distance)
        success = distance <= success_distance_m
        reward = -distance + (1.0 if success else 0.0)
        total_reward += reward
        final_distance = distance
        if success:
            return {
                "policy": policy_name,
                "success": True,
                "steps": step + 1,
                "final_distance_m": final_distance,
                "min_distance_m": min_distance,
                "total_reward": total_reward,
            }

        if policy_name == "sac_checkpoint":
            assert sac_policy is not None
            action = _sac_action(sac_policy, q, gripper, ee_xyz, target_xyz, device)
        elif policy_name == "ik_baseline":
            action = _ik_action(model, end_frame_id, q, target_xyz)
        elif policy_name == "random_baseline":
            action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        else:
            raise ValueError(policy_name)
        q, gripper = _apply_action(q, gripper, action)

    return {
        "policy": policy_name,
        "success": False,
        "steps": max_steps,
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "total_reward": total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--success-distance-m", type=float, default=0.015)
    args = parser.parse_args()

    model = load_robot_model()
    end_frame_id = get_end_effector_frame_id(model)
    sac_policy = _load_policy(args.dataset_root, args.checkpoint, args.device)
    targets = _sample_targets(args.episodes, args.seed)
    rng = np.random.default_rng(args.seed + 1)

    rows: list[dict[str, Any]] = []
    for episode_index, target in enumerate(targets):
        for policy_name in ("sac_checkpoint", "ik_baseline", "random_baseline"):
            result = _rollout(
                policy_name=policy_name,
                target_xyz=target,
                model=model,
                end_frame_id=end_frame_id,
                sac_policy=sac_policy,
                rng=rng,
                device=args.device,
                max_steps=args.max_steps,
                success_distance_m=args.success_distance_m,
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "target_x": target[0],
                    "target_y": target[1],
                    "target_z": target[2],
                    **result,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for policy_name in ("sac_checkpoint", "ik_baseline", "random_baseline"):
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        summary[policy_name] = {
            "episodes": len(policy_rows),
            "success_count": int(sum(bool(row["success"]) for row in policy_rows)),
            "success_rate": float(np.mean([bool(row["success"]) for row in policy_rows])),
            "mean_steps": float(np.mean([row["steps"] for row in policy_rows])),
            "mean_final_distance_m": float(np.mean([row["final_distance_m"] for row in policy_rows])),
            "mean_min_distance_m": float(np.mean([row["min_distance_m"] for row in policy_rows])),
            "mean_total_reward": float(np.mean([row["total_reward"] for row in policy_rows])),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
