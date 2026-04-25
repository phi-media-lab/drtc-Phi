#!/usr/bin/env python3
"""Generate a small multi-episode reBot kinematic simulation dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BASE_CONFIG: dict[str, Any] = {
    "task": {
        "name": "placeholder",
        "max_steps": 80,
        "dt": 0.0666666667,
        "success_distance_m": 0.015,
        "target_xyz": [0.30, 0.10, 0.25],
    },
    "initial_state": {
        "q_deg": [0, 0, 0, 0, 0, 0],
        "gripper": 0.0,
    },
    "action_adapter": {
        "joint_delta_scale_rad": 0.035,
        "gripper_delta_scale": 0.05,
        "max_abs_joint_rad": 2.8,
        "min_gripper": 0.0,
        "max_gripper": 1.0,
    },
    "policy": {
        "type": "mock_proportional_ik",
        "ik_target_xyz": [0.30, 0.10, 0.25],
        "action_clip": 1.0,
        "random_seed": 0,
    },
}


def _sample_target(rng: np.random.Generator) -> list[float]:
    # Conservative workspace around previously validated reachable targets.
    return [
        float(rng.uniform(0.24, 0.32)),
        float(rng.uniform(-0.08, 0.12)),
        float(rng.uniform(0.21, 0.29)),
    ]


def _episode_config(*, policy_type: str, target_xyz: list[float], seed: int, index: int) -> dict[str, Any]:
    cfg = json.loads(json.dumps(BASE_CONFIG))
    suffix = "ik" if policy_type == "mock_proportional_ik" else "random"
    cfg["task"]["name"] = f"rebot_batch_{suffix}_{index:03d}"
    cfg["task"]["target_xyz"] = target_xyz
    cfg["policy"]["type"] = policy_type
    cfg["policy"]["random_seed"] = seed
    cfg["policy"]["ik_target_xyz"] = target_xyz
    if policy_type == "mock_random":
        cfg["task"]["success_distance_m"] = 0.015
    return cfg


def _run_episode(
    *,
    sim_script: Path,
    cfg: dict[str, Any],
    output_dir: Path,
    export_jsonl: Path,
    index: int,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        config_path = Path(f.name)

    policy_type = cfg["policy"]["type"]
    stem = f"{index:03d}_{policy_type}"
    output_csv = output_dir / f"{stem}.csv"
    output_json = output_dir / f"{stem}.json"
    cmd = [
        sys.executable,
        str(sim_script),
        "--config",
        str(config_path),
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
        "--export-jsonl",
        str(export_jsonl),
    ]
    try:
        subprocess.run(cmd, check=True)
        return json.loads(output_json.read_text())
    finally:
        config_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-jsonl", required=True, type=Path)
    parser.add_argument("--ik-episodes", type=int, default=32)
    parser.add_argument("--random-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.export_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.export_jsonl.unlink(missing_ok=True)

    summaries = []
    episode_index = 0
    for _ in range(args.ik_episodes):
        target = _sample_target(rng)
        cfg = _episode_config(
            policy_type="mock_proportional_ik",
            target_xyz=target,
            seed=int(rng.integers(0, 2**31 - 1)),
            index=episode_index,
        )
        summaries.append(
            _run_episode(
                sim_script=args.sim_script,
                cfg=cfg,
                output_dir=args.output_dir,
                export_jsonl=args.export_jsonl,
                index=episode_index,
            )
        )
        episode_index += 1

    for _ in range(args.random_episodes):
        target = _sample_target(rng)
        cfg = _episode_config(
            policy_type="mock_random",
            target_xyz=target,
            seed=int(rng.integers(0, 2**31 - 1)),
            index=episode_index,
        )
        summaries.append(
            _run_episode(
                sim_script=args.sim_script,
                cfg=cfg,
                output_dir=args.output_dir,
                export_jsonl=args.export_jsonl,
                index=episode_index,
            )
        )
        episode_index += 1

    success_count = sum(1 for summary in summaries if summary["success"])
    total_steps = sum(int(summary["steps"]) for summary in summaries)
    summary = {
        "episodes": len(summaries),
        "success_count": success_count,
        "failure_count": len(summaries) - success_count,
        "total_steps": total_steps,
        "export_jsonl": str(args.export_jsonl),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
