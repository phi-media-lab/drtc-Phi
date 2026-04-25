#!/usr/bin/env python3
"""Validate that a reBot LeRobot dataset can populate an Evo-RL ReplayBuffer."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="phi/rebot-sim-smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--storage-device", default="cpu")
    parser.add_argument("--batch-size", default=8, type=int)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    state_keys = ["observation.state"]
    buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        device=args.device,
        storage_device=args.storage_device,
        state_keys=state_keys,
        capacity=len(dataset),
        use_drq=False,
        optimize_memory=True,
    )
    batch = buffer.sample(args.batch_size)

    print(f"dataset_len={len(dataset)}")
    print(f"buffer_len={len(buffer)}")
    print(f"state_keys={state_keys}")
    print(f"state_shape={tuple(batch['state']['observation.state'].shape)}")
    print(f"next_state_shape={tuple(batch['next_state']['observation.state'].shape)}")
    print(f"action_shape={tuple(batch['action'].shape)}")
    print(f"reward_shape={tuple(batch['reward'].shape)}")
    done_indices = buffer.dones[: len(buffer)].nonzero(as_tuple=False).flatten().tolist()
    print(f"buffer_done_count={len(done_indices)}")
    print(f"buffer_done_indices={done_indices}")
    print(f"done_sum={float(batch['done'].sum().item())}")
    print(f"reward_mean={float(batch['reward'].mean().item()):.6f}")
    print(f"action_abs_mean={float(batch['action'].abs().mean().item()):.6f}")


if __name__ == "__main__":
    main()
