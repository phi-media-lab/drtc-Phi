#!/usr/bin/env python3
"""Run a minimal SAC learner dry-run on the reBot LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.configuration_sac import ActorNetworkConfig, CriticNetworkConfig, SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION, OBS_STATE


def _load_min_max(dataset_root: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text())
    state_min = stats[OBS_STATE]["min"]
    state_max = stats[OBS_STATE]["max"]
    action_min = stats[ACTION]["min"]
    action_max = stats[ACTION]["max"]
    return state_min, state_max, action_min, action_max


def _make_policy(dataset_root: Path, device: str) -> SACPolicy:
    state_min, state_max, action_min, action_max = _load_min_max(dataset_root)
    state_dim = len(state_min)
    action_dim = len(action_min)

    cfg = SACConfig(
        device=device,
        storage_device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
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
    return SACPolicy(config=cfg).to(device)


def _make_optimizers(policy: SACPolicy) -> dict[str, torch.optim.Optimizer]:
    actor_params = [
        p
        for name, p in policy.actor.named_parameters()
        if not policy.config.shared_encoder or not name.startswith("encoder")
    ]
    return {
        "actor": torch.optim.Adam(actor_params, lr=policy.config.actor_lr),
        "critic": torch.optim.Adam(policy.critic_ensemble.parameters(), lr=policy.config.critic_lr),
        "temperature": torch.optim.Adam([policy.log_alpha], lr=policy.config.temperature_lr),
    }


def _step_optimizer(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="phi/rebot-sim-smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage-device", default="cpu")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--steps", default=3, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    replay_buffer = ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        device=args.device,
        storage_device=args.storage_device,
        state_keys=[OBS_STATE],
        capacity=len(dataset),
        use_drq=False,
        optimize_memory=True,
    )
    policy = _make_policy(args.dataset_root, args.device)
    policy.train()
    optimizers = _make_optimizers(policy)
    metrics_rows: list[dict[str, float | int | str]] = []

    print(f"dataset_len={len(dataset)}")
    print(f"buffer_len={len(replay_buffer)}")
    print(f"device={args.device}")

    for step in range(args.steps):
        batch = replay_buffer.sample(args.batch_size)

        critic_loss = policy.forward(batch, model="critic")["loss_critic"]
        critic_loss_value = _step_optimizer(critic_loss, optimizers["critic"])

        actor_loss = policy.forward(batch, model="actor")["loss_actor"]
        actor_loss_value = _step_optimizer(actor_loss, optimizers["actor"])

        temperature_loss = policy.forward(batch, model="temperature")["loss_temperature"]
        temperature_loss_value = _step_optimizer(temperature_loss, optimizers["temperature"])

        policy.update_target_networks()

        with torch.no_grad():
            selected_action = policy.select_action(batch["state"])

        print(
            "step={step} loss_critic={critic:.6f} loss_actor={actor:.6f} "
            "loss_temperature={temperature:.6f} selected_action_shape={shape}".format(
                step=step,
                critic=critic_loss_value,
                actor=actor_loss_value,
                temperature=temperature_loss_value,
                shape=tuple(selected_action.shape),
            )
        )
        metrics_rows.append(
            {
                "step": step,
                "loss_critic": critic_loss_value,
                "loss_actor": actor_loss_value,
                "loss_temperature": temperature_loss_value,
                "selected_action_abs_mean": float(selected_action.detach().abs().mean().cpu().item()),
            }
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "metrics.jsonl"
        with metrics_path.open("w") as f:
            for row in metrics_rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        torch.save(policy.state_dict(), args.output_dir / "sac_policy_state.pt")
        torch.save({name: opt.state_dict() for name, opt in optimizers.items()}, args.output_dir / "optimizers.pt")
        summary = {
            "dataset_root": str(args.dataset_root),
            "dataset_len": len(dataset),
            "buffer_len": len(replay_buffer),
            "device": args.device,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "metrics_path": str(metrics_path),
            "checkpoint_path": str(args.output_dir / "sac_policy_state.pt"),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote_output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
