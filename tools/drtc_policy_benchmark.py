"""Benchmark action-chunk inference latency for DRTC planning.

Three modes are supported:

1. Synthetic CUDA baseline (default): measures device-side tensor work without
   loading a LeRobot policy checkpoint.
2. Random ACT benchmark: instantiates a real ACT visual policy from scratch and
   times `predict_action_chunk` without requiring checkpoint downloads.
3. Real policy benchmark: loads a LeRobot policy from `from_pretrained`, builds
   a dummy observation, and times `predict_action_chunk`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class BenchResult:
    mode: str
    device: str
    warmup: int
    iters: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    cuda_available: bool
    cuda_device: str | None
    extra: dict[str, Any]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _summarize(
    *,
    mode: str,
    device: str,
    warmup: int,
    iters: int,
    times_ms: list[float],
    extra: dict[str, Any],
) -> BenchResult:
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    return BenchResult(
        mode=mode,
        device=device,
        warmup=warmup,
        iters=iters,
        mean_ms=float(statistics.fmean(times_ms)),
        median_ms=float(statistics.median(times_ms)),
        p90_ms=float(_percentile(times_ms, 90)),
        p95_ms=float(_percentile(times_ms, 95)),
        p99_ms=float(_percentile(times_ms, 99)),
        min_ms=float(min(times_ms)),
        max_ms=float(max(times_ms)),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        extra=extra,
    )


def run_synthetic(args: argparse.Namespace) -> BenchResult:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    hidden = int(args.synthetic_hidden)
    batch = int(args.batch_size)
    horizon = int(args.actions_per_chunk)
    action_dim = int(args.action_dim)

    w1 = torch.randn(hidden, hidden, device=device)
    w2 = torch.randn(hidden, action_dim, device=device)
    x = torch.randn(batch * horizon, hidden, device=device)

    def step() -> torch.Tensor:
        y = torch.nn.functional.silu(x @ w1)
        return y @ w2

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = step()
        _sync_if_cuda(device)

        times_ms: list[float] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = step()
            _sync_if_cuda(device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize(
        mode="synthetic",
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        times_ms=times_ms,
        extra={
            "batch_size": batch,
            "actions_per_chunk": horizon,
            "action_dim": action_dim,
            "synthetic_hidden": hidden,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    )


def run_random_act(args: argparse.Namespace) -> BenchResult:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
    }
    for idx in range(args.camera_count):
        input_features[f"observation.images.camera{idx}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, args.camera_height, args.camera_width),
        )

    config = ACTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))},
        device=args.device,
        chunk_size=args.actions_per_chunk,
        n_action_steps=args.actions_per_chunk,
        pretrained_backbone_weights=None,
        dim_model=args.act_dim_model,
        n_heads=args.act_heads,
        dim_feedforward=args.act_dim_feedforward,
        n_encoder_layers=args.act_encoder_layers,
        n_decoder_layers=args.act_decoder_layers,
        use_vae=False,
        dropout=0.0,
    )

    load_start = time.perf_counter()
    policy = ACTPolicy(config).to(args.device)
    policy.eval()
    build_ms = (time.perf_counter() - load_start) * 1000.0

    obs: dict[str, torch.Tensor] = {
        OBS_STATE: torch.zeros(args.batch_size, args.state_dim, device=args.device),
    }
    for key, feat in config.image_features.items():
        c, h, w = feat.shape
        obs[key] = torch.zeros(args.batch_size, c, h, w, dtype=torch.float32, device=args.device)

    def step() -> torch.Tensor:
        return policy.predict_action_chunk(obs)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = step()
        _sync_if_cuda(args.device)

        times_ms: list[float] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = step()
            _sync_if_cuda(args.device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize(
        mode="random-act",
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        times_ms=times_ms,
        extra={
            "build_ms": build_ms,
            "batch_size": args.batch_size,
            "actions_per_chunk": args.actions_per_chunk,
            "action_dim": args.action_dim,
            "state_dim": args.state_dim,
            "camera_count": args.camera_count,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "act_dim_model": args.act_dim_model,
            "act_heads": args.act_heads,
            "act_dim_feedforward": args.act_dim_feedforward,
            "act_encoder_layers": args.act_encoder_layers,
            "act_decoder_layers": args.act_decoder_layers,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    )


def _dummy_observation(policy: Any, *, state_dim: int, task: str, device: str) -> dict[str, Any]:
    obs: dict[str, Any] = {"observation.state": torch.zeros(1, state_dim, device=device)}

    image_features = getattr(policy.config, "image_features", {}) or {}
    for key, feat in image_features.items():
        c, h, w = feat.shape
        obs[key] = torch.zeros(1, c, h, w, dtype=torch.float32, device=device)

    obs["task"] = task
    return obs


def _infer_horizon(policy_config: Any) -> int | None:
    for field_name in ("chunk_size", "n_action_steps", "horizon"):
        value = getattr(policy_config, field_name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def run_policy(args: argparse.Namespace) -> BenchResult:
    if not args.policy_type or not args.pretrained_name_or_path:
        raise ValueError("--policy-type and --pretrained-name-or-path are required for --mode policy")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_class = get_policy_class(args.policy_type)

    load_start = time.perf_counter()
    policy_config = None
    if args.vlm_model_name is not None:
        from lerobot.configs.policies import PreTrainedConfig

        policy_config = PreTrainedConfig.from_pretrained(args.pretrained_name_or_path)
        if not hasattr(policy_config, "vlm_model_name"):
            raise ValueError(f"--vlm-model-name is not supported by policy type {args.policy_type}")
        policy_config.vlm_model_name = args.vlm_model_name
    if args.load_vlm_weights is not None:
        if policy_config is None:
            from lerobot.configs.policies import PreTrainedConfig

            policy_config = PreTrainedConfig.from_pretrained(args.pretrained_name_or_path)
        if not hasattr(policy_config, "load_vlm_weights"):
            raise ValueError(f"--load-vlm-weights is not supported by policy type {args.policy_type}")
        policy_config.load_vlm_weights = args.load_vlm_weights
    if args.no_act_pretrained_backbone_weights:
        if policy_config is None:
            from lerobot.configs.policies import PreTrainedConfig

            policy_config = PreTrainedConfig.from_pretrained(args.pretrained_name_or_path)
        if not hasattr(policy_config, "pretrained_backbone_weights"):
            raise ValueError(
                f"--no-act-pretrained-backbone-weights is not supported by policy type {args.policy_type}"
            )
        policy_config.pretrained_backbone_weights = None
    policy = policy_class.from_pretrained(args.pretrained_name_or_path, config=policy_config)
    policy.to(args.device)
    policy.eval()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    cfg_obj = getattr(policy, "config", None)
    if args.num_flow_matching_steps is not None and cfg_obj is not None:
        if hasattr(cfg_obj, "num_inference_steps"):
            cfg_obj.num_inference_steps = args.num_flow_matching_steps
        elif hasattr(cfg_obj, "num_steps"):
            cfg_obj.num_steps = args.num_flow_matching_steps

    preprocess_start = time.perf_counter()
    device_override = {"device": args.device}
    preprocessor_overrides = {"device_processor": device_override}
    if args.vlm_model_name is not None:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": args.vlm_model_name}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.pretrained_name_or_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides={"device_processor": device_override},
    )
    processor_ms = (time.perf_counter() - preprocess_start) * 1000.0

    obs = _dummy_observation(policy, state_dim=args.state_dim, task=args.task, device=args.device)
    obs = preprocessor(obs)

    requested_horizon = args.actions_per_chunk
    model_horizon = _infer_horizon(policy.config)

    def step() -> torch.Tensor:
        action_tensor = policy.predict_action_chunk(obs)
        if action_tensor.ndim != 3:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor[:, :requested_horizon, :]
        if not args.skip_postprocess:
            b, t, a = action_tensor.shape
            flat = postprocessor(action_tensor.reshape(b * t, a))
            action_tensor = flat.reshape(b, t, flat.shape[-1])
        return action_tensor

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = step()
        _sync_if_cuda(args.device)

        times_ms: list[float] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = step()
            _sync_if_cuda(args.device)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    return _summarize(
        mode="policy",
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        times_ms=times_ms,
        extra={
            "policy_type": args.policy_type,
            "pretrained_name_or_path": args.pretrained_name_or_path,
            "load_ms": load_ms,
            "processor_ms": processor_ms,
            "model_horizon": model_horizon,
            "actions_per_chunk": requested_horizon,
            "state_dim": args.state_dim,
            "num_flow_matching_steps": args.num_flow_matching_steps,
            "skip_postprocess": args.skip_postprocess,
            "vlm_model_name": args.vlm_model_name,
            "load_vlm_weights": args.load_vlm_weights,
            "no_act_pretrained_backbone_weights": args.no_act_pretrained_backbone_weights,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "random-act", "policy"), default="synthetic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--synthetic-hidden", type=int, default=2048)
    parser.add_argument("--camera-count", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--act-dim-model", type=int, default=512)
    parser.add_argument("--act-heads", type=int, default=8)
    parser.add_argument("--act-dim-feedforward", type=int, default=3200)
    parser.add_argument("--act-encoder-layers", type=int, default=4)
    parser.add_argument("--act-decoder-layers", type=int, default=1)
    parser.add_argument("--policy-type")
    parser.add_argument("--pretrained-name-or-path")
    parser.add_argument("--vlm-model-name")
    parser.add_argument("--load-vlm-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-act-pretrained-backbone-weights", action="store_true")
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--task", default="benchmark task")
    parser.add_argument("--num-flow-matching-steps", type=int, default=None)
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.mode == "synthetic":
        result = run_synthetic(args)
    elif args.mode == "random-act":
        result = run_random_act(args)
    else:
        result = run_policy(args)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return

    print("DRTC policy benchmark")
    for key, value in asdict(result).items():
        if key == "extra":
            continue
        print(f"  {key}: {value}")
    print("  extra:")
    for key, value in result.extra.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
