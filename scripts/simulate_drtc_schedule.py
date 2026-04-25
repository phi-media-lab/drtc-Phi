#!/usr/bin/env python3
"""Simulate DRTC action schedule stability under high remote latency.

This is a lightweight discrete-time simulator for choosing robust DRTC control
parameters before running a real robot. It models:

- local control ticks at a fixed fps,
- action chunks arriving after a fixed observation-to-action latency,
- cooldown-gated observation triggers,
- local execution from an action schedule,
- starvation when the schedule is empty.

It intentionally does not model action quality or RTC inpainting quality.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    fps: int
    actions_per_chunk: int
    s_min: int
    epsilon: int
    latency_ms: float
    duration_s: float
    initial_prefill_chunks: int


@dataclass(frozen=True)
class Result:
    fps: int
    actions_per_chunk: int
    s_min: int
    epsilon: int
    latency_ms: float
    duration_s: float
    initial_prefill_chunks: int
    latency_steps: int
    starvation_ticks: int
    starvation_ratio: float
    triggers: int
    chunks_arrived: int
    min_schedule: int
    mean_schedule: float
    max_schedule: int
    final_schedule: int


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def _parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one number")
    return values


def simulate(cfg: Config) -> Result:
    total_ticks = int(round(cfg.duration_s * cfg.fps))
    if total_ticks <= 0:
        raise ValueError("duration_s * fps must produce at least one tick")

    latency_steps = max(1, int(round(cfg.latency_ms / 1000.0 * cfg.fps)))
    trigger_threshold = cfg.actions_per_chunk - cfg.s_min
    schedule_size = cfg.initial_prefill_chunks * cfg.actions_per_chunk
    cooldown = 0

    arrivals: list[int] = []
    starvation_ticks = 0
    triggers = 0
    chunks_arrived = 0
    schedule_samples: list[int] = []

    for tick in range(total_ticks):
        # Arrivals are processed at the beginning of the tick.
        arrived_now = [arrival for arrival in arrivals if arrival <= tick]
        if arrived_now:
            chunks_arrived += len(arrived_now)
            schedule_size += len(arrived_now) * cfg.actions_per_chunk
            arrivals = [arrival for arrival in arrivals if arrival > tick]

        # Execute one local action if available.
        if schedule_size > 0:
            schedule_size -= 1
        else:
            starvation_ticks += 1

        schedule_samples.append(schedule_size)

        # Trigger observation/inference when the local schedule is below threshold.
        should_trigger = schedule_size <= trigger_threshold and cooldown == 0
        if should_trigger:
            triggers += 1
            arrivals.append(tick + latency_steps)
            cooldown = latency_steps + cfg.epsilon
        elif cooldown > 0:
            cooldown -= 1

    mean_schedule = sum(schedule_samples) / len(schedule_samples)
    return Result(
        fps=cfg.fps,
        actions_per_chunk=cfg.actions_per_chunk,
        s_min=cfg.s_min,
        epsilon=cfg.epsilon,
        latency_ms=cfg.latency_ms,
        duration_s=cfg.duration_s,
        initial_prefill_chunks=cfg.initial_prefill_chunks,
        latency_steps=latency_steps,
        starvation_ticks=starvation_ticks,
        starvation_ratio=starvation_ticks / total_ticks,
        triggers=triggers,
        chunks_arrived=chunks_arrived,
        min_schedule=min(schedule_samples),
        mean_schedule=mean_schedule,
        max_schedule=max(schedule_samples),
        final_schedule=schedule_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=_parse_csv_ints, default=[10, 15, 30])
    parser.add_argument("--actions-per-chunk", type=_parse_csv_ints, default=[50, 100])
    parser.add_argument("--s-min", type=_parse_csv_ints, default=[15, 25, 40])
    parser.add_argument("--epsilon", type=int, default=2)
    parser.add_argument("--latency-ms", type=_parse_csv_floats, default=[250, 350, 500, 750, 1000])
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--initial-prefill-chunks", type=int, default=1)
    parser.add_argument("--output", default="results/drtc_schedule_simulation.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Result] = []
    for fps in args.fps:
        for actions_per_chunk in args.actions_per_chunk:
            for s_min in args.s_min:
                if s_min >= actions_per_chunk:
                    continue
                for latency_ms in args.latency_ms:
                    cfg = Config(
                        fps=fps,
                        actions_per_chunk=actions_per_chunk,
                        s_min=s_min,
                        epsilon=args.epsilon,
                        latency_ms=latency_ms,
                        duration_s=args.duration_s,
                        initial_prefill_chunks=args.initial_prefill_chunks,
                    )
                    rows.append(simulate(cfg))

    fieldnames = list(Result.__dataclass_fields__.keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    # Print the most useful zero-starvation rows sorted by latency and schedule depth.
    stable = [row for row in rows if row.starvation_ticks == 0]
    stable.sort(key=lambda row: (row.latency_ms, row.fps, row.actions_per_chunk, row.mean_schedule))
    print(f"wrote {output_path}")
    print(f"stable_configs={len(stable)} total_configs={len(rows)}")
    print("top stable configs:")
    for row in stable[:20]:
        print(
            " ".join(
                [
                    f"fps={row.fps}",
                    f"H={row.actions_per_chunk}",
                    f"s_min={row.s_min}",
                    f"latency_ms={row.latency_ms:g}",
                    f"latency_steps={row.latency_steps}",
                    f"mean_schedule={row.mean_schedule:.1f}",
                    f"min_schedule={row.min_schedule}",
                    f"triggers={row.triggers}",
                ]
            )
        )

    unstable = [row for row in rows if row.starvation_ticks > 0]
    if unstable:
        unstable.sort(key=lambda row: (row.starvation_ratio, row.latency_ms))
        print("least bad unstable configs:")
        for row in unstable[:10]:
            print(
                " ".join(
                    [
                        f"fps={row.fps}",
                        f"H={row.actions_per_chunk}",
                        f"s_min={row.s_min}",
                        f"latency_ms={row.latency_ms:g}",
                        f"starvation_ratio={row.starvation_ratio:.3f}",
                        f"min_schedule={row.min_schedule}",
                    ]
                )
            )


if __name__ == "__main__":
    main()
