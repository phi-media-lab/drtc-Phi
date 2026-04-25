#!/usr/bin/env python3
"""Convert reBot sim episode JSONL to a local LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

ACTION_NAMES = [f"action_{i}" for i in range(7)]


def _state_from_observation(obs: dict[str, Any]) -> list[float]:
    q = list(obs["q"])
    return [float(x) for x in [*q[:6], obs["gripper"], *obs["ee_xyz"], *obs["target_xyz"]]]


def _action_or_zeros(action: list[float] | None) -> list[float]:
    if action is None:
        return [0.0] * 7
    return [float(x) for x in action[:7]]


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    for line in path.read_text().splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    if not episodes:
        raise ValueError(f"No episodes found in {path}")
    return episodes


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _stats_for_array(values: np.ndarray) -> dict[str, list[Any]]:
    return {
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


def _stats_for_scalar(series: pd.Series, dtype: str) -> dict[str, list[Any]]:
    values = series.to_numpy()
    if dtype == "bool":
        values = values.astype(np.int8)
    else:
        values = values.astype(np.float32)
    return {
        "max": [values.max().item()],
        "mean": [values.mean().item()],
        "min": [values.min().item()],
        "std": [values.std().item()],
        "count": [int(values.shape[0])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="phi/rebot-sim-smoke")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    episodes = _load_episodes(args.input_jsonl)
    rows: list[dict[str, Any]] = []
    episode_meta: list[dict[str, Any]] = []
    task_to_index: dict[str, int] = {}
    global_index = 0

    for episode_index, episode in enumerate(episodes):
        task_name = str(episode["summary"]["task"])
        if task_name not in task_to_index:
            task_to_index[task_name] = len(task_to_index)
        task_index = task_to_index[task_name]
        from_index = global_index

        for frame_index, step in enumerate(episode["steps"]):
            obs = step["observation"]
            action = _action_or_zeros(step["action"])
            row = {
                "observation.state": _state_from_observation(obs),
                "action": action,
                "next.reward": float(step["reward"]),
                "next.done": bool(frame_index == len(episode["steps"]) - 1),
                "success": bool(step["success"]),
                "distance_m": float(step["distance_m"]),
                "policy_type": str(episode["policy_type"]),
                "policy_info_json": json.dumps(step.get("policy_info"), separators=(",", ":")),
                "task": task_name,
                "task_index": task_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "timestamp": frame_index / args.fps,
                "index": global_index,
            }
            rows.append(row)
            global_index += 1

        episode_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [task_name],
                "length": len(episode["steps"]),
                "dataset_from_index": from_index,
                "dataset_to_index": global_index,
                "success": bool(episode["summary"]["success"]),
                "policy_type": str(episode["policy_type"]),
            }
        )

    output_root = args.output_root
    data_dir = output_root / "data" / "chunk-000"
    meta_dir = output_root / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    data_columns = [
        "observation.state",
        "action",
        "next.reward",
        "next.done",
        "success",
        "distance_m",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    data_df = df[data_columns].copy()
    parquet_path = data_dir / "file-000.parquet"
    data_df.to_parquet(parquet_path, index=False)

    tasks = [{"task_index": index, "task": task} for task, index in sorted(task_to_index.items(), key=lambda item: item[1])]
    tasks_df = pd.DataFrame({"task_index": [task["task_index"] for task in tasks]}, index=[task["task"] for task in tasks])
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    episodes_df = pd.DataFrame(
        [
            {
                **episode,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "data/chunk_index": 0,
                "data/file_index": 0,
            }
            for episode in episode_meta
        ]
    )
    episodes_df.to_parquet(episodes_dir / "file-000.parquet", index=False)

    _write_jsonl(meta_dir / "episodes.jsonl", episode_meta)
    _write_jsonl(meta_dir / "tasks.jsonl", tasks)

    info = {
        "codebase_version": "v3.0",
        "repo_id": args.repo_id,
        "robot_type": "rebot_b601_dm_sim",
        "fps": args.fps,
        "total_episodes": len(episodes),
        "total_frames": len(rows),
        "total_tasks": len(tasks),
        "total_videos": 0,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "splits": {"train": f"0:{len(rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [13],
                "names": STATE_NAMES,
            },
            "action": {"dtype": "float32", "shape": [7], "names": ACTION_NAMES},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "success": {"dtype": "bool", "shape": [1]},
            "distance_m": {"dtype": "float32", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    stats = {
        "observation.state": _stats_for_array(np.asarray(df["observation.state"].to_list(), dtype=np.float32)),
        "action": _stats_for_array(np.asarray(df["action"].to_list(), dtype=np.float32)),
        "next.reward": _stats_for_scalar(df["next.reward"], "float32"),
        "next.done": _stats_for_scalar(df["next.done"], "bool"),
        "success": _stats_for_scalar(df["success"], "bool"),
        "distance_m": _stats_for_scalar(df["distance_m"], "float32"),
        "timestamp": _stats_for_scalar(df["timestamp"], "float32"),
        "frame_index": _stats_for_scalar(df["frame_index"], "float32"),
        "episode_index": _stats_for_scalar(df["episode_index"], "float32"),
        "index": _stats_for_scalar(df["index"], "float32"),
        "task_index": _stats_for_scalar(df["task_index"], "float32"),
    }
    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    summary = {
        "output_root": str(output_root),
        "parquet_path": str(parquet_path),
        "tasks_path": str(meta_dir / "tasks.parquet"),
        "episodes_path": str(episodes_dir / "file-000.parquet"),
        "episodes": len(episodes),
        "frames": len(rows),
        "tasks": len(tasks),
    }
    (output_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(df.groupby(["policy_type", "episode_index"])["success"].max().to_string())


if __name__ == "__main__":
    main()
