#!/usr/bin/env python3
"""Validate reBot B601-DM Cartesian waypoints with reBotArm_control_py.

Run this from an environment where vectorBH6/reBotArm_control_py has been set up:

    cd /Users/fbsh/projects/reBotArm_control_py
    uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/validate_rebot_waypoints.py \
      --waypoints /Users/fbsh/projects/drtc-Phi/config/rebot_waypoints_smoke.yaml

The script does not talk to hardware. It loads the URDF/Pinocchio model, solves IK,
plans 50Hz Cartesian geodesic trajectories between accepted waypoints, and writes a CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin
import yaml

from reBotArm_control_py.kinematics import compute_fk, get_end_effector_frame_id, load_robot_model
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams, solve_ik
from reBotArm_control_py.trajectory import (
    IKParams as TrajIKParams,
    TrajPlanParams,
    TrajProfile,
    compute_traj_stats,
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
)


def _pose_from_waypoint(waypoint: dict[str, Any]) -> pin.SE3:
    xyz = np.asarray(waypoint["xyz"], dtype=float)
    rpy = waypoint.get("rpy", [0.0, 0.0, 0.0])
    if len(xyz) != 3:
        raise ValueError(f"Waypoint {waypoint.get('name')} xyz must have 3 values")
    if len(rpy) != 3:
        raise ValueError(f"Waypoint {waypoint.get('name')} rpy must have 3 values")
    return pin.SE3(pin.rpy.rpyToMatrix(float(rpy[0]), float(rpy[1]), float(rpy[2])), xyz)


def _load_waypoints(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("waypoints"), list):
        raise ValueError("Waypoint YAML must contain a top-level 'waypoints' list")
    for i, waypoint in enumerate(data["waypoints"]):
        if "name" not in waypoint:
            waypoint["name"] = f"waypoint_{i:03d}"
        if "xyz" not in waypoint:
            raise ValueError(f"Waypoint {waypoint['name']} is missing xyz")
    return data["waypoints"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("rebot_waypoint_validation.csv"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 50.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--ik-max-iter", type=int, default=300)
    parser.add_argument("--ik-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    waypoints = _load_waypoints(args.waypoints)
    model = load_robot_model()
    data = model.createData()
    end_frame_id = get_end_effector_frame_id(model)
    q_current = np.zeros(model.nq)
    t_current = compute_fk(model, q_current)[2]

    ik_params = IKParams(max_iter=args.ik_max_iter, tolerance=args.ik_tolerance, step_size=0.5, damping=1e-6)
    traj_params = TrajPlanParams(dt=args.dt, profile=TrajProfile.MIN_JERK, accel_ratio=0.25)
    traj_ik_params = TrajIKParams(max_iter=args.ik_max_iter, tolerance=args.ik_tolerance, step_size=0.8, damping=1e-6)

    rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for index, waypoint in enumerate(waypoints):
        name = str(waypoint["name"])
        target = _pose_from_waypoint(waypoint)
        ik_result = solve_ik(model, data, end_frame_id, target, q_current, ik_params)

        row: dict[str, Any] = {
            "index": index,
            "name": name,
            "x": target.translation[0],
            "y": target.translation[1],
            "z": target.translation[2],
            "ik_success": int(ik_result.success),
            "ik_error": f"{ik_result.error:.8e}",
            "ik_iterations": ik_result.iterations,
            "traj_success_rate": "",
            "traj_avg_error": "",
            "traj_max_error": "",
            "traj_points": "",
            "q_deg": json.dumps(np.round(np.degrees(ik_result.q), 4).tolist()),
            "accepted": 0,
        }

        if ik_result.success:
            cart = plan_cartesian_geodesic_trajectory(t_current, target, duration=args.duration, params=traj_params)
            joint_traj = track_trajectory(model, end_frame_id, cart.trajectory, q_current, traj_ik_params)
            stats = compute_traj_stats(model, end_frame_id, joint_traj, t_current, target, args.duration, traj_params)
            row.update(
                {
                    "traj_success_rate": f"{stats.success_rate:.6f}",
                    "traj_avg_error": f"{stats.avg_ik_error:.8e}",
                    "traj_max_error": f"{stats.max_ik_error:.8e}",
                    "traj_points": len(joint_traj),
                }
            )
            if stats.success_rate >= 0.99 and stats.max_ik_error <= 1e-3:
                row["accepted"] = 1
                q_current = ik_result.q.copy()
                t_current = target
                accepted.append({"name": name, "xyz": waypoint["xyz"], "rpy": waypoint.get("rpy", [0.0, 0.0, 0.0]), "q_deg": json.loads(row["q_deg"])})

        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"accepted_waypoints": accepted}, indent=2))

    print(f"wrote {args.output_csv}")
    if args.output_json is not None:
        print(f"wrote {args.output_json}")
    print(f"accepted={len(accepted)} total={len(rows)}")
    for row in rows:
        print(
            f"{row['name']}: ik={row['ik_success']} accepted={row['accepted']} "
            f"ik_error={row['ik_error']} traj_max={row['traj_max_error']}"
        )


if __name__ == "__main__":
    main()
