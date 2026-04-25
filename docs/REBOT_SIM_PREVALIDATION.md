# reBot B601-DM Simulation Prevalidation

Last updated: 2026-04-24

This note records whether we can validate reBot B601-DM motion plans before having physical hardware.

Short answer: yes, partially.

The `vectorBH6/reBotArm_control_py` repository provides a hardware-independent Pinocchio + MeshCat path for:

- URDF loading.
- Forward kinematics.
- Inverse kinematics.
- SE(3) geodesic trajectory planning.
- CLIK trajectory tracking.
- MeshCat visualization.

It does not validate:

- MotorBridge hardware communication.
- Damiao USB2CAN behavior.
- Real calibration offsets.
- Joint friction/backlash/gravity compensation quality.
- Camera-to-arm hand-eye calibration.
- Gripper-food physical interaction.

## Source Repositories and Docs

- GitHub: https://github.com/vectorBH6/reBotArm_control_py
- Seeed Wiki: https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/

The repository describes itself as a Pinocchio kinematics/dynamics and MotorBridge control library for reBotArm.
It supports B601-DM and B601-RS, includes URDF assets, and has examples under:

```text
example/sim/fk_sim.py
example/sim/ik_sim.py
example/sim/traj_sim.py
example/sim/visualizer.py
```

## Local Clone

Local path used for validation:

```text
/Users/fbsh/projects/reBotArm_control_py
```

Clone and setup:

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git /Users/fbsh/projects/reBotArm_control_py
cd /Users/fbsh/projects/reBotArm_control_py
uv sync
```

`uv sync` succeeded locally with Python `3.10.17` and installed `pin==3.9.0`, `meshcat==0.3.2`, `motorbridge==0.1.7`, and related dependencies.

## Verified Non-Hardware Smoke

Command run locally:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
uv run python - <<'PY'
import numpy as np
from reBotArm_control_py.kinematics import compute_fk, load_robot_model, get_end_effector_frame_id
from reBotArm_control_py.kinematics.inverse_kinematics import compute_ik, IKParams
from reBotArm_control_py.trajectory import (
    plan_cartesian_geodesic_trajectory,
    track_trajectory,
    compute_traj_stats,
    TrajPlanParams,
    TrajProfile,
)

model = load_robot_model()
end_frame_id = get_end_effector_frame_id(model)
q0 = np.zeros(model.nq)
pos0, rot0, T0 = compute_fk(model, q0)
print("model_nq", model.nq)
print("end_frame_id", end_frame_id)
print("fk_zero_pos_m", np.round(pos0, 5).tolist())

for target in ([0.25, 0.0, 0.25], [0.30, 0.10, 0.25], [0.20, -0.10, 0.20]):
    result = compute_ik(
        q0,
        np.array(target),
        params=IKParams(max_iter=300, tolerance=1e-4, step_size=0.5, damping=1e-6),
    )
    print(
        "ik_target",
        target,
        "success",
        result.success,
        "error",
        f"{result.error:.3e}",
        "iters",
        result.iterations,
        "q_deg",
        np.round(np.degrees(result.q), 2).tolist(),
    )

q1 = compute_ik(q0, np.array([0.25, 0.0, 0.25])).q
T1 = compute_fk(model, q1)[2]
params = TrajPlanParams(dt=1 / 50, profile=TrajProfile.MIN_JERK, accel_ratio=0.25)
cart = plan_cartesian_geodesic_trajectory(T0, T1, duration=2.0, params=params)
joint_traj = track_trajectory(model, end_frame_id, cart.trajectory, q0)
stats = compute_traj_stats(model, end_frame_id, joint_traj, T0, T1, duration=2.0, params=params)
print("traj_points", len(joint_traj))
print("traj_success_rate", f"{stats.success_rate:.3f}")
print("traj_avg_err", f"{stats.avg_ik_error:.3e}")
print("traj_max_err", f"{stats.max_ik_error:.3e}")
PY
```

Observed output:

```text
model_nq 6
end_frame_id 15
fk_zero_pos_m [0.26031, 0.0, 0.1917]
ik_target [0.25, 0.0, 0.25] success True error 5.840e-05 iters 10 q_deg [-0.0, -4.69, -13.53, 8.84, -0.0, 0.01]
ik_target [0.3, 0.1, 0.25] success True error 5.733e-05 iters 11 q_deg [40.6, -47.11, -15.5, -31.61, 40.6, 0.01]
ik_target [0.2, -0.1, 0.2] success False error 1.171e-01 iters 300 q_deg [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
traj_points 101
traj_success_rate 1.000
traj_avg_err 4.516e-05
traj_max_err 9.879e-05
```

Interpretation:

- The Pinocchio model loads with `6` controllable joints.
- Zero-pose end-effector position is approximately `[0.26031, 0.0, 0.1917] m`.
- Some reasonable workspace targets solve cleanly.
- Some targets fail IK even if they look nearby, so we need a precomputed reachable waypoint set.
- A 2s, 50Hz Cartesian trajectory from zero pose to `[0.25, 0.0, 0.25]` tracked successfully with low error.

## MeshCat Interactive Simulators

The repository provides interactive visual tools:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
uv run python example/sim/fk_sim.py
uv run python example/sim/ik_sim.py
uv run python example/sim/traj_sim.py
```

Expected behavior:

- A MeshCat URL is printed.
- Opening it in a browser shows the URDF robot model.
- `fk_sim.py` accepts six joint angles in degrees.
- `ik_sim.py` accepts `x y z` or `x y z roll pitch yaw`.
- `traj_sim.py` accepts target pose and animates planned trajectory.

These are suitable for visual inspection of proposed cooking waypoints before touching hardware.

## How This Helps the Hackathon Plan

Use this simulation path before the event to define conservative primitives:

- Home pose.
- Safe hover pose.
- Ingredient pick approach pose.
- Ingredient pick contact pose.
- Bowl/pot place pose.
- Pour pre-pose.
- Pour tilted pose.
- Stir start pose.
- Stir loop waypoints.
- Retreat pose.

Each waypoint should be validated for:

- IK convergence.
- Joint angle range.
- Trajectory tracking success.
- Smoothness and duration.
- No obvious self-collision in MeshCat.

This does not replace real-hardware testing. It reduces the probability of choosing impossible or dangerous target poses during the 48-hour event.

## Recommended Next Step

Create a small waypoint validation script that:

1. Loads a YAML file of named Cartesian waypoints.
2. Runs IK from the previous accepted waypoint.
3. Plans a 50Hz trajectory.
4. Emits CSV with success, error, joint angle min/max, and trajectory point count.
5. Optionally exports a JSON primitive plan for the event machine.

This should be done before writing a reBot DRTC policy adapter. The first robot-facing asset should be a safe primitive library.

## Waypoint Batch Validator

Implemented in this repo:

```text
scripts/hackathon/validate_rebot_waypoints.py
config/rebot_waypoints_smoke.yaml
```

Run from the cloned `reBotArm_control_py` repo:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/validate_rebot_waypoints.py \
  --waypoints /Users/fbsh/projects/drtc-Phi/config/rebot_waypoints_smoke.yaml \
  --output-csv /Users/fbsh/projects/drtc-Phi/results/rebot_waypoints_smoke_validation.csv \
  --output-json /Users/fbsh/projects/drtc-Phi/results/rebot_waypoints_smoke_accepted.json
```

Observed result:

```text
accepted=2 total=3
zero_reachable_lift: ik=1 accepted=1 ik_error=5.83964251e-05 traj_max=9.88836754e-05
front_right_hover: ik=1 accepted=1 ik_error=6.13775942e-05 traj_max=9.92113779e-05
known_bad_left_low_probe: ik=0 accepted=0 ik_error=1.38390434e-01 traj_max=
```

Generated outputs:

```text
results/rebot_waypoints_smoke_validation.csv
results/rebot_waypoints_smoke_accepted.json
```

This gives us a concrete pre-event workflow: define candidate cooking waypoints in YAML, validate IK and trajectory tracking offline, then bring only accepted conservative waypoints to the hardware test.
