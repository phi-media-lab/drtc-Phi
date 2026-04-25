# reBot Sim Policy/RL Loop

Last updated: 2026-04-24

This document records the first complete local plumbing test for:

```text
constructed observation -> policy -> action adapter -> reBot kinematic simulator -> reward/success log
```

This is not a physics or cooking-quality validation. It is a system-loop test that prepares the interfaces needed to later connect:

- MI300X DRTC policy serving.
- LeRobot dataset generation.
- Evo-RL-Phi ACP/value/policy training.

## Implemented Files

```text
scripts/hackathon/sim_rebot_policy_loop.py
config/rebot_sim_task_smoke.yaml
config/rebot_sim_task_random.yaml
config/rebot_sim_task_drtc.yaml
results/rebot_sim_policy_loop_mock.csv
results/rebot_sim_policy_loop_mock_summary.json
results/rebot_sim_policy_loop_random.csv
results/rebot_sim_policy_loop_random_summary.json
results/rebot_sim_policy_loop_drtc.csv
results/rebot_sim_policy_loop_drtc_summary.json
results/rebot_sim_episodes.jsonl
results/rebot_sim_lerobot_like_dataset/
```

## Current Loop

The current script:

1. Loads the reBot B601-DM Pinocchio model from `reBotArm_control_py`.
2. Initializes six robot joints plus one scalar gripper state.
3. Constructs an observation containing:
   - step
   - task name
   - joint state
   - gripper
   - current end-effector XYZ
   - target XYZ
4. Runs a mock `proportional_ik` policy.
5. Emits a 7D action vector:
   - first 6 values: joint deltas
   - last value: gripper delta
6. Applies an action adapter with scaling and clipping.
7. Computes FK for the next state.
8. Computes reward and success from end-effector distance to target.
9. Writes per-step CSV and summary JSON.
10. Optionally appends one full episode record to JSONL with `--export-jsonl`.

The script now has a small policy backend abstraction:

- `mock_proportional_ik`: stable deterministic baseline.
- `mock_random`: negative-control baseline that should usually fail but still produce valid episodes.
- `drtc`: connects to the MI300X DRTC server, receives one `ActionsDense` chunk, and consumes it through the same reBot kinematic loop.

## Run Command

Run from the `reBotArm_control_py` clone:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/sim_rebot_policy_loop.py \
  --config /Users/fbsh/projects/drtc-Phi/config/rebot_sim_task_smoke.yaml \
  --output-csv /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_mock.csv \
  --output-json /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_mock_summary.json
```

Observed result:

```json
{
  "task": "reach_front_right_hover",
  "steps": 25,
  "success": true,
  "final_distance_m": 0.013568738111069746,
  "total_reward": -0.5961811153604618,
  "output_csv": "/Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_mock.csv"
}
```

Random-policy negative control:

```bash
cp /Users/fbsh/projects/drtc-Phi/config/rebot_sim_task_smoke.yaml /tmp/rebot_sim_task_random.yaml
perl -0pi -e 's/type: mock_proportional_ik/type: mock_random/' /tmp/rebot_sim_task_random.yaml
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/sim_rebot_policy_loop.py \
  --config /tmp/rebot_sim_task_random.yaml \
  --output-csv /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_random.csv \
  --output-json /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_random_summary.json
```

Observed result:

```json
{
  "task": "reach_front_right_hover",
  "steps": 80,
  "success": false,
  "final_distance_m": 0.1729222562270507,
  "total_reward": -14.464378583153618,
  "output_csv": "/Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_random.csv"
}
```

DRTC policy plumbing test:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py:/Users/fbsh/projects/drtc-Phi/src \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/sim_rebot_policy_loop.py \
  --config /Users/fbsh/projects/drtc-Phi/config/rebot_sim_task_drtc.yaml \
  --output-csv /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_drtc.csv \
  --output-json /Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_drtc_summary.json
```

Observed result:

```json
{
  "task": "drtc_chunk_to_rebot_kinematic_sim",
  "steps": 50,
  "success": false,
  "final_distance_m": 0.11467729605766903,
  "total_reward": -5.7515617147172495,
  "output_csv": "/Users/fbsh/projects/drtc-Phi/results/rebot_sim_policy_loop_drtc.csv"
}
```

Important first-row policy metadata:

```json
{
  "backend": "drtc",
  "received_actions": 50,
  "action_dim": 7,
  "source_control_step": 1777038179564,
  "rtt_ms": 1970.883846282959
}
```

Interpretation:

- The complete path now works: constructed observation -> MI300X DRTC server -> `50 x 7` action chunk -> safe clipped action adapter -> reBot Pinocchio state updates -> reward log.
- The DRTC run did not solve the reaching task. That is expected because the loaded checkpoint is not trained for this synthetic reBot task and the `7D` actions are only interpreted as clipped joint deltas for plumbing.
- This validates the interface path, not action quality.

## Episode JSONL Export

The script can append full episode records to a JSONL file:

```bash
--export-jsonl /Users/fbsh/projects/drtc-Phi/results/rebot_sim_episodes.jsonl
```

The current JSONL contains three episodes:

```text
0 mock_proportional_ik reach_front_right_hover steps 25 success True final_distance 0.013568738111069746
1 mock_random reach_front_right_hover_random_baseline steps 80 success False final_distance 0.1729222562270507
2 drtc drtc_chunk_to_rebot_kinematic_sim steps 50 success False final_distance 0.11123447584416511
```

Each JSONL record contains:

- `schema_version`
- `created_unix_s`
- `config_path`
- `policy_type`
- `task`
- `action_adapter`
- `initial_state`
- `summary`
- `steps`

Each step contains:

- `observation`
- `action`
- `reward`
- `success`
- `distance_m`
- `policy_info`

This is the first bridge artifact toward an RL/ACP pipeline: the loop now produces explicit success and failure episodes, including a DRTC-generated trajectory sample.

## LeRobot-Like Dataset Export

Implemented converter:

```text
scripts/hackathon/export_rebot_sim_jsonl_to_dataset.py
```

Run:

```bash
/Users/fbsh/projects/drtc-Phi/.venv/bin/python \
  /Users/fbsh/projects/drtc-Phi/scripts/hackathon/export_rebot_sim_jsonl_to_dataset.py \
  --input-jsonl /Users/fbsh/projects/drtc-Phi/results/rebot_sim_episodes.jsonl \
  --output-root /Users/fbsh/projects/drtc-Phi/results/rebot_sim_lerobot_like_dataset \
  --repo-id phi/rebot-sim-smoke \
  --fps 15
```

Generated structure:

```text
results/rebot_sim_lerobot_like_dataset/
  conversion_summary.json
  data/chunk-000/file-000.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/episodes.jsonl
  meta/tasks.jsonl
```

Dataset summary:

```text
episodes=3
frames=155
tasks=3
```

Per-episode summary from the parquet:

```text
policy_type           episode_index  frames  success  reward      final_distance
drtc                  2              50      False    -5.729341   0.111234
mock_proportional_ik  0              25      True     -0.596181   0.013569
mock_random           1              80      False    -14.464379  0.172922
```

Main parquet columns:

- `observation.state`: 13D vector `[q0..q5, gripper, ee_xyz, target_xyz]`
- `action`: 7D vector
- `next.reward`
- `next.done`
- `success`
- `distance_m`
- `task_index`
- `episode_index`
- `frame_index`
- `timestamp`
- `index`

Debug provenance remains in `results/rebot_sim_episodes.jsonl` and episode metadata; the data parquet is kept strict so `LeRobotDataset` can cast it against `meta/info.json`.

## MI300X Evo-RL Loader and ReplayBuffer Validation

Validated on MI300X in:

```text
/mnt/models_alehe/phi-fbsh/Evo-RL-Phi
/mnt/models_alehe/phi-fbsh/Evo-RL-Phi/.venvs/pi05-openpi-ssp
```

Torch/ROCm device:

```text
torch_version=2.6.0+rocm7.0.2.git0568140d
cuda_available=True
device_count=1
device0=AMD Instinct MI300X VF
```

Dataset loader result:

```text
meta OK v3.0 3 155 3
dataset OK len 155
```

ReplayBuffer smoke script:

```text
scripts/hackathon/validate_rebot_dataset_replay_buffer.py
```

Run on MI300X:

```bash
cd /mnt/models_alehe/phi-fbsh/Evo-RL-Phi
.venvs/pi05-openpi-ssp/bin/python \
  /mnt/models_alehe/phi-fbsh/drtc-Phi/scripts/hackathon/validate_rebot_dataset_replay_buffer.py \
  --dataset-root /mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sim_lerobot_like_dataset \
  --repo-id phi/rebot-sim-smoke \
  --device cuda:0 \
  --storage-device cpu \
  --batch-size 8
```

Observed result:

```text
dataset_len=155
buffer_len=155
state_keys=['observation.state']
state_shape=(8, 13)
next_state_shape=(8, 13)
action_shape=(8, 7)
reward_shape=(8,)
buffer_done_count=3
buffer_done_indices=[24, 104, 154]
```

This confirms:

- `LeRobotDataset` v3.0 can load the exported dataset.
- `ReplayBuffer.from_lerobot_dataset` can convert all 155 frames into RL transitions.
- Episode termination boundaries are correct for the 25/80/50 frame episodes.
- Sampled batches can be moved to MI300X through the Torch ROCm `cuda:0` device.

Minimal SAC learner dry-run script:

```text
scripts/hackathon/validate_rebot_sac_dry_run.py
```

Run on MI300X:

```bash
cd /mnt/models_alehe/phi-fbsh/Evo-RL-Phi
.venvs/pi05-openpi-ssp/bin/python \
  /mnt/models_alehe/phi-fbsh/drtc-Phi/scripts/hackathon/validate_rebot_sac_dry_run.py \
  --dataset-root /mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sim_lerobot_like_dataset \
  --repo-id phi/rebot-sim-smoke \
  --device cuda:0 \
  --storage-device cpu \
  --batch-size 16 \
  --steps 3
```

Observed result:

```text
dataset_len=155
buffer_len=155
device=cuda:0
step=0 loss_critic=30.163258 loss_actor=-5.024768 loss_temperature=7.806723 selected_action_shape=(16, 7)
step=1 loss_critic=38.149658 loss_actor=-5.368615 loss_temperature=8.358242 selected_action_shape=(16, 7)
step=2 loss_critic=26.290892 loss_actor=-5.518153 loss_temperature=8.320305 selected_action_shape=(16, 7)
```

This confirms the minimal training compute path:

```text
LeRobotDataset -> ReplayBuffer -> SACPolicy critic/actor/temperature losses -> optimizer steps -> action selection
```

It is still a dry-run. The dataset is tiny and contains synthetic kinematic episodes, so the loss values only prove the pipeline executes; they do not measure useful policy quality.

## Batch Simulation Dataset and 50-Step SAC Dry-Run

Batch episode generator:

```text
scripts/hackathon/generate_rebot_sim_episodes.py
```

Local generation command:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/generate_rebot_sim_episodes.py \
  --sim-script /Users/fbsh/projects/drtc-Phi/scripts/hackathon/sim_rebot_policy_loop.py \
  --output-dir /Users/fbsh/projects/drtc-Phi/results/rebot_sim_batch_train \
  --export-jsonl /Users/fbsh/projects/drtc-Phi/results/rebot_sim_episodes_batch_train.jsonl \
  --ik-episodes 32 \
  --random-episodes 16 \
  --seed 20260424
```

Observed batch summary:

```text
episodes=48
frames=2044
success_count=29
failure_count=19
done_count=48
mock_proportional_ik: 29 success, 3 failure
mock_random: 16 failure
```

Converted dataset:

```text
local:  /Users/fbsh/projects/drtc-Phi/results/rebot_sim_lerobot_batch_train
MI300X: /mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sim_lerobot_batch_train
repo_id: phi/rebot-sim-batch-train
```

ReplayBuffer validation on MI300X:

```text
dataset_len=2044
buffer_len=2044
state_shape=(32, 13)
next_state_shape=(32, 13)
action_shape=(32, 7)
reward_shape=(32,)
buffer_done_count=48
```

50-step SAC GPU dry-run on MI300X:

```bash
cd /mnt/models_alehe/phi-fbsh/Evo-RL-Phi
.venvs/pi05-openpi-ssp/bin/python \
  /mnt/models_alehe/phi-fbsh/drtc-Phi/scripts/hackathon/validate_rebot_sac_dry_run.py \
  --dataset-root /mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sim_lerobot_batch_train \
  --repo-id phi/rebot-sim-batch-train \
  --device cuda:0 \
  --storage-device cpu \
  --batch-size 64 \
  --steps 50 \
  --output-dir /mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sac_batch_train_dry_run
```

Output artifacts:

```text
/mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sac_batch_train_dry_run/metrics.jsonl
/mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sac_batch_train_dry_run/sac_policy_state.pt
/mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sac_batch_train_dry_run/optimizers.pt
/mnt/models_alehe/phi-fbsh/drtc-Phi/results/rebot_sac_batch_train_dry_run/summary.json
```

Observed loss endpoints:

```text
step=0  loss_critic=36.745316 loss_actor=-4.604044 loss_temperature=8.082093 selected_action_shape=(64, 7)
step=49 loss_critic=4.130821  loss_actor=-8.616002 loss_temperature=8.099674 selected_action_shape=(64, 7)
```

Interpretation:

- The larger simulated dataset can drive a short SAC training loop on MI300X ROCm.
- Critic loss decreases over this tiny run, which is a useful sanity signal that gradients and optimizer updates are active.
- This still is not a policy-quality benchmark. It is a reproducible offline training pipeline check using synthetic kinematic data.

## SAC Checkpoint Rollout Evaluation

Rollout evaluation script:

```text
scripts/hackathon/evaluate_rebot_sac_rollout.py
```

The script loads:

```text
/Users/fbsh/projects/drtc-Phi/results/rebot_sac_batch_train_dry_run_sac_policy_state.pt
```

and compares three policies on the same target distribution:

```text
sac_checkpoint
ik_baseline
random_baseline
```

Full 48-target evaluation command:

```bash
cd /Users/fbsh/projects/reBotArm_control_py
PYTHONPATH=/Users/fbsh/projects/reBotArm_control_py:/Users/fbsh/projects/drtc-Phi/src \
uv run python /Users/fbsh/projects/drtc-Phi/scripts/hackathon/evaluate_rebot_sac_rollout.py \
  --dataset-root /Users/fbsh/projects/drtc-Phi/results/rebot_sim_lerobot_batch_train \
  --checkpoint /Users/fbsh/projects/drtc-Phi/results/rebot_sac_batch_train_dry_run_sac_policy_state.pt \
  --output-csv /Users/fbsh/projects/drtc-Phi/results/rebot_sac_rollout_eval_48.csv \
  --output-json /Users/fbsh/projects/drtc-Phi/results/rebot_sac_rollout_eval_48_summary.json \
  --episodes 48 \
  --seed 20260424 \
  --device cpu
```

Observed result:

```text
sac_checkpoint: 2/48 success, success_rate=0.0417, mean_final_distance_m=0.146299, mean_min_distance_m=0.060704
ik_baseline:    44/48 success, success_rate=0.9167, mean_final_distance_m=0.014765, mean_min_distance_m=0.014765
random_baseline:2/48 success, success_rate=0.0417, mean_final_distance_m=0.145369, mean_min_distance_m=0.058475
```

Interpretation:

- The checkpoint can be loaded and rolled out through the simulator.
- The 50-step SAC dry-run checkpoint does not produce a useful closed-loop reaching controller.
- Its success rate is indistinguishable from random on this evaluation, while the IK baseline solves most targets.
- This is expected for a tiny offline SAC dry-run over synthetic demonstrations and random failures; critic loss decrease alone is not a control-quality metric.

Immediate implication:

The current RL stack is now validated as an executable pipeline, but not as a successful policy-learning setup. To improve policy quality, the next experiment needs either much longer/offline training with proper evaluation checkpoints, behavior cloning or SAC-from-demonstrations pretraining, or a simulator reward/online interaction loop rather than only a short static-buffer dry-run.

## What This Validates

Validated:

- The simulator can act as a deterministic state transition backend.
- A 7D action adapter can drive the 6-DOF arm plus gripper scalar.
- The loop can produce reward/success signals.
- The loop can record episode-like rows for future dataset conversion.
- Evo-RL-Phi can read the exported LeRobot v3.0 dataset.
- Evo-RL-Phi can convert it into a ReplayBuffer and sample batches on MI300X.
- Evo-RL-Phi SAC can run critic, actor, and temperature optimizer steps on MI300X ROCm.
- A 48-episode / 2044-frame synthetic dataset can produce metrics and checkpoints from a 50-step SAC dry-run.
- The produced checkpoint can be loaded and evaluated in the reBot simulator.

Not validated:

- Real robot dynamics.
- Collisions.
- Gripper contact.
- Food manipulation.
- Camera observations.
- DRTC action semantics.
- Successful learned control policy from SAC.

## Next Integration Steps

Recommended order:

1. Add behavior cloning or demonstration-supervised pretraining as a stronger first learned baseline.
2. Run longer SAC experiments with periodic rollout evaluation, not just training loss.
3. Increase simulated data diversity and filter unreachable targets more systematically.
4. Add camera/image observations once the reBot hardware or a vision simulator is available.
5. Replace the DRTC action adapter with a task-appropriate policy/action head before real robot use.

Important: the current `vlash-pi05-libero-async5` output is `50 x 7`, but those 7 dimensions are not guaranteed to mean B601 joint deltas. For a plumbing test we can interpret them through a safe clipped adapter; for real hardware we must not use that adapter without validation.
