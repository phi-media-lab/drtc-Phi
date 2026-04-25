# reBot + reComputer + Ali L20 DRTC/SmolVLA Plan

Date: 2026-04-25

## Goal

Build a practical remote-inference control stack for the hackathon:

```text
reBot arm + cameras
  -> reComputer Jetson robot client
  -> DRTC/gRPC transport
  -> Aliyun L20 SmolVLA policy server
  -> action chunks
  -> reComputer scheduler / adapter / safety
  -> reBot actuator commands
```

The immediate target is not to prove that the current public SmolVLA checkpoint can solve the reBot task.
The immediate target is to validate the full system path and expose the remaining embodiment/action-adapter work.

After collecting the first reBot dataset, the model target is now clearer:

```text
current system baseline:
  SO101 SmolVLA checkpoint -> 50 x 6 action chunks

next reBot policy target:
  seeed_b601_dm_follower dataset -> 50 x 7 reBot action chunks
```

## Current Machines

### reComputer Robot Client

SSH:

```bash
ssh recomputer@10.42.0.254
```

Password used during setup: `1`

Observed configuration:

- Hostname: `recomputer-desktop`
- OS: Ubuntu 22.04.5 LTS
- L4T / JetPack: R36.4.3
- Image: Seeed `mfi_recomputer-robo-orin-nx-16g-j401-gmsl-6.2-36.4.3-2025-08-21`
- Kernel: `5.15.148-tegra`
- CPU: 8 x Cortex-A78AE, `aarch64`
- RAM: 15 GiB
- Disk: 116G total, 88G free
- GPU: NVIDIA Orin `nvgpu`
- CUDA: 12.6
- Python: 3.10.12
- Docker: not installed
- Existing Python packages: `cv2==4.8.0`, `numpy==1.21.5`
- Missing Python packages: `torch`, `torchvision`, `lerobot`, `pyserial`
- Network: `enP8p1s0` at `10.42.0.254/24`
- Mac -> reComputer ping: about 0.9 ms
- Current device gap: no `/dev/video*`, `/dev/ttyACM*`, or `/dev/ttyUSB*` observed yet

Role:

- Primary robot client.
- Camera capture and robot IO.
- Local schedule execution and safety checks.
- Not the primary VLA inference host.

### Aliyun L20 Policy Server

SSH:

```bash
ssh -i /Users/fbsh/ali-gpu-key.pem root@47.106.21.198
```

Observed configuration:

- Hostname: `iZwz90sy164cwmvwfk3eugZ`
- OS: Ubuntu 24.04.4 LTS
- CPU: Intel Xeon Gold 6462C, 16 vCPU / 8 cores / 2 threads per core
- RAM: 123 GiB
- Disk: 126G total, 83G free after setup
- GPU: NVIDIA L20, 46 GiB VRAM
- Driver: 580.126.09
- CUDA driver capability: 13.0
- Docker: installed with NVIDIA runtime
- Python env: `/root/work/drtc-Phi/.venv`
- Repo: `/root/work/drtc-Phi`
- Installed deps: `uv pip install -e ".[async,smolvla]"`
- PyTorch: `2.7.1+cu126`
- CUDA available: true
- Device: `NVIDIA L20`

Role:

- Primary remote policy server.
- Runs `drtc-Phi` DRTC server.
- Loads SmolVLA checkpoint.
- Emits action chunks.

## Current Verified Results

### reBot Native Dataset

Dataset:

```text
https://huggingface.co/datasets/Lisette1231/20260425_flipbreadtopot1
```

Repository metadata:

```text
dataset id: Lisette1231/20260425_flipbreadtopot1
sha: 358427cd9835a2aec63c576f4cff8f8f3cb23797
last modified: 2026-04-25 07:20:09 UTC
visibility: public, not gated
storage: about 223 MB
license tag: apache-2.0
format tags: LeRobot, parquet, video, timeseries
```

Files observed:

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
videos/observation.images.front/chunk-000/file-000.mp4
videos/observation.images.wrist/chunk-000/file-000.mp4
```

LeRobot metadata:

```text
codebase_version: v3.0
robot_type: seeed_b601_dm_follower
episodes: 10
frames: 3711
fps: 30
tasks: 1
split: train 0:10
task: flip the bread in the pot
```

Feature schema:

```text
observation.state:
  dtype: float32
  shape: [7]
  names:
    shoulder_pan.pos
    shoulder_lift.pos
    elbow_flex.pos
    wrist_flex.pos
    wrist_yaw.pos
    wrist_roll.pos
    gripper.pos

action:
  dtype: float32
  shape: [7]
  names:
    shoulder_pan.pos
    shoulder_lift.pos
    elbow_flex.pos
    wrist_flex.pos
    wrist_yaw.pos
    wrist_roll.pos
    gripper.pos

observation.images.front:
  dtype: video
  shape: [480, 640, 3]
  codec: av1
  fps: 30

observation.images.wrist:
  dtype: video
  shape: [480, 640, 3]
  codec: av1
  fps: 30
```

Action ranges from `meta/stats.json`:

```text
min:
  [-75.900002, -128.800003, -132.000000, -6.900000, -68.400002, -32.599998, -211.199997]

max:
  [7.500000, 1.000000, 1.000000, 90.000000, 18.000000, 8.100000, 0.000000]

mean:
  [-23.073085, -62.618144, -60.841318, 42.934629, -18.021124, -7.354488, -69.075021]

std:
  [21.721202, 52.807847, 47.422979, 32.395199, 22.913585, 7.788056, 64.077581]
```

Implications:

- This is the first usable reBot-native policy dataset in the current workflow.
- It matches the physical arm family: `seeed_b601_dm_follower`.
- It confirms the real robot action/state interface is 7D, not the current SO101 checkpoint's 6D interface.
- It confirms the real image keys should be `observation.images.front` and `observation.images.wrist`, not `camera1` and `camera2`.
- It is too small to claim robust task generalization, but sufficient for an overfit/smoke-test policy and full DRTC integration test.
- `complementary_info.policy_action` is all zeros in the observed stats, so the supervised learning target should remain `action`.

This changes the model plan:

```text
SO101 checkpoint:
  keep for transport/server validation only

Lisette1231/20260425_flipbreadtopot1:
  use as the reBot-native training/evaluation baseline
```

### reBot SmolVLA Training Result

Training host:

```text
Aliyun L20
repo: /root/work/drtc-Phi
env: /root/work/drtc-Phi/.venv
```

Important environment note:

- Default LeRobot video backend selected `torchcodec`.
- `torchcodec` failed because system FFmpeg shared libraries were unavailable.
- The working path is to force the dataset backend to `pyav`:

```bash
--dataset.video_backend=pyav
```

Do not fine-tune directly from `lerobot/smolvla_base` with `--policy.path`.
That checkpoint config contains SO101-style features:

```text
observation.images.camera1/camera2/camera3
observation.state: 6D
action: 6D
```

For this reBot dataset, use a dataset-driven SmolVLA config instead:

```bash
--policy.type=smolvla
--policy.load_vlm_weights=true
```

Smoke test command:

```bash
cd /root/work/drtc-Phi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=Lisette1231/20260425_flipbreadtopot1 \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --steps=20 \
  --eval_freq=0 \
  --save_freq=20 \
  --log_freq=1 \
  --num_workers=2 \
  --wandb.enable=false \
  --output_dir=outputs/train/rebot_smolvla_flipbread_smoke_20260425_20steps
```

Smoke result:

```text
status: completed
checkpoint: outputs/train/rebot_smolvla_flipbread_smoke_20260425_20steps/checkpoints/000020/pretrained_model
dataset frames: 3711
dataset episodes: 10
effective batch size: 2
learnable params: 100M
total params: 450M
final logged loss: 1.226
checkpoint size: about 1.3G
```

The smoke checkpoint reloads and predicts:

```text
input:
  observation.images.wrist: (1, 3, 480, 640)
  observation.images.front: (1, 3, 480, 640)
  observation.state: (1, 7)
  task: flip the bread in the pot

output:
  action chunk: (1, 50, 7)

sample first_action:
  [6.600660, -17.100586, -13.392845, 39.998962, -13.478397, -7.479803, -111.679703]
```

Overfit run command:

```bash
cd /root/work/drtc-Phi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=Lisette1231/20260425_flipbreadtopot1 \
  --dataset.video_backend=pyav \
  --batch_size=8 \
  --steps=1000 \
  --eval_freq=0 \
  --save_freq=500 \
  --log_freq=20 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps
```

Overfit result:

```text
status: completed
checkpoint 500:  outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps/checkpoints/000500/pretrained_model
checkpoint 1000: outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps/checkpoints/001000/pretrained_model
effective batch size: 8
learnable params: 100M
total params: 450M
output dir size: about 2.5G
loss:
  step 20:   1.342
  step 100:  0.461
  step 300:  0.190
  step 500:  0.127
  step 780:  0.087
  step 1000: 0.090
```

Final checkpoint reload/predict result:

```text
checkpoint:
  outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps/checkpoints/001000/pretrained_model

input features:
  observation.state: (7,)
  observation.images.wrist: (3, 480, 640)
  observation.images.front: (3, 480, 640)

output features:
  action: (7,)

single action chunk:
  shape: (1, 50, 7)
  dtype: float32
  first-call latency on L20: 540.30 ms
  min: -105.027985
  max: 39.408401
  mean: -17.334152
  std: 30.411362

sample first_action:
  [-4.231995, -23.582581, -25.166920, 19.646269, -8.994103, -8.583467, -33.616550]
```

This checkpoint is a reBot-native DRTC candidate.
It should still be treated as an overfit validation model, not a safe autonomous control policy.

### Expanded Flip-Bread Dataset

Additional datasets collected for the same action:

```text
https://huggingface.co/datasets/Lisette1231/20260425_flipbreadtopot2
https://huggingface.co/datasets/Lisette1231/20260425_flipbreadtopot3
https://huggingface.co/datasets/Lisette1231/20260425_flipbreadtopot4_newway
https://huggingface.co/datasets/Lisette1231/20260425_flipbreadtopot5_newway
```

All five datasets share the same LeRobot schema:

```text
robot_type: seeed_b601_dm_follower
fps: 30
task: flip the bread in the pot
observation.state: 7D
action: 7D
observation.images.front: video [480, 640, 3]
observation.images.wrist: video [480, 640, 3]
```

Dataset sizes observed:

```text
Lisette1231/20260425_flipbreadtopot1:
  episodes: 10
  frames: 3711
  storage: 233795126 bytes

Lisette1231/20260425_flipbreadtopot2:
  episodes: 10
  frames: 3606
  storage: 163336493 bytes

Lisette1231/20260425_flipbreadtopot3:
  episodes: 4
  frames: 1410
  storage: 68498309 bytes

Lisette1231/20260425_flipbreadtopot4_newway:
  episodes: 10
  frames: 3946
  storage: 199017392 bytes

Lisette1231/20260425_flipbreadtopot5_newway:
  episodes: 10
  frames: 5753
  storage: 277949778 bytes
```

Total:

```text
episodes: 44
frames: 18426
```

Note:

- The user described these as another 40 demonstrations.
- The Hugging Face metadata currently shows 34 additional episodes beyond the first dataset because `flipbreadtopot3` has 4 episodes.
- Current combined training baseline therefore uses 44 episodes total, not 50.

LeRobot multi-dataset lists are not supported by the current `TrainPipelineConfig`.
The working path is to merge the datasets locally first:

```bash
cat > /tmp/rebot_merge_44eps.json <<'JSON'
{
  "repo_id": "phi-media-lab/rebot_flipbreadtopot_20260425_44eps",
  "root": null,
  "new_repo_id": null,
  "push_to_hub": false,
  "operation": {
    "type": "merge",
    "repo_ids": [
      "Lisette1231/20260425_flipbreadtopot1",
      "Lisette1231/20260425_flipbreadtopot2",
      "Lisette1231/20260425_flipbreadtopot3",
      "Lisette1231/20260425_flipbreadtopot4_newway",
      "Lisette1231/20260425_flipbreadtopot5_newway"
    ]
  }
}
JSON

cd /root/work/drtc-Phi
source .venv/bin/activate
lerobot-edit-dataset --config_path /tmp/rebot_merge_44eps.json
```

Merge result:

```text
local repo id: phi-media-lab/rebot_flipbreadtopot_20260425_44eps
local path: /root/.cache/huggingface/lerobot/phi-media-lab/rebot_flipbreadtopot_20260425_44eps
episodes: 44
frames: 18426
size: about 899M
```

### reBot SmolVLA 44-Episode Training Result

Training initialization:

```text
base checkpoint:
  outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps/checkpoints/001000/pretrained_model

dataset:
  phi-media-lab/rebot_flipbreadtopot_20260425_44eps
```

Training command:

```bash
cd /root/work/drtc-Phi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.path=outputs/train/rebot_smolvla_flipbread_overfit_20260425_1000steps/checkpoints/001000/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=phi-media-lab/rebot_flipbreadtopot_20260425_44eps \
  --dataset.video_backend=pyav \
  --batch_size=8 \
  --steps=3000 \
  --eval_freq=0 \
  --save_freq=1000 \
  --log_freq=50 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=outputs/train/rebot_smolvla_flipbread_44eps_20260425_3000steps
```

Training result:

```text
status: completed
dataset frames: 18426
dataset episodes: 44
effective batch size: 8
learnable params: 100M
total params: 450M
checkpoints:
  outputs/train/rebot_smolvla_flipbread_44eps_20260425_3000steps/checkpoints/001000/pretrained_model
  outputs/train/rebot_smolvla_flipbread_44eps_20260425_3000steps/checkpoints/002000/pretrained_model
  outputs/train/rebot_smolvla_flipbread_44eps_20260425_3000steps/checkpoints/003000/pretrained_model
output dir size: about 3.7G
```

Loss curve:

```text
step 50:   0.148
step 500:  0.111
step 1000: 0.087
step 1500: 0.070
step 2000: 0.057
step 2500: 0.049
step 3000: 0.049
```

Final checkpoint reload/predict result:

```text
checkpoint:
  outputs/train/rebot_smolvla_flipbread_44eps_20260425_3000steps/checkpoints/003000/pretrained_model

Hugging Face:
  https://huggingface.co/fbsh96/rebot_smolvla_flipbread_44eps_20260425_3000steps

input features:
  observation.state: (7,)
  observation.images.wrist: (3, 480, 640)
  observation.images.front: (3, 480, 640)

output features:
  action: (7,)

output:
  action chunk: (1, 50, 7)

latency on L20:
  first call: about 535 ms
  steady calls: about 151-153 ms/chunk
```

Sample predictions from three merged-dataset frames:

```text
idx 0:
  first_action:
    [0.315390, -3.481686, -7.239395, 9.588888, 0.757010, -9.134871, -14.674559]

idx 736:
  first_action:
    [-34.396896, -117.993027, -103.424217, 73.848999, 0.255486, -21.714380, -76.466095]

idx 1471:
  first_action:
    [0.437017, -1.945919, -1.674629, 12.638105, -0.255466, -4.052654, 7.723423]
```

This is the best current reBot-native SmolVLA checkpoint for DRTC serving.
It still requires logging-only validation and a safety/action adapter before any real actuator execution.

### reBot ACT 44-Episode MI300X Training Result

This run validates that the same 44-episode reBot LeRobot dataset can also train
an ACT policy on the MI300X ROCm stack.

Machine:

```text
host: phi-amd-work
gpu: AMD Instinct MI300X VF
runtime: PyTorch ROCm, exposed as cuda:0 by torch
repo: /mnt/models_alehe/phi-fbsh/drtc-Phi
```

Dataset:

```text
repo id: phi-media-lab/rebot_flipbreadtopot_20260425_44eps
episodes: 44
frames: 18426
state/action dims: 7D
images:
  observation.images.front: (3, 480, 640)
  observation.images.wrist: (3, 480, 640)
```

Training command:

```bash
cd /mnt/models_alehe/phi-fbsh/drtc-Phi
source /mnt/models_alehe/phi-fbsh/.venvs/drtc-mi300x/bin/activate
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=act \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.push_to_hub=false \
  --dataset.repo_id=phi-media-lab/rebot_flipbreadtopot_20260425_44eps \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --steps=10000 \
  --eval_freq=0 \
  --save_freq=2500 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir=outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps
```

Training result:

```text
status: completed
effective batch size: 16
learnable params: 52M
checkpoints:
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/002500/pretrained_model
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/005000/pretrained_model
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/007500/pretrained_model
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/010000/pretrained_model
```

Loss curve:

```text
step 100:   9.431
step 1000:  1.672
step 2500:  0.650
step 5000:  0.241
step 7500:  0.157
step 10000: 0.124
```

Final checkpoint reload/predict result:

```text
checkpoint:
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/010000/pretrained_model

input features:
  observation.state: (7,)
  observation.images.wrist: (3, 480, 640)
  observation.images.front: (3, 480, 640)

output features:
  action: (7,)

output:
  predict_action_chunk: (1, 50, 7)
  select_action: (1, 7)

latency on MI300X:
  first chunk call: about 6654 ms
  steady chunk calls: about 7.86 ms/chunk, about 127 Hz
  select_action queue consumption: about 0.28 ms/action
```

DRTC loopback validation:

```text
server: PolicyServerDrtc on 127.0.0.1
client: synthetic reBot observation sender on the same MI300X host
policy type: act
checkpoint:
  outputs/train/rebot_act_flipbread_44eps_mi300x_b16_10000steps/checkpoints/010000/pretrained_model

synthetic observation:
  observation.state: 7D zeros
  observation.images.front: 480x640x3 uint8 zeros
  observation.images.wrist: 480x640x3 uint8 zeros
  task: "flip the bread in the pot"

result:
  policy setup with 1 warmup pass: about 7.64 s
  warmup pass: about 6719 ms
  loopback roundtrip after warmup: about 72 ms
  server observation-to-action-send: about 48.9 ms
  returned action chunk: (50, 7)
```

Implementation note:

The first DRTC loopback exposed a warmup bug in `policy_server_drtc.py`: the
dummy warmup observation assumed a 6D state fallback when `lerobot_features`
used the normal dataset-feature dict form. For reBot/B601 this mismatched the
7D normalizer. The warmup state dimension now uses `observation.state.shape[0]`
or the length of `observation.state.names` before falling back to 6.

Interpretation:

- The ACT checkpoint is reBot-native and matches the B601 7D action/state schema.
- The steady chunk inference path is much faster than the current SmolVLA checkpoint on L20, but this is a different model class and should not be interpreted as a VLA quality comparison.
- Like the SmolVLA checkpoint, this remains an offline imitation checkpoint. It still needs logging-only replay, action scaling/safety limits, and hardware dry-run validation before real actuator execution.

### SmolVLA Checkpoint

Checkpoint:

```text
jackvial/so101_smolvla_pickplaceorangecube_e100
```

Hugging Face status:

- Public.
- Complete files found: `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`, normalization safetensors, `train_config.json`.
- This is an SO101 checkpoint, not a reBot checkpoint.

Feature schema:

```text
input:
  observation.state: shape (6,)
  observation.images.camera1: shape (3, 600, 800)
  observation.images.camera2: shape (3, 600, 800)

output:
  action: shape (6,)

chunk:
  chunk_size: 50
  n_action_steps: 50
```

Single-policy inference on Ali L20:

```text
synthetic batch:
  observation.state: (1, 1, 6)
  observation.images.camera1: (1, 1, 3, 600, 800)
  observation.images.camera2: (1, 1, 3, 600, 800)

output:
  action chunk: (1, 50, 6)
  dtype: float32
  device: cuda:0

timing:
  warmup: 517.14 ms
  steady median: 130.85 ms/chunk
  equivalent chunk rate: about 7.64 chunks/s

sample first action:
  [-0.150116, 0.181613, -0.356215, -0.366001, 0.089527, 0.15717]
```

Action statistics from synthetic zero-image input:

```text
min:  -1.062439
max:   0.423081
mean: -0.174317
std:   0.402507
```

Interpretation:

- The model loads and runs correctly on L20.
- The model returns the expected `50 x 6` action chunk.
- The output is not directly safe for reBot until the action semantics are mapped and bounded.

### DRTC Full-Path Synthetic Test

Server:

```bash
cd /root/work/drtc-Phi
source .venv/bin/activate
python examples/tutorial/async-inf/policy_server_drtc.py --host 0.0.0.0 --port 18201 --fps 15
```

Client used via local SSH tunnel during validation:

```bash
ssh -f -i /Users/fbsh/ali-gpu-key.pem \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18202:127.0.0.1:18201 \
  root@47.106.21.198 -N
```

Synthetic client command:

```bash
cd /Users/fbsh/projects/drtc-Phi
.venv/bin/python scripts/run_high_latency_split_client.py \
  --server-address 127.0.0.1:18202 \
  --pretrained-name-or-path jackvial/so101_smolvla_pickplaceorangecube_e100 \
  --policy-type smolvla \
  --device cuda \
  --duration-s 8 \
  --fps 15 \
  --actions-per-chunk 50 \
  --s-min 25 \
  --epsilon 2 \
  --initial-latency-ms 50 \
  --prefill-chunks 1 \
  --prefill-timeout-s 40 \
  --state-dim 6 \
  --image-size 800x600 \
  --image-feature-names camera1,camera2 \
  --image-mode zeros \
  --jpeg-quality 40 \
  --task 'Pick up the orange cube and place it in the target area.' \
  --output results/drtc_smolvla_ali_l20_tunnel_camera12_venue_20260425.csv
```

Result:

```text
ready_s: 0.054
policy_setup_s: 11.965
ticks: 121
stalls: 0
starvation_ratio: 0.0000
observations_sent: 5
chunks_received: 5
schedule_min: 23
schedule_max: 49
latency_estimate_ms:
  min: 175.0
  median: 195.0
  max: 199.5
last_num_actions: 50
last_action_dim: 6
payload_bytes: 16936
raw_image_bytes: 2880000
encoded_image_bytes: 16452
```

Important fix from validation:

- The initial synthetic client used `observation.images.image` and `observation.images.wrist_image`.
- The SmolVLA checkpoint requires `observation.images.camera1` and `observation.images.camera2`.
- `scripts/run_high_latency_split_client.py` now supports:

```bash
--image-feature-names camera1,camera2
```

### reComputer Direct Public DRTC Test

After opening Aliyun inbound TCP `18201`, the reComputer can reach the L20 policy server directly,
without SSH tunneling:

```text
reComputer -> 47.106.21.198:18201 -> Ali L20 DRTC/SmolVLA -> reComputer
```

Lightweight client location on reComputer:

```text
~/drtc-lite/scripts/recomputer_drtc_observation_client.py
~/venvs/rebot-drtc-client
```

Direct validation command:

```bash
cd ~/drtc-lite
PYTHONPATH=$PWD ~/venvs/rebot-drtc-client/bin/python \
  scripts/recomputer_drtc_observation_client.py \
  --server-address 47.106.21.198:18201 \
  --setup-policy \
  --timeout-s 60
```

Observed result with raw synthetic dual `800x600` images:

```text
ready_ms=67.7
policy_setup_ms=12786.4
observation_sent_ms=1409.6
payload_bytes=2880601
chunk_received num_actions=50 action_dim=6
actions shape=(50, 6)
min=-46.737064 max=62.399841 mean=6.024907 std=32.959862
first_action=[-11.614491, 4.149872, 32.659039, 27.594967, -4.467266, -0.082888]
```

Interpretation:

- Direct public DRTC access from reComputer to Ali L20 is working.
- The `2.88MB` raw image payload is too slow for the control loop.
- Use JPEG compression for camera payloads before moving to scheduler tests.

Compressed validation command:

```bash
cd ~/drtc-lite
PYTHONPATH=$PWD ~/venvs/rebot-drtc-client/bin/python \
  scripts/recomputer_drtc_observation_client.py \
  --server-address 47.106.21.198:18201 \
  --setup-policy \
  --jpeg-quality 40 \
  --timeout-s 60
```

Observed result with JPEG q40 synthetic dual `800x600` zero images:

```text
ready_ms=81.9
policy_setup_ms=12726.0
observation_sent_ms=23.4
payload_bytes=16936
chunk_received num_actions=50 action_dim=6
actions shape=(50, 6)
min=-45.761475 max=61.118378 mean=6.487365 std=32.542755
first_action=[-10.112844, 3.084328, 31.675591, 29.949953, -3.637812, 0.196829]
```

JPEG q40 impact:

```text
raw payload:  2,880,601 bytes, observation_sent_ms=1409.6
JPEG payload:    16,936 bytes, observation_sent_ms=23.4
```

This is the transport mode to use for the next simulated scheduler and camera-only client tests.

## Why This Architecture

### Why reComputer Should Be Client-Only

The reComputer has an Orin GPU, but only 16 GiB RAM and currently no PyTorch/LeRobot environment.
It is suitable for IO, camera capture, safety, and local scheduling.
It is not the best target for large VLA inference during the hackathon.

### Why Ali L20 Should Run SmolVLA

Ali L20 has:

- 46 GiB GPU memory.
- Working CUDA PyTorch environment.
- Much lower venue latency than the MI300X instance.
- Successful SmolVLA and DRTC validation.

### Why SmolVLA Is the Current Best VLA Choice

SmolVLA is LeRobot-native and integrates with:

- LeRobot policy factory.
- LeRobot pre/post processors.
- DRTC async inference server.
- PyTorch CUDA on L20.

This is much lower-risk than continuing with JAX/ROCm or pi0.5 for the immediate demo.

## Current Gaps

### Hardware Enumeration

The reComputer currently does not show:

```text
/dev/video*
/dev/ttyACM*
/dev/ttyUSB*
```

This blocks real robot client validation.

Next checks after plugging devices:

```bash
ls -l /dev/video* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
lsusb
dmesg -T | tail -120
v4l2-ctl --list-devices
```

### reBot Action Semantics

The public checkpoint emits SO101-style 6D actions.
Do not send these directly to reBot actuators without an adapter.

The collected reBot dataset defines the native B601 command/state schema as 7D:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_yaw.pos
wrist_roll.pos
gripper.pos
```

Required before real movement:

- Identify reBot command interface.
- Identify command units: joint position, joint delta, servo ticks, radians, or end-effector command.
- Define safe joint limits.
- Define speed/acceleration limits.
- Define emergency stop.
- For SO101 checkpoint validation, keep actions logging-only.
- For reBot policy validation, treat model output as 7D B601 commands and still pass through clipping, rate limiting, and joint-limit checks.

### reBot-Specific Policy

The current checkpoint is not trained for reBot.
It can validate the system path, but not task competence.

Paths to task competence:

1. Use rule-based / IK controller for the demo-critical path.
2. Use `Lisette1231/20260425_flipbreadtopot1` as the first reBot-native training dataset.
3. Fine-tune SmolVLA or train ACT/Diffusion on the 7D B601 data.
4. Validate overfit inference on Ali L20.
5. Use DRTC to serve the trained reBot policy remotely.

### Network Exposure

Direct access to `47.106.21.198:18201` was blocked during earlier tests, likely by Aliyun security group or cloud firewall.
Local validation used SSH tunneling.

Options:

1. Open inbound TCP `18201` in Aliyun security group and host firewall.
2. Use SSH tunnel from reComputer to Ali L20.
3. Use VPN/Tailscale/WireGuard if allowed.

For the venue, SSH tunnel is lowest-risk.

## Execution Plan

### Phase 1: Stabilize reComputer Access

1. Confirm direct SSH:

```bash
ssh recomputer@10.42.0.254
```

2. Confirm network to Ali L20:

```bash
ping -c 10 47.106.21.198
nc -vz -w 3 47.106.21.198 22
```

3. Establish tunnel from reComputer to Ali L20:

```bash
ssh -f -N \
  -i /path/to/ali-gpu-key.pem \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18201:127.0.0.1:18201 \
  root@47.106.21.198
```

If key transfer to reComputer is not acceptable, use Mac as an intermediate tunnel only for validation.

### Phase 2: Install Minimal reComputer Client Dependencies

Avoid full model dependencies first.
Install only robot-client essentials:

```bash
python3 -m venv ~/venvs/rebot-drtc-client
source ~/venvs/rebot-drtc-client/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install grpcio protobuf numpy opencv-python pyserial pillow
```

Then install a lightweight editable `drtc-Phi` client if needed.
If full LeRobot import is required, install with minimum extras and verify import cost.

### Phase 3: Detect Robot and Cameras

After plugging in hardware:

```bash
lsusb
ls -l /dev/video* /dev/ttyACM* /dev/ttyUSB*
v4l2-ctl --list-devices
dmesg -T | tail -120
```

Expected outcome:

- One or more serial devices for reBot control.
- One or two camera devices.

If serial devices do not appear:

- Check cable power/data capability.
- Check udev permissions.
- Check whether reBot uses CAN, USB CDC, serial, or vendor-specific bridge.

### Phase 4: Observation-Only Client

Build a reComputer client that does not move the robot:

```text
read front camera
read wrist camera
read 7D joint/state if available
construct observation.state[7]
construct observation.images.front/wrist
send to DRTC server
receive action chunk
log action chunk only
```

Success criteria:

- Receives action chunks from Ali L20.
- No starvation at 15 Hz scheduling.
- Logs payload size, send time, chunk latency, action range.

### Phase 5: Safety Adapter

Before actuator execution:

- Clip action range.
- Rate-limit action deltas.
- Enforce joint limits.
- Add software emergency stop.
- Add manual deadman switch if possible.
- Support dry-run mode and one-step mode.

Execution should start with:

```text
model output -> adapter -> print only
model output -> adapter -> simulated command
model output -> adapter -> one joint low-speed test
model output -> adapter -> full arm low-speed test
```

### Phase 6: Demo Strategy

Use two tracks:

1. Robust demo track:
   - IK / scripted / teleop-assisted control on reBot.
   - LeRobot data recording.
   - Safe and deterministic.

2. VLA system demo track:
   - reComputer sends observations.
   - Ali L20 runs SmolVLA through DRTC.
   - Client receives action chunks.
   - Initially logging-only or low-authority adapter.

This avoids risking the whole demo on an SO101 checkpoint controlling a different arm.

## Commands Reference

### Ali L20 Server

```bash
ssh -i /Users/fbsh/ali-gpu-key.pem root@47.106.21.198
cd /root/work/drtc-Phi
source .venv/bin/activate
python examples/tutorial/async-inf/policy_server_drtc.py --host 0.0.0.0 --port 18201 --fps 15
```

### Local/Mac Tunnel

```bash
ssh -f -i /Users/fbsh/ali-gpu-key.pem \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18202:127.0.0.1:18201 \
  root@47.106.21.198 -N
```

### Validated Synthetic SmolVLA DRTC Test

```bash
cd /Users/fbsh/projects/drtc-Phi
.venv/bin/python scripts/run_high_latency_split_client.py \
  --server-address 127.0.0.1:18202 \
  --pretrained-name-or-path jackvial/so101_smolvla_pickplaceorangecube_e100 \
  --policy-type smolvla \
  --device cuda \
  --duration-s 8 \
  --fps 15 \
  --actions-per-chunk 50 \
  --s-min 25 \
  --epsilon 2 \
  --initial-latency-ms 50 \
  --prefill-chunks 1 \
  --prefill-timeout-s 40 \
  --state-dim 6 \
  --image-size 800x600 \
  --image-feature-names camera1,camera2 \
  --image-mode zeros \
  --jpeg-quality 40 \
  --task 'Pick up the orange cube and place it in the target area.' \
  --output results/drtc_smolvla_ali_l20_tunnel_camera12_venue_20260425.csv
```

## Decision

Proceed with:

```text
reComputer robot client + Ali L20 DRTC SmolVLA server
```

Use SmolVLA for remote VLA/chunking validation.
Use IK/scripted/LeRobot-native reBot policy for safe task execution until reBot-specific training data exists.
