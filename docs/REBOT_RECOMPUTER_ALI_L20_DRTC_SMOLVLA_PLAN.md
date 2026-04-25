# reBot + reComputer + Ali L20 DRTC/SmolVLA Plan

Date: 2026-04-25

## Goal

Build a practical remote-inference control stack for the hackathon:

```text
reBot arm + cameras
  -> reComputer Jetson robot client
  -> DRTC/gRPC transport
  -> Aliyun L20 SmolVLA policy server
  -> 50 x 6 action chunks
  -> reComputer scheduler / adapter / safety
  -> reBot actuator commands
```

The immediate target is not to prove that the current public SmolVLA checkpoint can solve the reBot task.
The immediate target is to validate the full system path and expose the remaining embodiment/action-adapter work.

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

Required before real movement:

- Identify reBot command interface.
- Identify command units: joint position, joint delta, servo ticks, radians, or end-effector command.
- Define safe joint limits.
- Define speed/acceleration limits.
- Define emergency stop.
- Map SmolVLA `action[6]` to reBot command space or disable direct execution and run logging-only mode.

### reBot-Specific Policy

The current checkpoint is not trained for reBot.
It can validate the system path, but not task competence.

Paths to task competence:

1. Use rule-based / IK controller for the demo-critical path.
2. Collect reBot data in LeRobot format.
3. Fine-tune SmolVLA or train ACT/Diffusion on reBot data.
4. Use DRTC to serve the trained policy remotely.

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
read camera1
read camera2
read joint/state if available
construct observation.state[6]
construct observation.images.camera1/camera2
send to DRTC server
receive 50 x 6 action chunk
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
