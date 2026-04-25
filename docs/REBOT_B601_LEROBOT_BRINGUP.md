# reBot B601-DM LeRobot Bring-up Notes

Last updated: 2026-04-24

This document summarizes the parts of the Seeed reBot B601-DM documentation that are relevant to the
Robot Cooking Hackathon and to the current MI300X + DRTC remote-inference plan.

Primary sources:

- Seeed Wiki: [Getting Started with reBot Arm B601-DM in LeRobot](https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/)
- Seeed Wiki: [reBot Arm B601-DM Visual Grasping Demo](https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/)
- GitHub: [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm)
- GitHub: [Seeed-Projects/lerobot-robot-seeed-b601](https://github.com/Seeed-Projects/lerobot-robot-seeed-b601)
- GitHub: [Seeed-Projects/lerobot-teleoperator-rebot-arm-102](https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-arm-102)
- SDK docs: [MotorBridge Python SDK](https://motorbridge.seeedstudio.com/)

## What Matters for the Hackathon

The official path is not a generic LeRobot install. Seeed recommends a Seeed-maintained LeRobot fork
plus two local integration packages:

- `Seeed-Projects/lerobot`
- `Seeed-Projects/lerobot-robot-seeed-b601`
- `Seeed-Projects/lerobot-teleoperator-rebot-arm-102`
- `motorbridge`

The B601-DM follower is integrated as a LeRobot robot type:

- `seeed_b601_dm_follower`

The reBot 102 leader is integrated as a LeRobot teleoperator type:

- `rebot_arm_102_leader`

The most important practical consequence is that the shortest reliable route is:

1. Bring up Seeed's local LeRobot environment on the on-site Ubuntu/Jetson machine.
2. Verify raw serial/CAN connectivity and calibration.
3. Verify leader-follower teleoperation.
4. Add cameras and confirm stable image FPS.
5. Record a small dataset and replay one episode.
6. Only then connect policy inference or DRTC.

## Hardware and Platform Assumptions

The official B601-DM stack targets:

- Ubuntu x86: Ubuntu 22.04, CUDA 12+, Python 3.10, Torch 2.6.
- Jetson Orin: JetPack 6.0 or 6.1, Python 3.10, Torch 2.3+.
- The Seeed page explicitly says JetPack 6.2 is not supported for this tutorial path.
- B601-DM uses 6 DOF plus gripper.
- B601-DM communication is CAN bus through either a SocketCAN-compatible adapter or Damiao USB2CAN serial bridge.
- The hackathon hardware mentions reComputer Robotics J4012, so expect Jetson constraints and avoid assuming a full desktop CUDA stack.

The reBot-DevArm repo lists B601-DM capabilities:

- Recommended payload: `1.5 kg`.
- Max reach: `650 mm`.
- Approximate arm weight: `4.5 kg`.
- Repeatability: `< 0.2 mm`.
- Supply voltage: `24V DC`.
- Supported ecosystems include ROS1/2, LeRobot, Pinocchio, Isaac Sim, Python SDK.

## Environment Setup

Official Seeed LeRobot setup:

```bash
cd ~
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
~/miniforge3/bin/conda init bash
source ~/.bashrc

mkdir ~/rebot_lerobot
cd ~/rebot_lerobot
git clone https://github.com/Seeed-Projects/lerobot.git
git clone https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-arm-102.git
git clone https://github.com/Seeed-Projects/lerobot-robot-seeed-b601.git

conda create -y -n lerobot python=3.12
conda activate lerobot
pip install -e ./lerobot
pip install -e ./lerobot-teleoperator-rebot-arm-102
pip install -e ./lerobot-robot-seeed-b601
pip install motorbridge
conda install ffmpeg -c conda-forge
```

Important compatibility notes:

- The wiki states Python 3.10 in the system requirements but uses Python 3.12 in the LeRobot conda command.
- On Jetson, installing LeRobot can replace GPU PyTorch with CPU wheels. Always check `torch.cuda.is_available()` afterward.
- On Jetson JetPack 6.0+, Seeed recommends `opencv-python==4.10.0.84` and `numpy==1.26.0`.
- If ffmpeg video errors occur, pin ffmpeg to `7.1.1`.

CUDA check:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## Serial and Device Permissions

Before calibration, teleoperation, record, or replay:

```bash
sudo chmod 666 /dev/ttyUSB*
sudo chmod 666 /dev/ttyACM*
```

Expected device roles:

- Leader arm: usually `/dev/ttyUSB*`.
- B601-DM follower through Damiao serial bridge: usually `/dev/ttyACM*`.
- SocketCAN path may use `can0` if the adapter is configured that way.

If `/dev/ttyACM0` cannot be found or the USB serial device disconnects, the wiki calls out `brltty` as a common conflict:

```bash
sudo dmesg | grep ttyUSB
sudo apt remove brltty
```

For Orbbec camera access:

```bash
sudo chmod a+rw /dev/bus/usb/*/*
```

If using `pyorbbecsdk`, install its udev rules:

```bash
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Calibration and Startup Behavior

B601-DM follower:

- The wiki says B601-DM automatically calibrates each time a LeRobot-related program starts.
- Before startup, place the arm in the documented initial pose with gripper fully closed.
- If recalibration is needed, delete calibration JSON files under:

```text
~/.cache/huggingface/lerobot/calibration/robots
~/.cache/huggingface/lerobot/calibration/teleoperators
```

reBot 102 leader:

- Startup calibration sets each servo's current position as zero.
- Joint ranges are taken from `config_rebot_arm_102_leader.py`, not from calibration data.
- Joint directions are config-defined. If signs are wrong, modify direction config or pass `joint_directions`; do not expect recalibration to fix it.

Leader calibration command:

```bash
sudo chmod 666 /dev/ttyUSB0
lerobot-calibrate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader
```

Leader raw-angle sanity check:

```bash
python ./lerobot-teleoperator-rebot-arm-102/examples/read_raw_angles.py \
  --port /dev/ttyUSB0
```

The expected zero pose output should show each joint near `0.00`.

## Teleoperation Baseline

Minimum leader-follower command from the Seeed wiki:

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --teleop.joint_directions='{"shoulder_pan":-1,"shoulder_lift":-1,"elbow_flex":1,"wrist_flex":1,"wrist_yaw":1,"wrist_roll":-1,"gripper":-4}'
```

The follower package README also shows a leader variant named `seeed_b601_dm_leader`, but the wiki path for the hackathon
should use `rebot_arm_102_leader` unless the actual on-site leader is another B601-based leader.

Useful safe direction-debug script from the 102 leader repo:

```bash
python examples/read_leader_follower_compare.py \
  --leader-port /dev/ttyUSB0 \
  --follower-port /dev/ttyACM0 \
  --follower-type dm \
  --follower-can-adapter damiao
```

Use this before sending commands if joint signs are suspect.

## Camera Bring-up

Find OpenCV cameras:

```bash
lerobot-find-cameras opencv
```

Single RGB camera teleop:

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true
```

Two RGB cameras:

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true
```

Practical camera notes:

- Use `MJPG` first. The wiki says `YUYV` can reduce resolution and FPS and can increase arm-operation lag.
- Avoid connecting two cameras through the same USB hub.
- Camera names such as `front` and `side` must remain consistent between data collection, training, and evaluation.
- If image read fails, connect the USB camera directly to the machine.

## Orbbec Gemini 2 Path

The LeRobot page supports Orbbec by adding an `orbbec` camera type manually:

1. Install `pyorbbecsdk`.
2. Install its requirements and force `numpy==1.26.0`.
3. Clone the Orbbec camera integration into the LeRobot camera package.
4. Modify `lerobot/cameras/utils.py` to instantiate `OrbbecCamera`.
5. Modify `lerobot/cameras/__init__.py` to import `OrbbecCameraConfig`.

Orbbec example command:

```bash
lerobot-teleoperate \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ up: {type: orbbec, width: 640, height: 880, fps: 30, focus_area:[60,300]}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true
```

For the hackathon, Orbbec is strategically valuable because the visual grasping demo uses depth and YOLO. It is also higher setup risk
than plain OpenCV RGB. Bring up RGB first, then Orbbec.

## Dataset Collection

Local dataset recording:

```bash
lerobot-record \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true \
  --dataset.repo_id=seeed_rebot_b601_dm/test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Grab the black cube" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30
```

Storage location for local data:

```text
~/.cache/huggingface/lerobot/<repo_id>
```

Recording keyboard controls:

- Right arrow: early stop current episode/reset and move to next.
- Left arrow: cancel and re-record current episode.
- ESC: stop, encode videos, and upload if enabled.

If keyboard controls do not work, the wiki suggests downgrading `pynput`:

```bash
pip install pynput==1.6.8
```

Dataset collection rules that matter for policy quality:

- Keep camera positions fixed.
- Keep lighting stable.
- Keep manipulated objects visible.
- Start with a reliable primitive such as grasp-and-place before adding variations.
- Record at least enough episodes for statistics to be computed; do not manually interrupt before completion.
- The operator should be able to perform the task from camera views alone.

## Visualize and Replay

Visualize local dataset:

```bash
lerobot-dataset-viz \
  --repo-id seeed_rebot_b601_dm/test \
  --episode-index 0 \
  --display-compressed-images=false
```

Replay one recorded episode:

```bash
lerobot-replay \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.can_adapter=damiao \
  --robot.id=follower1 \
  --dataset.repo_id=seeed_rebot_b601_dm/test \
  --dataset.episode=0
```

The wiki marks replay as unstable. Treat it as a useful primitive source and diagnostics tool, not the only demo path.

## Training and Evaluation Paths

Official examples:

ACT local training:

```bash
lerobot-train \
  --dataset.repo_id=seeed_rebot_b601_dm/test \
  --policy.type=act \
  --output_dir=outputs/train/act_rebot_test \
  --job_name=act_rebot_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=300000
```

ACT evaluation:

```bash
lerobot-record \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --robot.id=follower1 \
  --display_data=false \
  --dataset.repo_id=seeed/eval_test123 \
  --dataset.single_task="Put lego brick into the transparent box" \
  --policy.path=outputs/train/act_rebot_test/checkpoints/last/pretrained_model
```

Pi0 training example:

```bash
lerobot-train \
  --policy.type=pi0 \
  --dataset.repo_id=seeed/eval_test123 \
  --job_name=pi0_training \
  --output_dir=outputs/pi0_training \
  --policy.pretrained_path=lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --steps=20000 \
  --policy.device=cuda \
  --batch_size=32 \
  --wandb.enable=false
```

Pi0.5 training example:

```bash
lerobot-train \
  --dataset.repo_id=seeed/eval_test123 \
  --policy.type=pi05 \
  --output_dir=outputs/pi05_training \
  --job_name=pi05_training \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=false \
  --policy.dtype=bfloat16 \
  --steps=3000 \
  --policy.device=cuda \
  --batch_size=32
```

For the hackathon, ACT is a practical imitation-learning fallback if enough targeted demos can be collected.
Pi0/Pi0.5 is a stronger technical story but less likely to produce a reliable cooking policy in 48 hours without a prepared dataset.

## Visual Grasping Demo

The visual grasping demo is a separate Seeed route using:

- B601-DM
- Orbbec Gemini 2 depth camera
- YOLO + OBB or minimum-area rectangle
- Eye-in-hand hand-eye calibration
- `RebotArm` interface, IK, trajectory control, and gripper state machine

This demo is highly relevant to cooking because reliable grasping of ingredients and tools is a better near-term primitive than
end-to-end VLA control.

Setup summary:

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
conda create -n rebotarm python=3.10 -y
conda activate rebotarm
pip install -r requirements.txt

git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..

sudo apt-get update
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev
cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Dependency verification:

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
python -c "import motorbridge; print('motorbridge OK')"
```

Hand-eye calibration:

```bash
python scripts/collect_handeye_eih.py
```

Manual mode:

```bash
python scripts/collect_handeye_eih.py --manual
```

Calibration output:

```text
config/calibration/orbbec_gemini2/hand_eye.npz
```

Detection-only check:

```bash
python scripts/object_detection.py
```

Grasp-estimation-only check:

```bash
python scripts/ordinary_grasp_pipeline.py
```

Main grasping program:

```bash
python scripts/main.py --dry-run
python scripts/main.py
```

Runtime controls:

- `G`: capture current best target and grasp.
- `R`: resume live preview.
- `Q` / `Esc`: exit.

For hackathon use, always run `--dry-run` first to validate target pose and reachable workspace.

## How This Connects to MI300X + DRTC

The current MI300X/DRTC work already validates remote action-chunk serving for a synthetic client:

- Direct public endpoint: `165.245.129.215:18201`.
- Current viable DRTC schedule: `fps=15`, `actions_per_chunk=50`, `s_min=25`, `epsilon=2`, JPEG quality `40`.
- Current checkpoint horizon is `50`; `actions_per_chunk=100` requires a longer-horizon checkpoint.

The reBot path adds the real robot integration layer. The correct integration order is:

1. Prove local B601 teleoperation with `seeed_b601_dm_follower`.
2. Record a short dataset and inspect actual LeRobot feature names and action dimensions.
3. Write a B601-specific DRTC client adapter that maps:
   - B601 `observation_features` to remote policy observation schema.
   - Remote action vector back to B601 `send_action` dict.
4. If direct pi05 action semantics do not align, use DRTC as a system demo and keep low-level execution in local primitives.
5. Use Evo-RL/ACP only after demonstration data exists; it is not the first bring-up step.

Expected mismatch to resolve:

- The current `vlash-pi05-libero-async5` checkpoint returns `50 x 7` actions, but action names and normalization may not match B601.
- B601 state/action naming should be discovered from `robot.observation_features` and `robot.action_features` on the actual hardware.
- Camera key names must be stable; VLA checkpoints may expect names like `image` and `wrist_image`, while Seeed examples use `front`, `side`, or `up`.

## Hackathon Recommended Execution Plan

Day 0 / before venue:

- Prepare Seeed LeRobot environment on an Ubuntu x86 or Jetson machine.
- Prepare MI300X DRTC server and confirm public reachability.
- Prepare fallback local primitive scripts.
- Prepare printed ArUco marker for hand-eye calibration if using Orbbec grasping.

Day 1 at venue:

1. Verify device paths: `/dev/ttyUSB*`, `/dev/ttyACM*`, cameras.
2. Run leader calibration and raw-angle check.
3. Run leader-follower teleop without cameras.
4. Add one RGB camera; then add second RGB or Orbbec.
5. Record 3-5 very short episodes of a primitive task.
6. Replay one episode if safe.
7. If Orbbec is available, run detection-only and grasp-estimation-only checks.

Day 2:

- Use local primitives for reliable cooking steps.
- Use remote MI300X DRTC/VLA as a visible technical module for action proposal or policy demonstration.
- Keep human-in-the-loop fallback available; competition rules prohibit pure teleop as the whole solution, but allow semi-autonomous workflows.

## Open Questions to Verify on Real Hardware

- Actual follower serial path: `/dev/ttyACM0`, `/dev/ttyACM4`, `can0`, or another device.
- Actual leader type: `rebot_arm_102_leader` or B601 leader package.
- Actual `robot.action_features` and `robot.observation_features`.
- Whether the provided Orbbec camera is Gemini 2 and whether `pyorbbecsdk` works on the event machine.
- Whether the J4012 has JetPack 6.0/6.1 or an unsupported JetPack 6.2 image.
- Whether `motorbridge` installs cleanly on the target architecture.
- Whether the event network allows persistent outbound gRPC to the MI300X endpoint.

