# Hackathon reBot + MI300X + DRTC Execution Plan

Last updated: 2026-04-24

This is the main execution plan for the 2026 Robot Cooking Hackathon. It combines:

- The competition constraints from `drtc/202604_hunaochufang_hackathon.md`.
- Seeed reBot B601-DM LeRobot bring-up notes in `docs/REBOT_B601_LEROBOT_BRINGUP.md`.
- MI300X DRTC remote inference measurements in `docs/MI300X_REMOTE_INFERENCE_OPTIMIZATION_PLAN.md`.
- Evo-RL-Phi `pi05-rocm-acp` training/adaptation capability on MI300X.

## Executive Decision

The viable target is not "fully autonomous end-to-end VLA cooking". The viable target is:

> A semi-autonomous cooking robot system where reBot executes safe local primitives, local perception and/or an agent selects task steps, and MI300X provides remote VLA/DRTC policy capability as the main technical differentiator.

This is aligned with the competition rules because:

- Pure teleoperation is disallowed, but semi-autonomous operation is allowed.
- The scoring rewards VLA, imitation learning, target detection, Agent logic, GitHub/README, Jetson deployment, and camera usage.
- The event only gives 48 hours, so the system must prioritize reliable primitives and a clear demo over full retraining.

## Success Criteria

Minimum acceptable demo:

- Use the provided reBot B601-DM and camera setup.
- Complete at least two food-preparation workflows, one of which is a staple/main dish.
- Include algorithm-triggered actions for key steps; do not rely on continuous manual teleoperation.
- Show GitHub repo, README, architecture diagram, and demo video/logs.
- Have a human-safe stop/fallback process.

Technical-success demo:

- Bring up Seeed LeRobot stack on the event machine.
- Verify `seeed_b601_dm_follower` and `rebot_arm_102_leader`.
- Run at least one camera-driven autonomous primitive.
- Run local client to MI300X DRTC server and demonstrate remote action chunk streaming.
- Use local schedule parameters validated by prior testing: `fps=15`, `H=50`, `s_min=25`, `epsilon=2`, JPEG quality `40`.
- Show Evo-RL/ACP as the adaptation path, even if not used for final cooking control.

Stretch demo:

- Record reBot demonstrations in LeRobot format.
- Replay a recorded episode safely.
- Train a small ACT policy or run a short pi05/ACP adaptation smoke if time permits.
- Integrate VLA/Agent decisions into primitive selection.

## System Architecture

Recommended runtime split:

```text
Camera(s) + reBot B601-DM + Leader Arm
        |
        v
Event machine / reComputer J4012 / Ubuntu laptop
        |
        |-- Local safety controller
        |-- LeRobot robot wrapper: seeed_b601_dm_follower
        |-- Primitive library: grasp, move, pour, stir, place, wait
        |-- Perception: YOLO/OpenCV/Orbbec if available
        |-- Agent/task planner: selects next primitive
        |-- DRTC client: optional remote VLA action proposal
        |
        v
MI300X server 165.245.129.215
        |
        |-- drtc-Phi DRTC server
        |-- pi05/vlash checkpoint serving
        |-- Evo-RL-Phi pi05 ACP workflow for adaptation/training
```

Do not put safety-critical low-level control only in the remote loop. The network floor observed from local client to MI300X is roughly `250-270ms` even for tiny payloads, and action RTT can spike above `1s`. Remote inference is suitable for high-level policy proposals or chunked slow manipulation, not fast visual servoing.

## Current Verified Technical Assets

MI300X DRTC:

- Remote repo: `/mnt/models_alehe/phi-fbsh/drtc-Phi`.
- Public endpoint tested: `165.245.129.215:18201`.
- Current checkpoint: `/mnt/models_alehe/phi-fbsh/drtc-Phi/tmp/vlash-pi05-libero-async5-drtc-migrated`.
- Current action horizon: `50`.
- Split-client synthetic test result: `301` ticks, `0` stalls, `14` chunks received over 20 seconds.
- Median measured action RTT in the latest split-client run: about `514ms`.

Local DRTC tools:

- `scripts/run_high_latency_split_client.py`
- `scripts/benchmark_drtc_transport.py`
- `scripts/sweep_drtc_transport.py`
- `scripts/simulate_drtc_schedule.py`

Evo-RL-Phi on MI300X:

- Remote repo: `/mnt/models_alehe/phi-fbsh/Evo-RL-Phi`.
- Branch: `pi05-rocm-acp`.
- Validated MI300X pi05 environment and ACP staged workflow.
- Smoke/pilot/stage scripts under `scripts/experiments/pi05_acp/`.
- Useful for pi05 training/adaptation after data exists.
- Not a substitute for real reBot cooking data.

reBot/LeRobot:

- Official robot type: `seeed_b601_dm_follower`.
- Official teleop type in wiki: `rebot_arm_102_leader`.
- Official stack uses Seeed's LeRobot fork plus two local integration packages.
- Visual grasping demo exists using Orbbec Gemini 2 + YOLO + hand-eye calibration.

## Competition Strategy

The highest-probability path is a hybrid autonomy system:

1. Use local primitives to guarantee dish completion.
2. Use perception or agent logic to trigger key primitives, satisfying semi-autonomous requirements.
3. Use MI300X DRTC/VLA as a visible technical module, not as the only path to physical success.
4. Use Evo-RL/ACP as the "how this scales after collecting data" story.
5. Keep manual fallback for initialization, recovery, and safety.

Recommended dishes should be simple manipulation tasks:

- Main dish: instant noodles, simple mixed noodles, rice-ball assembly, or pre-cooked rice bowl assembly.
- Second dish: tomato/egg plating, salad assembly, sandwich/topping assembly, or drink/seasoning mix.

Avoid tasks requiring:

- High-force cutting.
- Fine deformable manipulation.
- Fast pan flipping.
- Precise temperature control.
- Food safety risk from raw meat.

## Workstreams

### Workstream A: Event Machine Bring-up

Goal: make the provided hardware controllable through Seeed LeRobot.

Steps:

1. Identify OS, JetPack, CUDA, Python, and PyTorch.
2. Install Seeed LeRobot stack.
3. Install B601 follower and 102 leader packages.
4. Install `motorbridge`.
5. Check `torch.cuda.is_available()` if using Jetson GPU.
6. Verify serial devices: `/dev/ttyUSB*`, `/dev/ttyACM*`, optionally `can0`.
7. Remove `brltty` if it blocks USB serial.
8. Run leader calibration and raw angle readout.
9. Run follower startup/calibration from the required initial pose.
10. Run leader-follower teleoperation without cameras.

Acceptance:

- The arm moves under leader control.
- Joint directions are correct.
- Gripper opens/closes.
- Operator can stop motion safely.

### Workstream B: Camera and Perception

Goal: establish stable visual input.

Steps:

1. Run `lerobot-find-cameras opencv`.
2. Bring up one RGB camera with `MJPG`.
3. Add second RGB camera only if the first is stable.
4. If Orbbec Gemini 2 is available, install `pyorbbecsdk` and udev rules.
5. Run Orbbec detection/grasp checks only after RGB and robot control are stable.
6. Record camera FPS and latency.

Acceptance:

- At least one camera stream is stable at usable FPS.
- Manipulated objects remain visible.
- Camera names are fixed and documented.

### Workstream C: Local Primitive Library

Goal: build reliable actions that can complete dishes.

Primitive candidates:

- `home()`
- `move_to_safe_pose()`
- `open_gripper()`
- `close_gripper()`
- `pick_from_pose()`
- `place_to_pose()`
- `pour_at_pose()`
- `stir_pattern()`
- `wait(seconds)`
- `emergency_stop()`

Implementation options:

- LeRobot `send_action` dicts.
- Replayed LeRobot episodes.
- MotorBridge / RebotArm SDK trajectory functions.
- Visual grasping demo's `RebotArm` IK/trajectory code if faster to adapt.

Acceptance:

- Each primitive has a dry-run mode or bounded test.
- Each primitive returns success/failure.
- Primitives do not exceed safe workspace boundaries.

### Workstream D: Data Collection and Replay

Goal: collect enough data to support replay, ACT fallback, and future pi05/ACP story.

Steps:

1. Record 3-5 short episodes for one simple primitive.
2. Visualize dataset with `lerobot-dataset-viz`.
3. Replay one episode if safe.
4. Record actual `robot.observation_features` and `robot.action_features`.
5. Keep `front`, `side`, or `up` camera names stable.

Acceptance:

- Dataset exists under `~/.cache/huggingface/lerobot/<repo_id>`.
- One episode can be inspected.
- Replay either works or failure mode is documented.

### Workstream E: MI300X DRTC Demo

Goal: demonstrate remote chunked VLA/policy serving and latency hiding.

Current known-good parameters:

```text
fps=15
actions_per_chunk=50
s_min=25
epsilon=2
jpeg_quality=40
```

Steps:

1. Keep MI300X DRTC server long-lived.
2. Confirm endpoint reachability from the event network.
3. Run synthetic split client as a network sanity check.
4. Run a real-observation client only after B601 features are known.
5. If action semantics do not align, present DRTC as action-proposal/system demo rather than physical controller.

Acceptance:

- Event machine can connect to MI300X.
- Synthetic split client receives chunks with no schedule starvation.
- Logs show action chunk flow and schedule size.

### Workstream F: Evo-RL/ACP Adaptation

Goal: frame a credible training/adaptation path.

Use Evo-RL-Phi for:

- Validated MI300X pi05 ROCm environment.
- ACP smoke/pilot/stage workflow.
- Post-event or stretch data adaptation once reBot demonstrations exist.

Do not depend on Evo-RL-Phi for:

- Day-one robot bring-up.
- Making an untrained checkpoint immediately control reBot cooking.
- Solving action-space mismatch without data.

Acceptance:

- README/PPT states Evo-RL/ACP is the adaptation layer.
- If time permits, run smoke or show existing validated stage result.
- Any new reBot dataset path is documented as future input to ACP.

## 48-Hour Schedule

### Before Arrival

- Confirm MI300X server reachable.
- Keep DRTC server scripts ready.
- Prepare Seeed LeRobot setup commands.
- Prepare GitHub README skeleton.
- Prepare safety checklist.
- Prepare simple dish plan with low-risk ingredients.

### Day 1 Morning: 09:00-11:30

- Check in and inspect hardware.
- Identify ports, OS, JetPack/CUDA, camera devices.
- Install or verify Seeed LeRobot environment.
- Fix serial permissions and `brltty` if needed.

Go/no-go:

- If robot cannot be controlled by noon, abandon VLA integration temporarily and focus on hardware bring-up.

### Day 1 Midday: 11:30-14:30

- Calibrate leader.
- Run raw angle check.
- Run follower teleop.
- Fix joint directions.
- Establish emergency stop procedure.

Go/no-go:

- If teleop is unstable, do not attempt policy control.

### Day 1 Afternoon: 14:30-17:30

- Bring up cameras.
- Record one small dataset.
- Test replay or primitive execution.
- Start local primitive library.

Go/no-go:

- If cameras are unstable, use fixed-pose primitives and document camera limitation.

### Day 1 Evening: 17:30-19:00

- Connect event machine to MI300X.
- Run synthetic split client.
- Capture logs/screenshots.
- Choose final dish workflows based on what is reliable.

### Day 2 Morning: 09:00-12:00

- Build final semi-autonomous sequence.
- Integrate perception trigger or agent trigger.
- Record demo videos.
- If Orbbec works, add visual grasping primitive.

### Day 2 Midday: 12:00-15:00

- Harden demo sequence.
- Add fallback buttons.
- Write README and architecture diagram.
- Collect logs/results for PPT.

### Day 2 Deadline Window: 15:00-17:00

- Stop adding features.
- Run final food demo for judges.
- Record clean video.
- Submit repo/materials.

### Day 2 Presentation: 17:00-19:00

Explain:

- Why split inference is needed.
- Why MI300X is used.
- How DRTC hides inference latency.
- Why local primitives preserve safety.
- How Evo-RL/ACP would adapt pi05 from collected demonstrations.

## Go/No-Go Matrix

| Condition | Decision |
| --- | --- |
| B601 teleop works, cameras work, MI300X reachable | Full hybrid autonomy demo |
| B601 teleop works, cameras work, MI300X unreachable | Local primitive + perception demo, show DRTC logs from pre-run |
| B601 teleop works, cameras fail | Fixed-pose primitive demo, use offline/recorded vision explanation |
| B601 teleop unstable | Use SDK/UI/manual-assisted safety demo, prioritize hardware recovery |
| Orbbec works | Add YOLO/depth grasping primitive |
| Orbbec fails | Use RGB/OpenCV or fixed workspace layout |
| No reBot dataset collected | Do not claim training success; present Evo-RL/ACP as future adaptation path |

## Risk Register

Highest risks:

- Jetson/Ubuntu environment mismatch.
- `motorbridge` or USB/CAN driver issues.
- Wrong joint directions or unsafe calibration pose.
- Camera bandwidth/USB hub instability.
- Orbbec SDK install time.
- Network to MI300X blocked or jittery.
- VLA checkpoint action semantics not matching B601.
- Not enough demonstrations for real policy training.

Mitigations:

- Verify teleop before all ML work.
- Use `MJPG` and one camera first.
- Keep all motion primitives bounded.
- Use remote VLA/DRTC as proposal/demo if direct control is unsafe.
- Keep a non-ML primitive fallback for cooking completion.
- Do not attempt full pi05 retraining unless core demo is already stable.

## Concrete Commands

Seeed LeRobot environment:

```bash
mkdir -p ~/rebot_lerobot
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

Serial permissions:

```bash
sudo chmod 666 /dev/ttyUSB* /dev/ttyACM*
```

Leader calibration:

```bash
lerobot-calibrate \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader
```

Teleoperation:

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

Camera discovery:

```bash
lerobot-find-cameras opencv
```

Record small dataset:

```bash
lerobot-record \
  --robot.type=seeed_b601_dm_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=rebot_arm_102_leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.id=rebot_arm_102_leader \
  --display_data=true \
  --dataset.repo_id=hackathon/rebot_cooking_smoke \
  --dataset.num_episodes=5 \
  --dataset.single_task="Pick and place one ingredient" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30
```

MI300X split-client check:

```bash
cd /path/to/drtc-Phi
.venv/bin/python scripts/run_high_latency_split_client.py \
  --server-address 165.245.129.215:18201 \
  --pretrained-name-or-path /mnt/models_alehe/phi-fbsh/drtc-Phi/tmp/vlash-pi05-libero-async5-drtc-migrated \
  --fps 15 \
  --actions-per-chunk 50 \
  --s-min 25 \
  --epsilon 2 \
  --jpeg-quality 40 \
  --duration-s 20 \
  --skip-policy-setup \
  --output results/drtc_high_latency_split_client_event.csv
```

Evo-RL-Phi smoke on MI300X:

```bash
cd /mnt/models_alehe/phi-fbsh/Evo-RL-Phi
source .venvs/pi05-openpi-ssp/bin/activate
pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py
bash scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh
```

## Repository Deliverables

Required:

- README explaining the project goal, setup, and demo.
- Hardware setup section.
- Safety section.
- Architecture diagram.
- Commands to reproduce reBot bring-up.
- Commands to reproduce MI300X DRTC split-client check.
- Demo video or screenshots.
- Known limitations.

Recommended docs already created:

- `docs/REBOT_B601_LEROBOT_BRINGUP.md`
- `docs/MI300X_REMOTE_INFERENCE_OPTIMIZATION_PLAN.md`
- `docs/HACKATHON_REBOT_MI300X_DRTC_EXECUTION_PLAN.md`
- `drtc/202604_hunaochufang_hackathon.md`

## What Not to Claim

Do not claim:

- The current `vlash-pi05-libero-async5` checkpoint is trained for reBot cooking.
- The current synthetic DRTC test validates physical action quality.
- Evo-RL/ACP can train a reliable cooking policy without reBot demonstrations.
- The remote MI300X loop is suitable for fast visual servoing.

Safe claims:

- MI300X remote DRTC serving is validated at the scheduling/transport level.
- reBot has an official LeRobot integration path.
- The system uses semi-autonomous primitives with remote VLA/DRTC policy capability.
- Evo-RL/ACP provides a concrete path to adapt pi05 once demonstrations are collected.

## Immediate Next Actions

1. Add this plan to the repo and sync it to MI300X.
2. Create a concise project README for the hackathon branch.
3. Prepare a one-command MI300X DRTC server restart script.
4. Prepare a one-command event-machine synthetic split-client test.
5. Prepare a hardware bring-up checklist printout.
6. On real hardware, inspect `robot.observation_features` and `robot.action_features` before writing any B601 policy adapter.

