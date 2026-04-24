# 胡闹厨房黑客松 DRTC 部署准备计划

## 目标

参加 2026 年 4 月胡闹厨房 Robot Cooking Hackathon，使用 DRTC 架构完成机械臂半自主/自主烹饪演示。

比赛约束：

- 必须使用主办方提供的 reBot B601 DM 机械臂、reComputer Robotics J4012、奥比相机等硬件。
- 需要完成不少于 2 道菜，其中必须包含 1 道主食。
- 禁止纯遥操作完成全部流程，关键烹饪步骤必须由算法触发和执行。
- 4 月 26 日 17:00 前完成现场烹饪演示和项目资料提交。

## 推荐系统架构

### 机器人端：reComputer Robotics J4012

职责：

- 运行 DRTC robot client。
- 连接 reBot B601 DM 机械臂。
- 采集奥比相机/USB 相机图像。
- 执行控制 loop、action schedule merge、cooldown、LWW register。
- 处理安全逻辑、急停、现场降级控制。
- 承担比赛评分项中的 Jetson 边缘部署展示。

J4012 是比赛现场必须重视的节点。即使最终大模型推理跑在外部机器，也应让 J4012 负责机器人闭环与主要部署入口。

### 开发/训练端 A：192.168.0.110 RTX 4070 服务器

当前本地开发用主力推理服务器，但不能带到比赛现场。

已确认信息：

- OS: Ubuntu 22.04.5 LTS
- GPU: NVIDIA GeForce RTX 4070
- Driver: 575.64.03
- CUDA: 12.9
- VRAM: 12GB
- 当前无 GPU 进程

用途：

- 先跑通 DRTC policy server。
- 验证真实 policy 的 chunk 推理耗时。
- 模拟外部/远端推理架构，给现场方案提供性能上限和对照基线。
- 赛前训练、离线 benchmark、调试 DRTC server/client。
- 测量 observation 传输、action chunk 返回、latency estimator、cooldown、schedule starvation。

注意：

- 这台 RTX 4070 无法带到比赛场地，因此不能作为现场执行依赖。
- RTX 4070 只能模拟 NVIDIA CUDA 软件生态和外部 policy server 角色。
- RTX 4070 不能模拟 J4012 的实际边缘算力、Jetson ARM64 环境、JetPack/L4T 驱动、Orin NX 内存带宽、相机 I/O 或机器人端实时性。
- 后续在 RTX 4070 上得到的真实模型 benchmark，只能证明“外部强算力 DRTC policy server 可行”和提供性能上限，不能推出“现场可直接使用同等算力”。

### 机器人端模拟：J4012-like client

由于当前本地没有 reComputer Robotics J4012，需要用另一台机器临时模拟 J4012 的角色。

模拟目标不是复现 J4012 的性能，而是复现机器人端职责：

- 运行 `RobotClientDrtc`。
- 采集或模拟 observation。
- 发送 observation 到远端 policy server。
- 接收 action chunk。
- 执行 action schedule、LWW、cooldown、latency estimator。
- 验证网络中断、推理中断时的安全停止/降级逻辑。

当前可用模拟方式：

- 短期：本地 Mac 运行 DRTC client，连接 `192.168.0.110` 上的 DRTC server。
- 更接近现场：找一台普通 Linux x86 小主机运行 DRTC client。
- 最终：拿到 J4012 后，把同样的 client-only smoke 平移到 J4012 上复测。

当前已经验证的完整环路：

```text
本地 Mac mock client -> 局域网 -> 192.168.0.110 RTX 4070 mock policy server -> action chunk -> 本地 Mac mock robot
```

下一步需要增强该模拟：

- mock robot observation 不能只用 `64x64` 随机图。
- 需要模拟比赛相机 payload，例如：
  - 1 路 `640x480`
  - 2 路 `640x480`
  - 1 路 `800x600`
  - JPEG quality 60
- 这样测到的 RTT、payload encode/decode 成本、网络抖动才更接近 J4012 现场链路。

### 现场推理端候选 B：AMD Ryzen AI 9 HX PRO 370

可带到比赛现场，因此从备选探索节点提升为现场 policy server 候选。

已确认信息：

- Hostname: `amd`
- CPU: AMD Ryzen AI 9 HX PRO 370 w/ Radeon 890M
- 12 cores / 24 threads
- Radeon 890M iGPU, ROCm 可识别为 `gfx1150`
- 约 54GiB 内存
- 当前未安装 PyTorch

风险：

- 该机器不是 Ryzen AI Max+ 395 / Strix Halo。
- NPU TOPS 不等价于 PyTorch/VLA 可用推理性能。
- LeRobot、SmolVLA、PI0、RTC flow policy 在 NVIDIA CUDA 路线更成熟。
- ROCm 在 Radeon 890M APU 上需要实测稳定性、算子覆盖和长时间运行可靠性。

定位：

- P0/P1 现场算力验证对象。
- 如果 ACT benchmark 和 DRTC server smoke 通过，可作为比赛现场外部 policy server。
- 如果 ROCm/PyTorch 不稳定，则降级为展示/探索节点，现场主流程必须回到 J4012 本地轻量方案和状态机/技能库。
- 当前不应先押 SmolVLA；优先验证 ACT 和轻量视觉策略。

## DRTC 方案价值

DRTC 适合当前比赛场景，因为：

- 机器人端和推理端可以分离。
- J4012 负责低延迟执行，外部机器负责重模型推理。
- cooldown 可以在 action chunk 丢失或延迟时恢复，而不会每 tick 疯狂触发推理。
- LWW register 可以吸收重复、乱序、旧 observation/action chunk。
- action schedule merge 允许新 chunk 覆盖未执行动作，减少模式切换。
- RTC in-painting 可以改善相邻 action chunk 之间的轨迹连续性。

当前 codebase 中的关键文件：

- `src/lerobot/async_inference/robot_client_drtc.py`
- `src/lerobot/async_inference/policy_server_drtc.py`
- `src/lerobot/async_inference/lww_register.py`
- `src/lerobot/async_inference/rtc_guidance.py`
- `src/lerobot/async_inference/utils/latency_estimation.py`
- `examples/experiments/configs/*.yaml`
- `scripts/start_drtc*.sh`

## 验证计划

### 阶段 1：本地 DRTC 通路跑通

目标：先在本地或局域网内跑通 DRTC client/server，不接真实机器人。

任务：

- 在 192.168.0.110 上安装项目依赖和 PyTorch CUDA。
- 启动 `policy_server_drtc`，先使用 mock policy。
- 在本地或另一台机器启动 `robot_client_drtc`，先使用 mock robot。
- 确认 observation 能发送，action chunk 能返回，client 能执行 mock action。
- 记录 baseline 的 RTT、chunk gap、schedule size、starvation。

验收：

- 连续运行 10 分钟不崩溃。
- action chunk 持续返回。
- starvation 事件可解释且数量可控。
- metrics 能落盘并可用于分析。

### 阶段 2：网络与传输评估

目标：先用 J4012-like client 验证机器人端到外部推理服务器的网络能否支撑实时控制；拿到 J4012 后复测真机。

任务：

- 测量当前 client 模拟机到 192.168.0.110 的 ping RTT 和抖动。
- 使用接近奥比/USB 相机的尺寸和 JPEG 压缩配置，测 observation payload 传输耗时。
- 增强 `tools/drtc_mock_smoke.py`，支持配置 mock camera 路数、分辨率、JPEG quality。
- 分别测试有线网络、现场 Wi-Fi、手机热点或备用路由。
- 注入 drop/reorder/latency spike，观察 DRTC 恢复行为。
- 拿到 J4012 后，在 J4012 上复跑相同 client-only smoke。

建议指标：

- 优先使用有线网络。
- 控制 FPS 先从 20-30Hz 开始，不要一开始追 60Hz。
- 记录 p50/p90/p99 RTT，而不只看平均值。

验收：

- 在目标 FPS 下，schedule 不长期清空。
- cooldown 能在 action/observation drop 后重新触发推理。
- latency estimator 不因单次 spike 长时间保守过度。

### 阶段 3：真实模型推理 benchmark

目标：确认模型推理是否满足 action chunk 实时性。

任务：

- 在 RTX 4070 上跑目标 policy 的 `predict_action_chunk` benchmark。
- 记录不同 `actions_per_chunk`、`num_flow_matching_steps`、图像分辨率下的耗时。
- 测试 RTC enabled/disabled 的额外耗时。
- 对比 SmolVLA、ACT、简单视觉检测 + 状态机方案。
- 在 AMD Ryzen AI 9 HX PRO 370 上安装 ROCm PyTorch 后做相同 benchmark。

优先指标：

- chunk 推理 wall-clock latency。
- 端到端 RTT：采集 observation 到收到 action chunk。
- 实际机器人执行时的 action discontinuity。
- 长时间运行稳定性和温度。

验收：

- RTX 4070 路线有可用性能数据。
- AMD 路线至少完成最小 PyTorch ROCm smoke test。
- 若 AMD 路线性能或稳定性不足，明确降为备用。

### 阶段 4：J4012 + reBot 集成

目标：拿到 J4012 和 reBot 后，把比赛硬件链路打通；当前阶段只能准备 checklist 和可平移 smoke。

任务：

- 熟悉 reBot B601 DM Python SDK。
- 跑通 reBot 的基础 joint 控制、夹爪控制、回零、安全限位。
- 跑通 LeRobot 对 reBot 的集成示例。
- 确认 action space、observation.state、camera keys 与 DRTC client 对齐。
- 在 J4012 上部署 robot client。
- 用 192.168.0.110 作为 policy server 做局域网闭环。

验收：

- J4012 能采集图像并发送 observation。
- 机械臂能执行从 policy server 返回的 action。
- 出现网络或推理故障时，机械臂有安全停止策略。

当前限制：

- 目前本地没有 J4012，不能验证 Jetson ARM64/JetPack/Orin NX 性能。
- RTX 4070 不能替代 J4012 做边缘性能评估。
- 所有 J4012 性能相关判断必须标记为“待真机验证”。

### 阶段 5：烹饪任务设计

目标：选择可控、可重复、评分收益高的菜品和自动化步骤。

建议策略：

- 选择动作简单、容错高、食材形态稳定的菜。
- 必须包含 1 道主食，优先考虑面包片/饭团/简易面食/预处理主食装盘等低风险方案。
- 另一道菜可以选择番茄鸡蛋、拌菜、煎蛋、倒料混合类任务。
- 把复杂烹饪拆成技能：识别、抓取、移动到锅/碗、倒料、搅拌、装盘。

技术展示重点：

- 视觉识别食材/容器位置。
- Agent 或状态机调度技能。
- 模仿学习/VLA 负责关键轨迹。
- DRTC 负责远端推理实时控制和故障恢复。

## 比赛降级方案

### Level 0：完整方案

J4012 运行 DRTC client，外部 RTX/AMD 运行 VLA 或模仿学习 policy server，完成多个关键动作的自主闭环。

### Level 1：混合技能方案

视觉检测 + 状态机负责流程，部分关键动作使用示教轨迹或 policy chunk。DRTC 仍用于远端 chunk 推理。

### Level 2：半自主演示方案

关键动作由算法触发，轨迹主要来自预录制技能库。遥操作只用于初始化、纠偏和安全处理，不作为完整流程主体。

### Level 3：保底方案

J4012 本地运行简单检测和固定轨迹，外部 server 只做辅助识别或计划。保证能稳定做出菜和完成汇报。

## 风险清单

### 算力风险

- AMD Ryzen AI 9 HX PRO 370 的 ROCm/PyTorch 支持不确定。
- NPU 未必能直接服务 LeRobot/VLA 推理。
- RTX 4070 服务器不一定能在比赛现场使用，需要确认网络可达性和供电/部署条件。

缓解：

- 主线保留 NVIDIA CUDA 路线。
- J4012 本地准备可运行的轻量方案。
- AMD 只在 benchmark 通过后提升为可用节点。

### 网络风险

- 现场 Wi-Fi 拥塞。
- observation 图像 payload 过大导致 RTT 抖动。
- 远端 server 连接不可用。

缓解：

- 优先有线连接。
- JPEG 压缩，限制分辨率和相机数量。
- 准备本地 fallback。
- 用 DRTC metrics 监控 latency_steps、cooldown、starvation。

### 机器人风险

- reBot action space 与现有策略不匹配。
- 机械臂精度、负载、夹爪能力不足。
- 烹饪环境导致安全员中止。

缓解：

- 选低风险菜品和动作。
- 提前测试抓取容器、倒料、搅拌。
- 所有热源和刀具相关动作尽量规避。
- 设计清晰的急停和人工接管流程。

### 比赛评分风险

- 只展示远端大模型但没有 J4012/奥比部署，会丢完整度分。
- 只做遥操作会被判违规。
- 菜品不好吃会直接影响 30% 味道分。

缓解：

- PPT 明确展示 J4012、奥比相机、DRTC、GitHub README。
- 每个菜至少有几个关键步骤由算法触发。
- 选择味道稳定的菜，必要时用预处理食材提升成功率。

## 近期行动清单

1. 在 AMD 机器上安装/确认 ROCm PyTorch，跑 `torch` smoke test。
2. 在 AMD 机器上跑 ACT `random-act` 和 SO101 ACT checkpoint benchmark。
3. 如果 AMD ACT benchmark 可用，启动 AMD DRTC policy server，跑 J4012-like client -> AMD server 的 60 秒 smoke。
4. 设计两道菜的最小可行流程和技能列表。
5. 采集/准备一个比赛相关的 ACT 示教数据集，训练/微调一个稳定技能 checkpoint。
6. 建立 metrics 记录模板：RTT、latency_steps、schedule_size、starvation、chunk_gap。
7. 梳理 reBot B601 DM 的 action space、SDK、LeRobot 集成方式。
8. 准备 GitHub README/PPT 框架，提前对齐评分项。

## 推荐优先级

P0：

- AMD ROCm/PyTorch smoke test。
- AMD ACT policy benchmark。
- J4012-like client 到 AMD server 的 DRTC smoke。
- 两道菜最小技能链设计。

P1：

- 比赛相关 ACT 技能示教数据采集/训练。
- 相机 observation 传输优化。
- RTC in-painting 参数调试。
- 故障注入实验。
- 准备 J4012 bring-up checklist。

P2：

- J4012/reBot 控制链路确认（待真机）。
- RTX 4070 继续作为赛前训练/benchmark 对照。
- NPU/ONNX/Vitis/Ryzen AI 软件栈探索。

## 当前结论

DRTC + J4012 client + 外部 policy server 是符合比赛场景的架构。

192.168.0.110 RTX 4070 服务器适合作为赛前开发、训练和性能上限参考，但不能作为现场执行依赖。

AMD Ryzen AI 9 HX PRO 370 可以带到比赛现场，因此必须尽快完成 ACT/DRTC 最小验证；验证通过后可作为现场外部 policy server，验证失败则降级为展示/探索节点。

比赛最终目标不是证明某个硬件概念，而是稳定完成烹饪、符合自主/半自主规则，并把 J4012、相机、DRTC、策略模型和工程完整度清楚展示出来。

## 执行记录

### 2026-04-24 P0：DRTC mock 链路

新增工具：

- `tools/drtc_mock_smoke.py`

用途：

- 使用 `RobotClientDrtc(use_mock_robot=True)` 模拟机器人端。
- 使用 `PolicyServerDrtc(mock_policy=True)` 模拟策略端。
- 支持本地一体化 smoke，也支持 server/client 分离验证。
- 支持通过 `--camera-count`、`--camera-width`、`--camera-height` 模拟 observation 图像 payload。
- 支持通过 `--json-output` 将 smoke 结果落盘，便于后续复测对比。

本地环境：

- 新建 `.venv-drtc`
- 使用 Python 3.11
- 安装 DRTC mock 路径所需最小依赖
- 注意：macOS 本地同时安装 `av` 和 `opencv-python-headless` 后，会出现 FFmpeg/AVFoundation duplicate class warning。当前 smoke 可运行，但后续真实相机链路应在 Linux/J4012 上优先验证。

本地一体化 smoke：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --duration-s 8 --fps 20`
- 结果：通过
- received_chunks: 22
- executed_actions: 150
- final_action_step: 149
- final_schedule_size: 18
- latency_steps: 1
- latency_ms: 25.03

远端 `192.168.0.110` 环境：

- 已同步仓库到 `/home/fbsh/drtc`
- 系统缺 `python3.10-venv`，且 sudo 需要密码，无法用系统 venv。
- 使用已有 Miniconda 新建环境：`drtc-smoke`
- 环境路径：`/home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke`
- 最初安装 PyTorch 2.11.0 + CUDA 13 wheel 后，`torch.cuda.is_available()` 为 False，因为当前 NVIDIA driver 575.64.03 对 CUDA 13 wheel 不够新。
- 已降级到 PyTorch 2.7.1 + CUDA 12.8 wheel。
- CUDA smoke test 已通过：
  - `torch 2.7.1+cu128`
  - `cuda_version 12.8`
  - `cuda_available True`
  - `device0 NVIDIA GeForce RTX 4070`
  - `1024x1024` CUDA matmul 通过
- 降级 PyTorch 后将 `fsspec` 固定回 `2025.9.0`，以满足 `datasets 4.1.1` 约束。

远端一体化 smoke：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_mock_smoke.py --duration-s 8 --fps 20`
- 结果：通过
- received_chunks: 22
- executed_actions: 150
- final_action_step: 149
- final_schedule_size: 18
- latency_steps: 1
- latency_ms: 23.25

远端 CUDA 修复后快速 smoke：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_mock_smoke.py --duration-s 3 --fps 20`
- 结果：通过
- received_chunks: 7
- executed_actions: 50
- final_action_step: 49
- final_schedule_size: 13
- latency_steps: 1
- latency_ms: 16.21

远端 server + 本地 client 分离 smoke：

- 远端 server：`192.168.0.110:18080`
- 本地 client 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20`
- 结果：通过
- received_chunks: 22
- executed_actions: 150
- final_action_step: 149
- final_schedule_size: 18
- latency_steps: 1
- latency_ms: 32.04

远端 server + 本地 client + 相机 payload smoke：

1 路 `640x480`：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20 --camera-count 1 --camera-width 640 --camera-height 480`
- 结果：通过
- received_chunks: 22
- executed_actions: 148
- final_action_step: 147
- final_schedule_size: 12
- latency_steps: 1
- latency_ms: 40.43
- diagnostic RTT avg/max 约 `34.58/61.27ms`

1 路 `320x240`：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20 --camera-count 1 --camera-width 320 --camera-height 240`
- 结果：通过
- received_chunks: 22
- executed_actions: 149
- final_action_step: 148
- final_schedule_size: 12
- latency_steps: 1
- latency_ms: 28.59
- diagnostic RTT avg/max 约 `24.15/32.12ms`

2 路 `320x240`：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20 --camera-count 2 --camera-width 320 --camera-height 240`
- 结果：通过
- received_chunks: 22
- executed_actions: 150
- final_action_step: 149
- final_schedule_size: 18
- latency_steps: 1
- latency_ms: 33.38
- diagnostic RTT avg/max 约 `28.23/34.42ms`

2 路 `640x480`：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20 --camera-count 2 --camera-width 640 --camera-height 480`
- 结果：通过
- received_chunks: 21
- executed_actions: 148
- final_action_step: 147
- final_schedule_size: 12
- latency_steps: 2
- latency_ms: 68.77
- diagnostic RTT avg/max 约 `57.21/88.38ms`

2 路 `640x480`，60 秒稳定性：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 60 --fps 20 --camera-count 2 --camera-width 640 --camera-height 480`
- 结果：通过
- received_chunks: 170
- executed_actions: 1188
- final_action_step: 1187
- final_schedule_size: 15
- latency_steps: 2
- latency_ms: 55.48
- diagnostic RTT avg/max 约 `53.97/135.25ms`

1 路 `800x600`：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 8 --fps 20 --camera-count 1 --camera-width 800 --camera-height 600`
- 结果：通过
- received_chunks: 21
- executed_actions: 145
- final_action_step: 144
- final_schedule_size: 17
- latency_steps: 2
- latency_ms: 78.27
- diagnostic RTT avg/max 约 `48.91/190.82ms`

2 路 `320x240`，60 秒稳定性：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 60 --fps 20 --camera-count 2 --camera-width 320 --camera-height 240`
- 结果：通过
- received_chunks: 170
- executed_actions: 1190
- final_action_step: 1189
- final_schedule_size: 14
- latency_steps: 1
- latency_ms: 30.42
- diagnostic RTT avg/max 约 `26.27/36.28ms`

2 路 `320x240`，20 秒结构化结果落盘：

- 命令：`PYTHONPATH=src python tools/drtc_mock_smoke.py --mode client --server-address 192.168.0.110:18080 --duration-s 20 --fps 20 --camera-count 2 --camera-width 320 --camera-height 240 --json-output artifacts/drtc_mock_2x320x240_20s.json`
- 结果：通过
- 输出文件：`artifacts/drtc_mock_2x320x240_20s.json`
- received_chunks: 56
- executed_actions: 389
- final_action_step: 388
- final_schedule_size: 17
- latency_steps: 1
- latency_ms: 34.69
- diagnostic RTT avg/max 约 `30.56/45.96ms`

观察：

- mock policy 下，当前局域网分离链路 RTT 约 32ms，20Hz 控制下 `latency_steps=1`。
- `schedule_size` 在 smoke 过程中维持在 13-18 左右，没有 starvation。
- stop 时 action stream 会打印 `StatusCode.CANCELLED`，这是 client 主动取消 stream 的预期现象，不是链路失败。
- 加入相机 payload 后，RTT 明显上升：
  - 1 路 `320x240` 约 `29ms`，2 路 `320x240` 约 `33ms`。
  - 2 路 `320x240` 60 秒 run 稳定保持 `latency_steps=1`，没有 starvation。
  - 1 路 `640x480` 仍基本能维持 `latency_steps=1`。
  - 2 路 `640x480` 会稳定进入 `latency_steps=2`，60 秒 run 没有 starvation，但出现过约 `135ms` 尖峰。
  - 1 路 `800x600` 会进入 `latency_steps=2`。
  - 1 路 `800x600` 出现过约 `190ms` 的尖峰，后续估计恢复，但说明大图 payload 对抖动敏感。
- 当前 mock 图像是随机噪声，JPEG 压缩效果会比真实相机画面差，因此这组结果偏保守；真实画面可能更小，但现场网络可能更差。
- 当前默认建议：
  - 比赛首选传输配置：2 路 `320x240`，20Hz 起步。
  - 如果视觉任务需要更高分辨率，优先尝试 1 路 `640x480` 主视角 + 低频辅助视角。
  - 2 路 `640x480` 可以作为高分辨率调试/识别模式，或在有线网络确认稳定后作为备选闭环配置。
  - 不建议一开始使用 1 路 `800x600` 作为控制闭环默认输入。

下一步：

- 继续测 SmolVLA/候选 checkpoint，并把真实 checkpoint 接入 DRTC server。
- 给 smoke 增加 metrics CSV 输出路径，便于长期记录 p50/p90/p99 latency。
- 准备 J4012 端复现同样 client-only smoke。
- 相机 payload 继续补测：
  - 1 路 `640x480` + 1 路 `320x240` 混合方案
  - 后续接真实相机帧替代随机噪声

### 2026-04-24 P0：RTX 4070 policy 推理 benchmark

新增工具：

- `tools/drtc_policy_benchmark.py`

用途：

- `synthetic`：只测 CUDA tensor baseline，不代表真实 policy。
- `random-act`：构造随机初始化但结构真实的 ACT 视觉策略，调用 `predict_action_chunk`，不依赖 checkpoint 下载。
- `policy`：从 checkpoint 加载 LeRobot policy，调用 `predict_action_chunk`，用于后续 SmolVLA/ACT 候选模型实测。

已完成的 RTX 4070 CUDA baseline：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_policy_benchmark.py --mode synthetic --device cuda --warmup 20 --iters 200 --actions-per-chunk 50 --action-dim 6 --synthetic-hidden 2048`
- 结果：通过
- torch: `2.7.1+cu128`
- CUDA: `12.8`
- device: `NVIDIA GeForce RTX 4070`
- mean: `0.097ms`
- p95: `0.105ms`
- p99: `0.110ms`

已完成的随机 ACT 结构 benchmark：

2 路 `320x240`，`chunk_size=50`：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_policy_benchmark.py --mode random-act --device cuda --warmup 10 --iters 50 --actions-per-chunk 50 --action-dim 6 --state-dim 6 --camera-count 2 --camera-width 320 --camera-height 240 --json`
- 结果：通过
- mean: `16.57ms`
- median: `16.53ms`
- p95: `16.88ms`
- p99: `17.05ms`
- max: `17.17ms`

2 路 `640x480`，`chunk_size=50`：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_policy_benchmark.py --mode random-act --device cuda --warmup 10 --iters 50 --actions-per-chunk 50 --action-dim 6 --state-dim 6 --camera-count 2 --camera-width 640 --camera-height 480 --json`
- 结果：通过
- mean: `17.92ms`
- median: `17.88ms`
- p95: `18.19ms`
- p99: `18.21ms`
- max: `18.21ms`

解释：

- 在 RTX 4070 上，随机 ACT 视觉结构本身可以稳定低于一个 20Hz control tick 的 `50ms`。
- 对比前面的 DRTC payload smoke，2 路 `640x480` 端到端延迟升高主要来自 observation payload 传输/编码/解码和网络抖动，而不是 ACT 网络前向。
- 这不能代表 SmolVLA/PI0/flow policy 的最终耗时；VLA checkpoint 仍需单独测，尤其是语言编码、扩散/flow inference steps、processor/postprocessor 和显存占用。
- 这也不能代表 J4012 本地推理性能；当前结论只覆盖“外部 RTX 4070 policy server”。

真实 SmolVLA checkpoint benchmark 尝试：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_policy_benchmark.py --mode policy --device cuda --policy-type smolvla --pretrained-name-or-path jackvial/so101_smolvla_pickplaceorangecube_e100 --warmup 2 --iters 5 --actions-per-chunk 50 --state-dim 6 --task "pick place orange cube" --json`
- 结果：未完成
- 已安装依赖：`transformers 4.57.6`、`num2words 0.5.14`
- 阻塞原因：`192.168.0.110` 当前无法访问 Hugging Face，报错 `Network is unreachable`，本地 cache 里也没有该 checkpoint 的 `config.json`。

影响：

- 真实 checkpoint benchmark 需要提前离线准备，不能假设比赛现场或本地服务器有外网。
- 后续应把候选模型 checkpoint 下载到本地，再 rsync 到 `192.168.0.110`，用本地路径运行 `--pretrained-name-or-path`。
- 当前 `random-act` 结果只能证明 ACT 类视觉网络前向预算，不等价于 SmolVLA/PI0 的最终性能。

离线 checkpoint 准备：

- 本地已下载 fine-tuned policy：`checkpoints/jackvial_so101_smolvla_pickplaceorangecube_e100`
- 大小：约 `865M`
- 已同步到 4070：`/home/fbsh/drtc/checkpoints/jackvial_so101_smolvla_pickplaceorangecube_e100`
- 该 checkpoint 的 config 依赖 base VLM：`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- base VLM 全量仓库约 `7.55GiB`，其中 ONNX 导出占大头。
- 本地已下载 base VLM PyTorch/processor 文件，排除 `onnx/*`：`checkpoints/HuggingFaceTB_SmolVLM2-500M-Video-Instruct`
- base VLM PyTorch 权重约 `1.9G`，但当前 benchmark 使用 `--no-load-vlm-weights`，初始化阶段只需要 config/processor/tokenizer。
- 已同步到 4070 的 base VLM 元数据/tokenizer 路径：`/home/fbsh/drtc/checkpoints/HuggingFaceTB_SmolVLM2-500M-Video-Instruct`
- 远端 base VLM 目录大小：约 `4.7M`

Benchmark 脚本更新：

- `tools/drtc_policy_benchmark.py` 增加 `--vlm-model-name`，用于把 SmolVLA 的 `vlm_model_name` 覆盖成本地路径。
- 增加 `--load-vlm-weights/--no-load-vlm-weights`，用于控制是否在初始化阶段加载 base VLM 权重。
- 加载 saved preprocessor 时同步 override `tokenizer_processor.tokenizer_name`，避免 tokenizer 继续访问 Hugging Face。

DRTC server/client 离线 policy override 更新：

- `RobotClientDrtcConfig` 新增：
  - `policy_vlm_model_name`
  - `policy_load_vlm_weights`
- `RemotePolicyConfig` 新增：
  - `vlm_model_name`
  - `load_vlm_weights`
- `PolicyServerDrtc` 加载 policy 时会把上述字段写入 policy config。
- `PolicyServerDrtc` 加载 saved preprocessor 时会 override `tokenizer_processor.tokenizer_name`，避免 tokenizer 继续使用 Hugging Face repo id。
- `tools/drtc_mock_smoke.py` 新增真实 policy server 模式：
  - `--policy-server-mode real`
  - `--policy-type`
  - `--pretrained-name-or-path`
  - `--policy-device`
  - `--policy-vlm-model-name`
  - `--policy-load-vlm-weights/--no-policy-load-vlm-weights`
  - `--actions-per-chunk`
  - `--num-flow-matching-steps`
- `tools/drtc_mock_smoke.py` 在真实 policy 模式下会把 mock robot 的 state/action/camera features 转成 dataset-style feature dict，确保 server preprocessing 真实消费 observation。

SmolVLA 本地 checkpoint benchmark：

共同设置：

- policy：`/home/fbsh/drtc/checkpoints/jackvial_so101_smolvla_pickplaceorangecube_e100`
- base VLM metadata/tokenizer：`/home/fbsh/drtc/checkpoints/HuggingFaceTB_SmolVLM2-500M-Video-Instruct`
- device：RTX 4070 CUDA
- torch：`2.7.1+cu128`
- chunk_size：`50`
- action_dim/state_dim：`6`
- `--no-load-vlm-weights`
- 说明：`--no-load-vlm-weights` 跳过初始化阶段加载 base VLM 的 1.9G 权重；随后仍加载 fine-tuned policy 的 `model.safetensors`。该结果适合做本 checkpoint 的端到端 latency 估算，但最终比赛前仍建议用完整离线依赖再复测一次。

默认 `num_steps=10`：

- mean: `452.44ms`
- median: `452.20ms`
- p99: `453.02ms`
- load_ms: `22860.77ms`

`num_steps=4`：

- mean: `256.34ms`
- median: `252.85ms`
- p99: `269.71ms`
- load_ms: `23930.32ms`

`num_steps=2`：

- mean: `177.12ms`
- median: `168.17ms`
- p99: `192.58ms`
- load_ms: `24174.58ms`

`num_steps=1`：

- mean: `117.20ms`
- median: `117.17ms`
- p99: `117.44ms`
- load_ms: `23140.73ms`

解释：

- 该 SmolVLA checkpoint 在 RTX 4070 上能离线加载并推理。
- 默认 `num_steps=10` 约 `452ms/chunk`，对 20Hz 控制 loop 来说不是 tick 级实时策略。
- DRTC 可以把它作为 chunk 级远端重模型使用，但需要足够长的 action horizon、稳定的 schedule buffer、较低触发频率和明确 fallback。
- 即使降到 `num_steps=1`，仍约 `117ms/chunk`，端到端再叠加 observation 传输后会超过此前 mock 链路的 `30-55ms`。
- 当前比赛主线应保留更轻策略/ACT/视觉检测+状态机作为 Level 1 fallback；SmolVLA 更适合展示 VLA/DRTC 架构或处理低频关键技能。

SmolVLA 真实 DRTC server/client smoke：

共同设置：

- server：`192.168.0.110:18080`
- server mode：`real`
- client：本地 Mac mock robot，模拟 J4012-like client
- policy：`smolvla`
- checkpoint：`/home/fbsh/drtc/checkpoints/jackvial_so101_smolvla_pickplaceorangecube_e100`
- local VLM metadata/tokenizer：`/home/fbsh/drtc/checkpoints/HuggingFaceTB_SmolVLM2-500M-Video-Instruct`
- `--no-policy-load-vlm-weights`
- `--num-flow-matching-steps 1`
- `actions_per_chunk=50`
- `fps=20`
- observation payload：2 路 `320x240` 随机 RGB 图像，JPEG quality 60

15 秒首次真实 smoke：

- 输出文件：`artifacts/drtc_real_smolvla_2x320x240_15s.json`
- 结果：通过
- received_chunks: 15
- executed_actions: 272
- final_action_step: 271
- final_schedule_size: 41
- latency_steps: 13
- latency_ms: 600.81
- 说明：这轮包含首次模型加载/预热后早期估计收敛影响，latency estimator 偏保守。

60 秒稳定性 smoke：

- 输出文件：`artifacts/drtc_real_smolvla_2x320x240_60s.json`
- 结果：通过
- received_chunks: 165
- executed_actions: 1180
- final_action_step: 1179
- final_schedule_size: 41
- latency_steps: 5
- latency_ms: 227.59
- diagnostic RTT avg 最终约 `216ms`
- diagnostic RTT max 约 `412.53ms`
- 期间 schedule size 稳定保持约 `37-44`，没有 starvation。

解释：

- 这已经不是单独 `predict_action_chunk` benchmark，而是完整 DRTC 环路：
  - 本地 mock robot capture
  - JPEG observation payload
  - gRPC 发送到 4070
  - 4070 真实 SmolVLA `predict_action_chunk`
  - action chunk 返回
  - client schedule merge + mock action execution
- 对比 mock policy 的 2 路 `320x240` 60 秒 run：真实 SmolVLA 将端到端 latency 从约 `30ms` 推高到约 `216-228ms`。
- DRTC 的 chunk buffer 能撑住该延迟，因为 `actions_per_chunk=50` 在 20Hz 下约等于 `2.5s` action horizon。
- 该设置适合作为远端重模型 chunk policy 验证；如果要做更灵敏的接触/精细操作，仍应准备轻量 policy 或技能状态机。

### 2026-04-24 P0：ACT 结构真实 DRTC 环路

背景：

- 当前本地没有可直接用于比赛任务的 ACT checkpoint。
- `tests/artifacts/policies/aloha_sim_insertion_human_act_*` 只包含测试用 safetensors 输出/统计，不是完整 `from_pretrained` policy 目录。
- 为了先评估 ACT 架构在完整 DRTC client/server 下的环路表现，新增 `tools/drtc_mock_smoke.py --policy-server-mode random-act`。

工具更新：

- `tools/drtc_mock_smoke.py` 新增 `RandomActPolicyServer`。
- server 在 handshake 后根据 client 传来的 state/action/camera feature 构造真实 `ACTPolicy`。
- policy 使用随机初始化权重、`pretrained_backbone_weights=None`、`use_vae=False`。
- server 仍走真实 DRTC observation preprocessing、`predict_action_chunk`、action chunk pack、gRPC stream。
- 这不是任务性能评估，只是 ACT 架构和 DRTC 环路性能评估。

随机 ACT 真实 DRTC smoke，2 路 `320x240`，60 秒：

- 输出文件：`artifacts/drtc_random_act_2x320x240_60s.json`
- 结果：通过
- received_chunks: 154
- executed_actions: 1178
- final_action_step: 1177
- final_schedule_size: 47
- latency_steps: 2
- latency_ms: 82.24
- diagnostic RTT avg 最终约 `84ms`
- diagnostic RTT max 约 `572.28ms`，主要来自早期冷启动/估计收敛。

随机 ACT 真实 DRTC smoke，2 路 `640x480`，60 秒：

- 输出文件：`artifacts/drtc_random_act_2x640x480_60s.json`
- 结果：通过
- received_chunks: 169
- executed_actions: 1186
- final_action_step: 1185
- final_schedule_size: 42
- latency_steps: 3
- latency_ms: 140.01
- diagnostic RTT avg 最终约 `127ms`
- diagnostic RTT max 约 `405.82ms`

ACT vs SmolVLA 当前实测对比：

- 随机 ACT 单独前向：
  - 2 路 `320x240`：约 `16.57ms`
  - 2 路 `640x480`：约 `17.92ms`
- 随机 ACT 完整 DRTC 环路：
  - 2 路 `320x240`：约 `82ms`
  - 2 路 `640x480`：约 `140ms`
- SmolVLA 单独前向：
  - `num_steps=1`：约 `117ms`
  - `num_steps=10`：约 `452ms`
- SmolVLA 完整 DRTC 环路：
  - 2 路 `320x240`、`num_steps=1`：约 `228ms`

解释：

- ACT 架构本身比 SmolVLA 明显更适合比赛里的高频、可重复技能。
- 完整 DRTC 环路下，ACT 的主要成本不再是 policy 前向，而是图像 encode/decode、gRPC transport、preprocessing、latency estimator 初期收敛和网络抖动。
- 2 路 `320x240` 对 ACT 更适合作为默认闭环输入。
- 2 路 `640x480` 也能跑，但 latency 进入 `3` steps 量级，适合作为识别/debug 或更慢技能，不建议默认用于精细闭环。
- 当前 ACT 只是随机权重，下一步若要用于比赛，需要针对具体技能采集示教数据并训练/微调 checkpoint。

比赛建议更新：

- 主流程优先押 ACT/示教轨迹/状态机，而不是 SmolVLA。
- SmolVLA 作为 VLA + DRTC 技术亮点，完成低频关键技能或自然语言条件触发。
- ACT 作为关键轨迹技能候选：抓取、移动、倒料、搅拌、装盘。
- 真实比赛前要至少训练一个小 ACT checkpoint，否则当前 random ACT 只能说明延迟可行，不能说明任务可行。

现成 SO101 ACT checkpoint 测试：

候选：

- Hugging Face repo：`jliu6718/lerobot-so101-act`
- 本地路径：`checkpoints/jliu6718_lerobot-so101-act`
- 4070 路径：`/home/fbsh/drtc/checkpoints/jliu6718_lerobot-so101-act`
- 大小：约 `197M`
- input features：
  - `observation.state`: shape `[6]`
  - `observation.images.front`: shape `[3, 480, 640]`
- output features：
  - `action`: shape `[6]`
- chunk_size: `100`
- n_action_steps: `100`
- 注意：config 默认 `pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`。远端无外网时需要 override 为 `None`，否则 torchvision 会尝试下载 ResNet18 权重。

工具更新：

- `tools/drtc_policy_benchmark.py` 新增 `--no-act-pretrained-backbone-weights`。
- `RobotClientDrtcConfig` / `RemotePolicyConfig` / `PolicyServerDrtc` 新增 `policy_no_act_pretrained_backbone_weights` / `no_act_pretrained_backbone_weights`。
- `tools/drtc_mock_smoke.py` 新增 `--policy-no-act-pretrained-backbone-weights`。
- `tools/drtc_mock_smoke.py` 新增 `--camera-names`，用于把 mock camera 命名为 checkpoint 需要的 `front`。

SO101 ACT 单独 policy benchmark：

- 命令：`PYTHONPATH=src /home/fbsh/real-time-chunking-kinetix/.conda/envs/drtc-smoke/bin/python tools/drtc_policy_benchmark.py --mode policy --device cuda --policy-type act --pretrained-name-or-path /home/fbsh/drtc/checkpoints/jliu6718_lerobot-so101-act --no-act-pretrained-backbone-weights --warmup 10 --iters 50 --actions-per-chunk 50 --state-dim 6 --task "act benchmark" --json`
- 结果：通过
- mean: `12.26ms`
- median: `12.25ms`
- p95: `12.42ms`
- p99: `12.50ms`
- max: `12.52ms`
- load_ms: `991.78ms`
- processor_ms: `7.09ms`

SO101 ACT 真实 DRTC server/client smoke：

- server：`192.168.0.110:18081`
- server mode：`real`
- client：本地 Mac mock robot
- policy：`act`
- checkpoint：`/home/fbsh/drtc/checkpoints/jliu6718_lerobot-so101-act`
- `--policy-no-act-pretrained-backbone-weights`
- `actions_per_chunk=50`
- `fps=20`
- observation payload：1 路 `front` camera，`640x480` 随机 RGB 图像，JPEG quality 60
- 输出文件：`artifacts/drtc_real_act_so101_front_640x480_60s.json`
- 结果：通过
- received_chunks: 158
- executed_actions: 1180
- final_action_step: 1179
- final_schedule_size: 46
- latency_steps: 2
- latency_ms: 94.21
- diagnostic RTT avg 最终约 `77ms`
- diagnostic RTT max 约 `489.84ms`，主要来自早期冷启动/估计收敛。

ACT checkpoint 结论：

- 这个现成 ACT checkpoint 的 4070 推理延迟非常低，单独 `predict_action_chunk` 约 `12ms`。
- 完整 DRTC 环路用单路 `640x480` 时最终约 `77-94ms`，明显优于 SmolVLA 的 `216-228ms`。
- 该 checkpoint 的任务不一定与比赛烹饪相关，因此不能直接代表比赛任务成功率。
- 但它证明了 ACT 路线在当前 DRTC + 4070 架构里有足够好的实时预算。
- 比赛准备应优先把 ACT 用在可示教、可重复的烹饪技能上，而不是继续把主流程押给 SmolVLA。

### 2026-04-24 现场算力约束更新

新增约束：

- `192.168.0.110` RTX 4070 无法带到比赛现场。
- AMD Ryzen AI 9 HX PRO 370 机器可以带到比赛现场。

影响：

- RTX 4070 从“现场主力推理端候选”降级为“赛前开发/训练/benchmark 对照机器”。
- AMD 从“P2 备机探索”提升为“P0/P1 现场 policy server 候选”。
- 当前已经在 4070 上证明 ACT/DRTC 路线性能可行，但这只是上限参考；现场方案必须在 AMD 或 J4012 上复测。
- SmolVLA 在 4070 上已经偏慢，短期不应优先迁移到 AMD；AMD 第一优先级是 ACT 和轻量视觉策略。

新的现场候选架构：

```text
J4012 robot client
  - 相机采集
  - reBot 控制
  - action schedule / cooldown / LWW / safety
  - 状态机和本地 fallback

AMD Ryzen AI 9 HX PRO 370 policy server
  - ACT / 轻量 policy chunk 推理
  - 可选低频视觉识别或策略辅助

RTX 4070
  - 赛前训练
  - 离线 benchmark
  - 作为性能上限和回归对照
```

AMD 最小验证计划：

1. `torch`/ROCm smoke：
   - `torch.__version__`
   - ROCm/HIP version
   - device name
   - `torch.cuda.is_available()`
   - CUDA-equivalent matmul smoke
2. ACT benchmark：
   - `random-act`
   - `jliu6718/lerobot-so101-act`
   - 先测单独 `predict_action_chunk`
3. DRTC smoke：
   - J4012-like client -> AMD server
   - 2 路 `320x240`，20Hz，60 秒
   - 若通过，再测 1 路 `front 640x480`
4. 判定：
   - 如果 ACT chunk 能稳定在 `50-150ms` 且 DRTC schedule 不清空，AMD 可作为现场 policy server。
   - 如果 AMD ROCm 不稳定或 latency 过高，现场主流程转为 J4012 本地轻量检测 + 状态机/示教轨迹，AMD 只做展示/探索。

AMD 验证结果：

- 机器：`AMD Ryzen AI 9 HX PRO 370 w/ Radeon 890M`
- ROCm：`7.2.1` 系统栈，`gfx1150`
- PyTorch：`2.13.0.dev20260423+rocm7.1`
- HIP：`7.1.52802`
- `torch.cuda.is_available()`: `True`
- device: `AMD Radeon 890M`
- FP16 `2048x2048` matmul smoke: `42.74ms`

AMD policy benchmark：

- synthetic baseline, hidden 512, chunk 50: mean `0.12ms`
- random ACT, 1 路 `640x480`, chunk 50: mean `34.67ms`, p99 `34.95ms`
- `jliu6718/lerobot-so101-act`, chunk 50: mean `35.78ms`, p99 `36.59ms`, load `451ms`

AMD DRTC smoke：

- 拓扑：本地 Mac 模拟 J4012 client -> AMD policy server
- server: `192.168.0.128:18082`
- policy: `jliu6718/lerobot-so101-act`
- 输入：1 路 `front 640x480`
- 频率：`20Hz`
- 时长：`60s`
- received_chunks: `147`
- executed_actions: `1157`
- final_schedule_size: `47`
- latency_steps: `4`
- latency_ms: `173.11`
- artifact: `artifacts/drtc_amd_real_act_so101_front_640x480_60s.json`

AMD 2 路相机补测：

- 尝试配置：2 路 `320x240`，camera names `front,wrist`
- 结果：失败，未收到 action chunk，client schedule 一直为空。
- server 错误：`KeyError: observation.images.wrist`
- 判断：当前 `jliu6718/lerobot-so101-act` checkpoint 只声明 `observation.images.front`，真实 policy server 路径没有忽略额外相机输入。
- 影响：不能用这个 checkpoint 直接验证 2 路相机链路。若要验证 2 路传输，需要：
  - 使用 `random-act` 2-camera server。
  - 或训练/构造声明 `front,wrist` 两路输入的 ACT checkpoint。
  - 或在 DRTC server/preprocessor 层增加“只转发 policy config 需要的 observation key”的过滤。

AMD 2 路传输/调度隔离验证：

- server mode: `random-act`
- 输入：2 路 `front,wrist 320x240`
- 频率：`20Hz`
- 时长：`60s`
- received_chunks: `120`
- executed_actions: `1146`
- final_schedule_size: `47`
- latency_steps: `4`
- latency_ms: `154.45`
- artifact: `artifacts/drtc_amd_random_act_2x320x240_60s.json`
- 结论：AMD + 网络 + DRTC 调度在双相机小图下可跑通；当前双相机阻塞点不是链路，而是实际 checkpoint 的输入 schema。

AMD 启动脚本：

- AMD policy server: `scripts/start_amd_drtc_policy_server.sh`
- 本地/J4012-like client smoke: `scripts/run_amd_drtc_client_smoke.sh`
- client 脚本会清理 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，并把 policy server IP 注入 `NO_PROXY`，避免 gRPC 被本机代理劫持。
- 默认 server 地址为 `192.168.0.128:18082`，现场需要按实际 AMD IP 覆盖 `POLICY_SERVER_ADDRESS`。
- 脚本化 15 秒 smoke 已通过：
  - server: `PORT=18085 ./scripts/start_amd_drtc_policy_server.sh`
  - client: `POLICY_SERVER_ADDRESS=192.168.0.128:18085 DURATION_S=15 ./scripts/run_amd_drtc_client_smoke.sh`
  - received_chunks: `29`
  - executed_actions: `280`
  - final_schedule_size: `46`
  - latency_steps: `4`
  - latency_ms: `173.12`
  - artifact: `artifacts/drtc_amd_scripted_act_so101_front_640x480_15s.json`

AMD 10 分钟长测：

- 拓扑：本地 Mac 模拟 J4012 client -> AMD policy server
- 命令路径：`scripts/start_amd_drtc_policy_server.sh` + `scripts/run_amd_drtc_client_smoke.sh`
- server: `192.168.0.128:18086`
- policy: `jliu6718/lerobot-so101-act`
- 输入：1 路 `front 640x480`
- 频率：`20Hz`
- 时长：`600s`
- received_chunks: `1549`
- executed_actions: `11709`
- final_schedule_size: `44`
- latency_steps: `4`
- latency_ms: `186.89`
- artifact: `artifacts/drtc_amd_scripted_act_so101_front_640x480_10min.json`

10 分钟长测观察：

- 没有出现 schedule starvation，`schedule_size` 未归零。
- 早期出现一次约 `2.3s` 的 diagnostic RTT max 尖峰，之后也有 `1.0-1.5s` 级别的偶发尖峰。
- DRTC schedule 能吸收这些尖峰，`latency_steps` 在尖峰后会短时升到约 `10-18`，随后恢复到 `3-5`。
- 中后段窗口更稳定，diagnostic max 多数回落到几百 ms，最终 `latency_steps=4`。
- 当前结论：AMD 作为现场 ACT policy server 可继续推进；但比赛现场必须使用有线网络、固定 IP、关闭代理，并准备本地 fallback，避免 Wi-Fi/交换机抖动放大尖峰。

AMD 结论：

- AMD 已通过 P0 最小验证，可以作为现场 policy server 的主候选继续推进。
- 单次 ACT 推理约 `36ms`，说明主要风险不在模型前向，而在完整链路的序列化、网络、调度和偶发尖峰。
- 60 秒闭环没有 schedule starvation，最终 queue 仍有 `47` 个 action，基本满足 DRTC 可用性判断。
- 运行中出现过 `total_latency_rtt_ms max ~= 2.3s` 的尖峰。
- 10 分钟长测已确认尖峰不会清空 schedule，但仍会短时提高 latency/cooldown；真实机器人动作层需要限速、急停和本地状态机保护。
- 本地 gRPC client 会受代理环境变量影响；现场运行脚本必须显式设置 `NO_PROXY=<policy_server_ip>,localhost,127.0.0.1`，或清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`。
- `amd` 是 SSH alias，不是可解析主机名；机器人端应使用局域网 IP 或现场 DNS/hosts 绑定。
- PyTorch ROCm nightly wheel 下载体积约 `5.8GB`，且安装中发生过一次下载超时；现场前必须离线缓存 wheel、checkpoint 和 venv/conda 环境。

当前架构判断更新：

- 现场主线候选改为 AMD policy server + J4012 robot client。
- RTX 4070 继续作为赛前开发、训练和性能上限参考，不作为现场执行依赖。
- 默认闭环输入继续建议 2 路 `320x240`，因为网络端到端更稳。
- 2 路 `640x480` 的 policy 前向不是问题，但链路 latency 已经进入 `latency_steps=2`，只建议作为有线网络确认后的备选或高分辨率识别路径。
- AMD 上 1 路 `640x480` 已可跑通，但最终 latency 约 `173ms`；现场主流程仍建议先按 1 路 `640x480` 或 2 路 `320x240` 取舍，不要一开始押 2 路 `640x480`。
- 现有 SO101 ACT checkpoint 只能作为 1 路 `front` 闭环验证基准；比赛任务如果需要多视角，必须训练多相机 checkpoint 或在策略前做显式相机选择。

下一步：

- 决定 ACT 训练输入：优先 1 路 `front 640x480` 快速收敛；若比赛场景遮挡明显，再升级到 `front,wrist` 双相机 checkpoint。
- 离线缓存 PyTorch ROCm wheel、torchvision wheel、checkpoint、pip wheelhouse，避免现场网络依赖。
- 在 AMD 上做故障注入：latency spike、observation drop、action drop，验证 cooldown 和 fallback。
- 采集/准备一个比赛相关的 ACT 示教数据集，优先选一个稳定技能训练 checkpoint。
- 等拿到 J4012 后，把本地 Mac client 替换为真实 J4012 client 复测同一套指标。
- 给 `tools/drtc_mock_smoke.py` 继续增加更细的 CSV metrics 输出，后续能长期跑 10 分钟并记录 p50/p90/p99。
