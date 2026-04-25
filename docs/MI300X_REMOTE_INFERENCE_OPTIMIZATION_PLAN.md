# MI300X Remote Inference Optimization Plan

This plan records the current DRTC split-deployment baseline and the concrete work needed to make
local-client to MI300X-server inference viable for real-time robot control.

## Current Baseline

Environment:

- Local client repo: `/Users/fbsh/projects/drtc-Phi`
- MI300X server repo: `/mnt/models_alehe/phi-fbsh/drtc-Phi`
- MI300X server env: `/mnt/models_alehe/phi-fbsh/.venvs/drtc-mi300x`
- MI300X public IP: `165.245.129.215`
- Direct DRTC test port: `18201`
- Test checkpoint: `/mnt/models_alehe/phi-fbsh/drtc-Phi/tmp/vlash-pi05-libero-async5-drtc-migrated`

Observed split-deployment result:

- Local client connected directly to `165.245.129.215:18201`.
- Server loaded the PI05 checkpoint and returned `50 x 7` action chunks.
- `Ready` RPC: about `0.94s`
- `SendPolicyInstructions` / model load: about `40s`
- Observation payload: about `302KB`
- First chunk: observation send about `1.23s`, action wait about `12.33s`
- Second chunk: observation send about `0.77s`, action wait about `0.14s`
- SSH tunnel comparison: observation send was about `6s`, so SSH tunnel is not acceptable for realtime use.

Important limitation:

- The migrated `vlash-pi05-libero-async5` checkpoint has loader/key compatibility warnings in this DRTC fork.
- These tests validate transport and serving mechanics, not final action quality.

## External Reference

Modal's Physical Intelligence remote inference writeup is the key reference:

https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal

Useful takeaways:

- Avoid SSH tunnels for realtime robot control.
- Keep a persistent bidirectional channel between robot runtime and GPU server.
- Minimize head-of-line blocking and jitter; QUIC/UDP-style transport is worth evaluating if TCP/gRPC remains unstable.
- Place GPU inference close to robots; the referenced system targets roughly `10-15ms` added network overhead.
- Keep checkpoints next to GPU compute and load/warm them before control begins.

## Goal

Near-term goal:

- Reduce local-client to MI300X observation send time from `0.77-1.23s` to less than `100ms`.

Target goal for practical robot control:

- Observation send time: less than `30-50ms`.
- Warm action chunk return after observation: less than `200ms` for current PI05 baseline, then improve as model/runtime permits.
- Server model load and warmup are allowed startup costs, never control-loop costs.

## Single Cloud Instance Constraint

The practical constraint is that only this MI300X cloud instance is available. If the robot/client must
connect from the current local network to `165.245.129.215`, the measured `~250ms` direct-IP floor is a
hard constraint. Code-level payload optimization cannot by itself turn this path into a `30-50ms`
closed-loop link.

Revised strategy:

- Treat MI300X as a high-latency remote policy server.
- Keep the model loaded and warm on the server.
- Minimize client-to-server observation frequency.
- Use longer action chunks so the local client can keep executing while the next chunk is in flight.
- Tune DRTC parameters for schedule stability and smoothness, not for low-latency visual servoing.
- Prefer slower manipulation tasks, not high-speed dynamic tasks.

Tasks that remain plausible under this constraint:

- Slow pick/place.
- Periodic visual correction with local action continuation.
- High-level remote policy plus local low-level control.
- Offline or semi-online evaluation of remote policy serving mechanics.

Tasks that are not plausible without a better network path:

- Ping-pong.
- Fast dynamic catching/hitting.
- High-frequency visual servoing.
- Tasks requiring reaction times below the measured network RTT floor.

## Phase 0: Keep Server Long-Lived

Status: partially done.

Tasks:

- Run DRTC server as a long-lived process on MI300X.
- Bind to `0.0.0.0` only on explicitly allowed test ports.
- Load checkpoint once.
- Warm the model after load.
- Do not call `SendPolicyInstructions` during each experiment tick.

Commands used for direct test:

```bash
ufw allow 18201/tcp comment "DRTC temporary test port"
python examples/tutorial/async-inf/policy_server_drtc.py \
  --host 0.0.0.0 \
  --port 18201 \
  --fps 30 \
  --obs-queue-timeout 0.2
```

Acceptance criteria:

- Local TCP connect to `165.245.129.215:18201` succeeds.
- Client can send multiple observations without restarting the server.
- `policy_setup` happens once per session, not per action chunk.

Security note:

- `ufw allow 18201/tcp` exposes an unauthenticated gRPC endpoint. Use only for controlled testing.
- Prefer VPN or source-IP-restricted firewall rules for longer runs.

## Phase 1: Network Baseline

Before changing DRTC code, measure whether the path can support realtime control.

Tasks:

- Measure ICMP RTT if allowed: `ping 165.245.129.215`.
- Measure TCP connect latency to `18201`.
- Add a small gRPC health/echo RPC or use a minimal Python gRPC script to measure empty-message RTT.
- Measure raw upload throughput for representative sizes: `32KB`, `128KB`, `300KB`, `1MB`.
- Compare direct IP vs SSH tunnel vs VPN if available.

Acceptance criteria:

- Empty RTT should be stable and ideally below `50ms`.
- `300KB` raw upload should be well below `100ms`.
- If raw network upload is already near seconds, prioritize network path/VPN/region before payload code.

Initial direct-IP measurements from local client to `165.245.129.215:18201`:

| Payload | Images | JPEG | Payload bytes | Send time |
| --- | --- | --- | ---: | --- |
| Empty-ish observation | 0 | none | `378` | min `251.1ms`, median `264.9ms`, max `727.7ms` |
| Raw zero images | 2 x `224x224` | none | `301650` | min `310.2ms`, median `322.4ms`, max `1228.4ms` |
| JPEG zero images | 2 x `224x224` | quality `60` | `3303` | min `253.2ms`, median `258.9ms`, max `766.8ms` |
| JPEG random images | 2 x `224x224` | quality `60` | about `49KB` | min `326.7ms`, median `349.7ms`, max `957.1ms` |

Interpretation:

- The current direct public network path has a `~250ms` floor, visible even with a `378B` payload.
- Payload compression helps large/random images, but it cannot get below the network/RPC floor on this path.
- SSH tunnel remains worse (`~6s` send in the earlier test) and should only be used for debugging.
- To reach sub-`100ms`, the next priority is network placement or a lower-jitter private path, not only payload compression.

Initial sweep output:

- CSV: `results/drtc_transport_sweep_initial.csv`
- Script: `scripts/sweep_drtc_transport.py`

Sweep summary, median send time:

| Image mode | Image count | JPEG | Payload bytes | Median send |
| --- | ---: | --- | ---: | ---: |
| random | 0 | none | `378` | `271.2ms` |
| random | 1 | q40 | `18-19KB` | `267.3ms` |
| random | 1 | q60 | `24-25KB` | `271.1ms` |
| random | 1 | none | `151KB` | `316.0ms` |
| random | 2 | q40 | `36KB` | `259.0ms` |
| random | 2 | q60 | `49KB` | `270.3ms` |
| random | 2 | none | `302KB` | `374.9ms` |
| zeros | 0 | none | `378` | `271.3ms` |
| zeros | 1 | q40 | `1.9KB` | `255.5ms` |
| zeros | 2 | q60 | `3.3KB` | `256.7ms` |

Updated interpretation:

- For compressed payloads below about `50KB`, median send time is dominated by the `~250-270ms` network/RPC floor.
- Raw `151-302KB` payloads add noticeable overhead, especially for zero-image raw payloads and occasional spikes.
- JPEG q40/q60 is sufficient to remove most payload-size overhead for transport benchmarking, but it does not solve the RTT floor.
- Under the single-cloud-instance constraint, the next useful control work is DRTC high-latency scheduling, not deeper payload work first.

## Phase 2: Instrument Observation Transfer

The current `0.77-1.23s` send time needs decomposition.

Add diagnostics around:

- Raw image byte size.
- JPEG-encoded image byte size.
- Pickled `TimedObservation` byte size.
- Client-side encode time.
- Client-side pickle time.
- gRPC send time.
- Server receive time.
- Server unpickle time.
- Server JPEG decode time.
- Server preprocessing time.

Relevant files:

- `src/lerobot/async_inference/robot_client_drtc.py`
- `src/lerobot/async_inference/policy_server_drtc.py`
- `src/lerobot/async_inference/utils/compression.py`
- `src/lerobot/transport/utils.py`

Acceptance criteria:

- Every observation logs a compact timing breakdown.
- Client and server timestamps allow one-way and end-to-end attribution.
- Logs clearly distinguish network transfer from serialization and image processing.

## Phase 3: Reduce Payload Size

Start with no protocol rewrite.

Tasks:

- Lower JPEG quality from `60` to `40`, then `30`, and record payload/action behavior.
- Verify only policy-required cameras are sent.
- Test one-camera mode as an explicit diagnostic baseline.
- Avoid sending unused keys in `raw_observation`.
- Confirm images are sent at the model-required resolution, not camera-native resolution.

Acceptance criteria:

- Payload falls significantly below `300KB`.
- Observation send drops proportionally.
- Server still returns valid `ActionsDense` chunks.

## Phase 4: Replace Pickle Observation Transport

The current transport pickles a Python object containing dicts and NumPy arrays. This is convenient but not a good realtime wire format.

Proposed dense observation message:

```text
timestamp: float64
control_step: int64
chunk_start_step: int64
task: string
state_f32: bytes
state_dim: int32
images: repeated {
  key: string
  encoding: jpeg | webp | raw
  height: int32
  width: int32
  channels: int32
  data: bytes
}
rtc_meta_json: optional string
```

Tasks:

- Add a new protobuf message for dense observations.
- Add either a unary `SendObservationDense` RPC or a long-lived client-streaming RPC.
- Implement client encoder without pickle.
- Implement server decoder into the same internal observation dict expected by policy preprocessing.
- Keep old pickle path for compatibility during migration.

Acceptance criteria:

- Dense path returns numerically sane `ActionsDense`.
- Dense path send time is lower than pickle path at the same image quality.
- Dense path supports state, task, multiple images, and optional RTC metadata.

## Phase 5: Persistent Bidirectional Stream

The Modal/Pi reference suggests persistent bidirectional communication rather than request-style control-loop RPCs.

Tasks:

- Keep `StreamActionsDense` for action output.
- Add a persistent observation stream for input.
- Avoid reopening a gRPC stream per observation.
- Consider one bidirectional stream if it simplifies flow control.
- Add sequence numbers and keep LWW semantics by `control_step`.

Acceptance criteria:

- Client sends observations continuously over one stream.
- Server receives newest observation without per-frame stream setup.
- Action chunks continue streaming back over persistent server stream.
- No correctness regression under duplicated/reordered/dropped observations.

## Phase 6: Transport Alternatives

Only do this if direct TCP/gRPC remains too slow or too jittery after Phases 1-5.

Options:

- Tailscale/WireGuard: best next operational step for secure private connectivity.
- QUIC/UDP: best long-term transport if TCP head-of-line blocking is a real bottleneck.
- Regional deployment: move GPU closer to robot site if RTT dominates.

Acceptance criteria:

- VPN path has lower jitter than public TCP.
- QUIC prototype improves p95/p99 observation latency under packet loss or jitter.
- Chosen path is operationally simple enough for repeated robot experiments.

## Phase 7: Production Safety

Tasks:

- Do not expose unauthenticated DRTC ports publicly.
- Restrict firewall source IPs or run behind VPN.
- Add server-side request size limits.
- Add client/server version checks for wire protocol compatibility.
- Add structured logs for session ID, model ID, checkpoint path, client host, and server GPU.

Acceptance criteria:

- A test operator can identify which model, server, client, and transport produced each result.
- A stale or incompatible client fails fast.
- The server cannot be trivially abused as an open public endpoint.

## Immediate Next Steps

1. Implement Phase 1 network baseline scripts.
2. Add Phase 2 transfer diagnostics to current pickle path.
3. Run direct IP tests with JPEG quality `60`, `40`, `30`.
4. Decide whether the remaining bottleneck is network path, payload size, or pickle/gRPC overhead.
5. If payload/protocol remains the bottleneck, implement Phase 4 dense observation protobuf.

## High-Latency DRTC Schedule Simulation

Because the single-cloud-instance path has a measured `~250-270ms` floor, the next control question is
whether the local action schedule can stay non-empty while remote inference is in flight.

Scripts:

- `scripts/simulate_drtc_schedule.py`

Outputs:

- `results/drtc_schedule_simulation_initial.csv`
- `results/drtc_schedule_simulation_conservative.csv`

The simulator models local control ticks, cooldown-gated observation triggers, fixed observation-to-action
latency, and action chunk arrivals. It does not model action quality or RTC inpainting quality.

Selected results:

| FPS | H | s_min | latency | starvation | min schedule | mean schedule |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 100 | 40 | `1000ms` | `0.000` | `51` | `100.5` |
| 15 | 100 | 40 | `1000ms` | `0.000` | `46` | `95.5` |
| 20 | 100 | 40 | `1000ms` | `0.000` | `41` | `90.5` |
| 15 | 150 | 60 | `1500ms` | `0.000` | `69` | `143.5` |
| 20 | 150 | 60 | `1500ms` | `0.000` | `61` | `135.5` |
| 30 | 50 | 25 | `500ms` | `0.000` | `11` | `35.5` |
| 30 | 50 | 40 | `500ms` | `0.073` | `0` | `22.9` |

Interpretation:

- `H=50, s_min=40` triggers too late under high latency because `H - s_min` leaves only 10 scheduled actions before requesting a new chunk.
- `H=100, s_min=40` is a better schedule for this network path if the policy checkpoint supports a 100-step action horizon.
- `H=150, s_min=60` is more conservative and remains stable even at `1500ms` modeled latency, but it requires an even longer supported horizon and may reduce responsiveness.
- The currently migrated `vlash-pi05-libero-async5` checkpoint returned `50 x 7` action chunks and the server rejects `actions_per_chunk` values above the model-supported horizon.
- With the current `H=50` checkpoint, `fps=15`, `s_min=25`, `epsilon=2` remains stable in the simulator up to `1500ms` modeled observation-to-action latency; `s_min=40` becomes unstable at `1000ms+`.
- For the measured direct-IP floor and current checkpoint, practical first experiments should use `15 FPS`, `H=50`, `s_min=25`, `epsilon=2`.
- If observed end-to-end latency regularly exceeds `1s` and real action quality suffers, the correct fix is a checkpoint/model path with a longer horizon rather than only retuning `s_min`.

Recommended next runtime configuration:

- `fps=15`
- `actions_per_chunk=50`
- `s_min=25`
- `epsilon=2`
- JPEG quality `40`
- One or two policy-required cameras only
- Long-lived MI300X server with checkpoint loaded before control begins

Longer-horizon target configuration, contingent on a compatible checkpoint:

- `actions_per_chunk=100`
- `s_min=40`
- `epsilon=2`

Next implementation target:

- Add a client/server experiment mode that runs the above high-latency configuration for several minutes and records schedule size, starvation ticks, chunk arrival gaps, and measured RTT.

Implemented split-client runtime check:

- `scripts/run_high_latency_split_client.py`

Example command for the current MI300X server and current `H=50` PI05 checkpoint:

```bash
.venv/bin/python scripts/run_high_latency_split_client.py \
  --server-address 165.245.129.215:18201 \
  --pretrained-name-or-path /mnt/models_alehe/phi-fbsh/drtc-Phi/tmp/vlash-pi05-libero-async5-drtc-migrated \
  --fps 15 \
  --actions-per-chunk 50 \
  --s-min 25 \
  --epsilon 2 \
  --jpeg-quality 40 \
  --duration-s 30 \
  --output results/drtc_high_latency_split_client_mi300x.csv
```

Observed result with an already-loaded MI300X server, direct public endpoint `165.245.129.215:18201`,
and stale action filtering plus wall-clock-based `control_step` offsets:

- Command used `--skip-policy-setup`, `fps=15`, `actions_per_chunk=50`, `s_min=25`, `epsilon=2`, JPEG quality `40`, `duration_s=20`.
- TCP connect was about `256ms`.
- Prefill returned one `50 x 7` chunk with measured RTT about `1214ms`.
- During the 20s control loop: `301` ticks, `0` stalls, `14` chunks received, `14` observations sent.
- Schedule stayed non-empty: min schedule `12`, max schedule `49`.
- Measured action RTT: min `438ms`, median `514ms`, max `1815ms`.

This validates the scheduling side of the single-cloud split-client path for the current checkpoint.
It does not validate physical action quality, camera realism, or RTC inpainting quality.
