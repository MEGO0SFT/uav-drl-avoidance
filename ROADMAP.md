# Long-term Development Plan

**English** | [简体中文](ROADMAP.zh-CN.md)

[← README](README.md) · [Contributing](CONTRIBUTING.md)

This plan distinguishes the existing AirSim baseline from proposed research. Priorities describe intended sequencing, not committed release dates or guaranteed outcomes.

## Existing baseline

The repository contains AirSim/SB3 source code, BSHM mapping, curriculum PPO and an additional safety-shield evaluation implementation. It retains historical conclusion tables and final figures. The repository alone cannot reconstruct the original training runs. See [benchmarks](benchmarks/README.md) and [paper scope](paper/README.md).

The legacy reference origin_26-0364 identifies the AirSim line of work; it is a reference name rather than an immutable byte-level archive.

## 1. First priority: AirSim to Isaac Lab

The immediate contribution priority is a usable Isaac Lab training environment. This includes more than simulator API compatibility:

- Build and validate vehicle dynamics, scene generation, reset/termination semantics and parallel sampling.
- Establish PX4 SITL and ROS 2 communication paths as needed by the selected control architecture. Document message definitions, coordinate frames, timestamps and interface ownership.
- Iterate on training methods, sampling throughput, action scaling and curriculum design.
- Redesign the sensor configuration and processing pipeline: field of view, scan rate, noise, latency, data budget and geometric representation.
- Reconcile policy observations, action frames and decision frequency with the controller interface.
- Provide documented environment versions, measured resource use and a repeatable evaluation protocol.

LiDAR implementation, communication transport and training library choices remain engineering decisions to benchmark. Environment counts and expected speedups should be reported from measurements rather than assumed in advance.

A useful completion criterion is a documented training/evaluation path with consistent frames, timing and metrics, plus a supported interface for the intended flight-controller setup. A claim of improved policy quality requires a comparable scenario and evaluation protocol.

## 2. After priority 1: rewards, architectures and broader agents

The current policy architecture has limited performance. The author is interested in how network structure affects small end-to-end policies, and in further exploring DRL rather than treating the present PPO configuration as a final design.

This phase starts after the first priority is sufficiently established. Candidate work includes:

- Reward-function analysis and controlled ablations separating task progress, obstacle response and motion regularity.
- Network iteration under explicit parameter, compute, memory and latency budgets.
- Alternatives to the current distance-image encoder and methods for temporal context.
- Exploration of vision-language-action (VLA) and composed-agent methods for a broader task range and larger capability boundary.

These are research directions. The repository does not currently implement a VLA system or a general-purpose agent. Broader methods should be compared with a documented DRL baseline rather than presumed superior.

## 3. Sim2Real

Real-world operation introduces factors absent or simplified in Stage 1. Future work will investigate:

- Matching policy/control update rates to actuator bandwidth and saturation.
- Sensing, estimation, inference and transport delays, including timestamp consistency.
- Disturbance rejection, model error, sensor noise and generalisation across environments.
- Domain randomisation, system identification and staged validation from simulation to hardware.

PX4 SITL is a software integration step, not evidence of real-world transfer. Each stage should report its assumptions, validation conditions and observed limitations. No real-flight performance is claimed by the current release.

## 4. Beyond conventional UAV structures

Other vehicle morphologies and control structures are a long-term possibility. This work is outside the near-term implementation scope. No specific platform or schedule is committed.

## Cross-cutting maintenance

Maintain bilingual documentation, precise experiment configuration, separate success/collision/timeout counters and a small evaluation harness. Address historical import names, rolling-window evaluation output and asynchronous observation/action timing before claiming exact reproducibility.

New training and evaluation artefacts should remain local. Public summaries and final figures must state their protocol and provenance.
