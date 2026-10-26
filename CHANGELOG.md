# Changelog

**English** | [简体中文](CHANGELOG.zh-CN.md)

[← README](README.md) · [Development plan](ROADMAP.md)

Changes are recorded by public scope and behavior.

## Unreleased — 2026-10-26

### Documentation and research plan

- Revise all English/Chinese Markdown pairs with consistent terminology and explicit evidence limits.
- Make Isaac Lab migration the immediate contribution priority, including PX4 SITL / ROS 2 links, training-method iteration and sensor redesign.
- Schedule reward/small end-to-end network research after the migration foundation, with DRL, VLA and composed-agent exploration as research directions.
- Document later Sim2Real work and long-term alternative vehicle morphologies without near-term commitments.
- Separate journal-policy work from the map-maintenance and shield extensions.
- Correct descriptions of probability/odds updates, frames, query distance, curriculum coefficients, learning schedules and heuristic shield limits.
- Record existing reproduction gaps: historical module imports, rolling evaluation statistics, failure-rate naming and unresolved current-code/historical-run configuration differences.
- Reconcile simulator-platform, plugin-pawn placement, file-size units and licence descriptions.

These documentation changes do not modify policy, reward, mapping or shield algorithms and do not recalculate historical conclusion values.

## 0.1.0 — Historical AirSim baseline

The historical release collected the AirSim/Stable-Baselines3 implementation, BSHM mapping, dual-branch PPO, reward-shaping curriculum, additional shield evaluation code, model/simulator catalogues and conclusion figures.

The delivered trainer currently specifies ten workers and 2560 rollout steps per worker. This code configuration has not been fully reconciled with original reporting metadata and is not presented as a verified exact historical setup.

The legacy reference origin_26-0364 denotes this AirSim line of work.

## Existing limitations

Static obstacles, ground-truth simulator state and a simulator velocity interface define the current scope. There is no demonstrated real-flight transfer or formal end-to-end safety guarantee. Isaac Lab/PX4/ROS 2 migration and broader policy methods remain planned work described in the [roadmap](ROADMAP.md).
