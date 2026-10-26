# Contributing

**English** | [简体中文](CONTRIBUTING.zh-CN.md)

[← README](README.md) · [Development plan](ROADMAP.md)

Contributions should help turn the AirSim research baseline into a better documented, more capable experimental platform. Please discuss substantial interface or research changes in an issue before investing in a large implementation.

## Most welcome contributions

1. **Isaac Lab migration comes first.** Port the AirSim training environment while also building the required PX4 SITL / ROS 2 communication paths, iterating on training methods and redesigning sensors and their processing pipeline. A simulator API replacement alone does not complete the migration.
2. **Reward optimisation and network architecture iteration, after item 1 is established.** The present architecture has limited performance. The author is investigating architecture effects on small end-to-end policies and intends to further explore DRL, then VLA and composed-agent methods, toward a broader application range and capability boundary. Contributions should make assumptions, compute budgets and evaluation protocols explicit.
3. **Sim2Real.** Investigate policy/control frequency versus actuator capability, latency, disturbance rejection and generalisation. Separate software-in-the-loop integration from evidence obtained on physical hardware.
4. **Structures beyond conventional UAVs.** This is a long-term direction and is not planned for near-term implementation.

The [roadmap](ROADMAP.md) explains sequencing. These directions are proposals, not existing features or commitments to particular performance gains.

## Scope and evidence

- Use main for maintenance and future development. Historical experiment directories are references; do not assume they are maintained runnable packages.
- Preserve historical conclusion values unless a correction is explicitly justified. Separate historical reports, current-code measurements and proposed outcomes.
- When a metric changes, state its protocol, seeds, sample count, evaluation mode and handling of collision versus timeout. Cumulative ablations do not isolate individual causal effects.
- Record configuration and provenance for new summaries. Training and evaluation artefacts are local and must not be committed.
- Keep final figures, approved summary tables and essential scene/configuration assets. Model weights and simulator archives belong in separately managed release artefacts.

## Repository checks

Run from the repository root:

    python scripts/repo_hygiene.py --all
    python scripts/repo_hygiene.py --all --docs
    python -m compileall -q RL_scripts methods scripts

The guard checks forbidden artefact types, local run-artefact paths, size limits and bilingual document links. It does not establish runtime correctness, training convergence or research reproducibility. Static parsing does not resolve imports or exercise the simulator. A lint command is appropriate only after its configuration and tool version are defined.

## Documentation and pull requests

Update English and Chinese documents together. Each Markdown document has a language-switch link to its counterpart; a Chinese page should use Chinese counterparts for document links where available.

Keep each pull request focused and explain the reason for the change. Distinguish tested behavior from design intent, and mention external simulator/checkpoint requirements. Include relevant checks and, when applicable, a [changelog](CHANGELOG.md) entry.

## Reporting a problem

Include the branch or commit, operating system, Python/dependency versions, command, traceback and relevant configuration. Simulation reports should specify AirSim/Unreal versions, settings, network topology and drone counts. Remove secrets and private material before attaching files.

For article metadata and citation, see [paper/README.md](paper/README.md).
