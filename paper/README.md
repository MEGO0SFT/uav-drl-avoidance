# Publication Metadata and Scope

**English** | [简体中文](README.zh-CN.md)

[← README](../README.md) · [Model catalogue](../MODELS.md) · [Citation metadata](../CITATION.cff)

## Citation

石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕. **基于渐进课程学习的无人机自主避障研究**. 系统仿真学报 (Journal of System Simulation), 2026.

[DOI: 10.16182/j.issn1004731x.joss.26-0364](https://doi.org/10.16182/j.issn1004731x.joss.26-0364). The manuscript identifier joss.26-0364 is the source of the historical reference name origin_26-0364.

    @article{shi2026uavcl,
      title   = {基于渐进课程学习的无人机自主避障研究},
      author  = {石辅天 and 李嘉轩 and 汪孝迪 and 邓中行 and 史格非 and 杨雪榕},
      journal = {系统仿真学报 (Journal of System Simulation)},
      year    = {2026},
      doi     = {10.16182/j.issn1004731x.joss.26-0364},
      url     = {https://doi.org/10.16182/j.issn1004731x.joss.26-0364}
    }

## Article versus repository extensions

The article covers the simulation platform, BSHM representation, observation/action and reward design, and curriculum-trained policy. Dual-layer map maintenance and the hybrid shield with its ablations are additional repository work, not part of the article's stated scope.

| Topic | Related public material |
|---|---|
| Simulation / data hub | [airsim_datamanager.py](../RL_scripts/body-axis_v2/scripts/airsim_datamanager.py) |
| Current sensor configuration | [settings.json](../sim/airsim/settings.json); methods/Sensors1.json is historical |
| Observation and action semantics | [Architecture](../docs/ARCHITECTURE.md) and vehicle_manager.py |
| Network / reward / curriculum | PPO_SB3_AirSim.py and PPO_SB3_AirSim_CL.py in the main scripts directory |
| Historical training conclusions | [Run metadata](../RL_scripts/body-axis_v2/logs/README.md) and [final figures](../benchmarks/figures/) |
| Additional shield experiments | [Aggregate JSON](../benchmarks/ablation_summary.json) and [benchmark notes](../benchmarks/README.md) |
| New-run failure diagnostics | [collision_diagnostics.py](../RL_scripts/body-axis_v2/scripts/collision_diagnostics.py) |

These links locate related implementations and retained summaries, not a verified executable snapshot of every original experiment.

## Reproduction boundary

Final figures and conclusion values remain public. The public checkout is not a complete reproduction package for all original curves and episode-level statistics.

The source configuration and historical reporting cadence have not been fully reconciled. New experiments require matching simulator assets, checkpoints, dependencies and evaluation protocols; see [the experiment guide](../docs/REPRODUCING.md). Historical reference names are labels rather than immutable commit identifiers.

## Copyright

The publisher's article PDF is not redistributed here; use the DOI for the publication. Repository code, scene assets, model packages and upstream components have distinct attribution and licence considerations. Git commit identity changes do not alter the article author list or existing copyright notices.
