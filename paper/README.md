# Publication metadata

[← Back to README](../README.md) · [Model artefacts](../MODELS.md) · [Citation metadata](../CITATION.cff)

## Journal article (primary citation)

> 石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕.
> **基于渐进课程学习的无人机自主避障研究**.
> *系统仿真学报* (*Journal of System Simulation*), 2026.

| | |
|---|---|
| DOI | [`10.16182/j.issn1004731x.joss.26-0364`](https://doi.org/10.16182/j.issn1004731x.joss.26-0364) |
| 稿件编号 | joss.26-0364 → 本仓库归档分支名 `origin_26-0364` |

```bibtex
@article{shi2026uavcl,
  title   = {基于渐进课程学习的无人机自主避障研究},
  author  = {石辅天 and 李嘉轩 and 汪孝迪 and 邓中行 and 史格非 and 杨雪榕},
  journal = {系统仿真学报 (Journal of System Simulation)},
  year    = {2026},
  doi     = {10.16182/j.issn1004731x.joss.26-0364},
  url     = {https://doi.org/10.16182/j.issn1004731x.joss.26-0364}
}
```

## Scope note

The journal article covers the simulator, the BSHM map, the observation / reward design and the
curriculum-trained policy. Two further components of this repository are **not** part of the
article and are documented only here, in [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md):
the dual-layer temporal decay of the map, and the hybrid safety shield with its ablation.

## What each stage of the work maps to in this repository

| Stage | Code |
|---|---|
| §1.1 Distributed parallel simulator / data-exchange hub | `RL_scripts/body-axis_v2/scripts/airsim_datamanager*.py` |
| §1.2 Sensors, assumptions | `methods/Sensors1.json`, `methods/airsim_lidar.py` |
| §2.1–2.2 State / action space | `vehicle_manager.py` (`APF_SeqCalculate3D`, `_Lidar_RLstat_update`) |
| §2.3 Network (DBE) | `OctomapCNN` in `PPO_SB3_AirSim.py` |
| §2.4 Reward function | `Avoid_Reward_Optimized_BodyFrame`, `v_clip_reward_calc`, `_compute_reward` |
| §3.1 Progressive curriculum | `TrainProgresserCal` + `PPO_SB3_AirSim_CL.py` |
| §3.2 Hyper-parameters | `PPO(...)` call in `PPO_SB3_AirSim.py` |
| §4 results (control vs. curriculum) | `PPO_SB3_AirSim.py` vs `PPO_SB3_AirSim_CL.py`; curves in `benchmarks/tb_curves/` |
| BSHM temporal decay | `methods/AirSim_ReinforceLearning/BSHM.py` (`_decay_factor`, `_expiration_timeout`, LRU eviction) |
| Hybrid safety shield | `RL_scripts/body-axis_v2/scripts/PPO_SB3_AirSim_model.py` (`safe_ruler` and friends) |
| Safety-shield ablation table | `ablation_experiment.py` → `benchmarks/ablation_summary.json` |
| Failure attribution | `collision_diagnostics.py` → `failure_analysis.txt` |

## Copyright note

The repository does not redistribute the publisher's PDF of the journal article.
Please use the DOI link above. If you need a copy and cannot access the publisher,
open an issue.
