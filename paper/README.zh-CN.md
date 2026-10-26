# 论文元数据与范围

[**English**](README.md) | **简体中文**

[← README](../README.zh-CN.md) · [模型清单](../MODELS.zh-CN.md) · [引用元数据](../CITATION.cff)

## 引用

石辅天, 李嘉轩, 汪孝迪, 邓中行, 史格非, 杨雪榕。**基于渐进课程学习的无人机自主避障研究**。系统仿真学报（Journal of System Simulation），2026。

[DOI: 10.16182/j.issn1004731x.joss.26-0364](https://doi.org/10.16182/j.issn1004731x.joss.26-0364)。稿件编号 joss.26-0364 是历史参照名称 origin_26-0364 的来源。

    @article{shi2026uavcl,
      title   = {基于渐进课程学习的无人机自主避障研究},
      author  = {石辅天 and 李嘉轩 and 汪孝迪 and 邓中行 and 史格非 and 杨雪榕},
      journal = {系统仿真学报 (Journal of System Simulation)},
      year    = {2026},
      doi     = {10.16182/j.issn1004731x.joss.26-0364},
      url     = {https://doi.org/10.16182/j.issn1004731x.joss.26-0364}
    }

## 论文与仓库扩展

论文覆盖仿真平台、BSHM 表示、观测／动作及奖励设计，以及课程训练策略。地图双层时间维护、混合安全盾及消融属于额外仓库工作，不属于期刊论文所述范围。

| 主题 | 相关公开材料 |
|---|---|
| 仿真／数据中心 | [airsim_datamanager.py](../RL_scripts/body-axis_v2/scripts/airsim_datamanager.py) |
| 当前传感器配置 | [settings.json](../sim/airsim/settings.json)；methods/Sensors1.json 属于历史模板 |
| 观测与动作语义 | [系统架构](../docs/ARCHITECTURE.zh-CN.md)及 vehicle_manager.py |
| 网络／奖励／课程 | 主脚本目录中的 PPO_SB3_AirSim.py 与 PPO_SB3_AirSim_CL.py |
| 历史训练结论 | [运行元数据](../RL_scripts/body-axis_v2/logs/README.zh-CN.md)和[最终图表](../benchmarks/figures/) |
| 额外安全盾实验 | [汇总 JSON](../benchmarks/ablation_summary.json)及[基准说明](../benchmarks/README.zh-CN.md) |
| 新运行失败诊断 | [collision_diagnostics.py](../RL_scripts/body-axis_v2/scripts/collision_diagnostics.py) |

这些链接用于定位相关实现和保留摘要，不表示每项原始实验都存在已经验证的可执行快照。

## 复现边界

最终图表和结论数值继续公开。公开检出不是全部原始曲线和逐回合统计的完整复现包。

当前源码参数与历史报告步数节奏尚未完整对齐。新实验需要匹配的仿真器、权重、依赖和评估协议，见[实验指南](../docs/REPRODUCING.zh-CN.md)。历史参照名称是标签，不保证对应固定提交 ID。

## 版权

本仓库不转发出版方论文 PDF，请通过 DOI 访问。源码、场景资产、模型包和上游组件分别具有相应署名及许可要求。Git 提交身份的调整不改变论文作者列表或既有版权说明。
