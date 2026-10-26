# 贡献指南

[**English**](CONTRIBUTING.md) | **简体中文**

[← 返回 README](README.zh-CN.md) · [路线图](ROADMAP.zh-CN.md)

感谢关注。本仓库主要是一个个人研究归档，同时也作为活跃项目维护，因此最有用的
贡献是那些提升**可复现性**的贡献，而不是功能数量。

## 目前最欢迎的贡献

1. **可移植性。** Stage 1 的代码是在 Windows 上写的，带有硬编码的局域网 IP 和
   Windows 风格路径。关于 Linux/WSL 运行情况的报告（以及相应修复）、AirSim
   版本锁定、依赖冲突，都很有价值。
2. **基准测试框架。** 任何让评估套件跑起来更便宜、或更难配错的东西 —— 场景注册表、
   固定种子集、用于 CI 的更小的冒烟测试子集。
3. **失败模式的记录。** 如果你复现了 `failure_analysis.txt` 未归类的碰撞模式，
   那就是一个真正新的数据点。
4. **Isaac Lab 方面的指引。** Stage 2 正在进行中；关于 PhysX raycast LiDAR 吞吐
   或 PyTorch 中块稀疏地图 kernel 的具体笔记能省下大量时间。

不太有用的：对已归档 Stage 1 脚本的纯风格重构，或为 Isaac Lab 之外的仿真器添加
新功能。

## 基本规则

- **绝不提交二进制产物。** `git ls-files` 必须保持为文本。模型权重、归档包和视频
  属于 GitHub Release。推送前运行守卫：

  ```bash
  python scripts/repo_hygiene.py --all
  ```

- **保持归档分支完整。** `origin_26-0364` 是与已发表论文对应的冻结快照。不要改写
  它的历史，也不要"修正"其中的数字 —— 如果需要更正什么，在 `main` 上做，并记录
  差异。
- **数字必须可追溯。** 如果某个改动影响已报告的指标，请说明它改变了哪张表或哪幅
  图，以及为什么。`benchmarks/` 是已发表数值的唯一真源。
- **每个 pull request 只做一处逻辑改动**，提交信息要解释*原因*，而不只是 diff。
  欢迎在同一个 PR 中提交 `CHANGELOG.md` 条目。

## 本地检查

```bash
python scripts/repo_hygiene.py --all     # no oversized / binary files
python scripts/repo_hygiene.py --all --docs   # every doc has a Chinese twin and no dead links
python -m compileall -q RL_scripts methods scripts   # syntax
ruff check .                             # lint (once configured)
```

**修改文档时。** 仓库是双语的：`X.md` 与 `X.zh-CN.md` 成对存在，标题下方都有语言切换行；
中文页面里的文档互链指向对应的中文版。`python scripts/repo_hygiene.py --all --docs` 会一次性
校验这三条约定，改动文字内容前请先跑一遍。

## 报告问题

请说明：分支、你的操作系统 + Python 版本、确切的命令、完整的 traceback，并且 ——
如果是仿真问题 —— AirSim 的版本和修改过的 `settings.json`。Stage 1 的大多数故障
是配置问题，不是代码问题。

## 联系方式

请开 Issue。与已发表文章相关的任何事项，请使用
[`paper/README.zh-CN.md`](paper/README.zh-CN.md) 中的 DOI。
