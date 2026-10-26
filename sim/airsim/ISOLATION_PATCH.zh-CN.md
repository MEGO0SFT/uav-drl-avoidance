# 多机隔离补丁 —— 精确复现

[**English**](ISOLATION_PATCH.md) | **简体中文**

[← 返回 README](../../README.zh-CN.md) · [仿真配置](README.zh-CN.md) · [UE4 工程](ENVIRONMENT.zh-CN.md)

没有这个补丁，平台无法训练单机策略：十架无人机全部生成在同一点（`X=0, Y=0, Z=-0.5`），而原版
AirSim 允许它们互相碰撞、相互遮挡，于是每一帧观测都被另外九个智能体污染。

该补丁**不是** C++ 改动，也**不是**关卡编辑。它是对一个 blueprint 中五个网格组件所做的碰撞
配置（collision profile）改动。

| | |
|---|---|
| 资产 | `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` |
| 涉及组件 | `BodyMesh`、`Prop0`、`Prop1`、`Prop2`、`Prop3` |
| 大小 / 哈希 | 168 972 bytes · sha256 `0f4f91c1…` |
| 入库副本 | [`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset)（字节完全相同） |

> **⚠️ 改完之后，要把文件放回去。** 修改必须在
> `<project>/Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` 上进行，**保存后的资产必须再
> 复制回这个确切的路径**，替换原文件 —— 否则游戏加载的仍然是不带补丁的 pawn。路径本身就是修复
> 的一部分：十架 `SimpleFlight` 无人机实例化的正是 AirSim 自己在
> `Plugins/AirSim/Content/Blueprints/` 下的那份副本，而丢在别处（例如工程级的 `Content/`）的副本，
> 只有当它按同样的相对路径遮蔽（shadow）原文件时才会被采用。如果不确定，就保存、关闭编辑器，
> 然后确认 `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` 的 mtime 和文件大小确实
> 发生了变化。

![BP_FlyingPawn 中 BodyMesh 与 Prop0..Prop3 的碰撞设置](figures/collision_settings_BP_FlyingPawn.png)

*`BP_FlyingPawn` 的五个网格组件，Collision 面板 —— 预设 `Custom`，碰撞启用为
`Query and Physics`，对象类型 `Pawn`，`Visibility` 与 `Camera` 的 trace 响应为 `Ignore`，
`Pawn` 对象响应为 `Ignore`；其余全部为 `Block`。*

> 截图里的编辑器是中文界面：`碰撞预设` 即 Collision Presets，`碰撞已启用` 即 Collision Enabled，
> `对象类型` 即 Object Type；右侧三列分别是 `忽略`（**Ignore**）、`重叠`（**Overlap**）、
> `阻挡`（**Block**）。要改的只有两个格子 —— 检测响应里 `Visibility` 那一行、以及物体响应里
> `Pawn` 那一行的 `忽略` 勾选项。

## 具体设置

对**五个网格组件中的每一个**（BodyMesh 以及四个螺旋桨）都应用下列设置：

| 分类 | 设置 | 取值 |
|---|---|---|
| Collision | Simulation Generates Hit Events | ✅ 开启 |
| Collision | Phys Material Override | `QuadBodyPhysicsMaterial` |
| Collision | Generate Overlap Events | ✅ 开启 |
| Collision | Can Character Step Up On | `Yes` |
| Collision | **Collision Presets** | **`Custom`** |
| Collision | Collision Enabled | **`Collision Enabled (Query and Physics)`** |
| Collision | Object Type | **`Pawn`** |
| Trace Responses | **Visibility** | **`Ignore`** |
| Trace Responses | Camera | `Ignore` |
| Object Responses | WorldStatic | `Block` |
| Object Responses | WorldDynamic | `Block` |
| Object Responses | **Pawn** | **`Ignore`** |
| Object Responses | PhysicsBody | `Block` |
| Object Responses | Vehicle | `Block` |
| Object Responses | Destructible | `Block` |

在编辑器中的操作：打开 `BP_FlyingPawn`，依次选中每个网格组件 → *Details → Collision* → 展开
**Collision Presets**，将其设为 *Custom*，然后展开 *Trace Responses* / *Object Responses*，把上表
中标为 **Ignore** 的两个单元格翻转过来。其余各项全部保持 `Block`。

## 为什么恰好是这两个单元格

两个目标是由两个**不同**的响应单元格解决的 —— 这正是最容易搞错的地方。

### 1. LiDAR 不能看到其他无人机 → `Visibility = Ignore`

AirSim 的 LiDAR 是在 **`ECC_Visibility`** 通道上做射线检测（ray cast）的，并且只忽略发出该射线的
那架无人机：

```cpp
// Plugins/AirSim/Source/UnrealSensors/UnrealLidarSensor.cpp:131
bool is_hit = UAirBlueprintLib::GetObstacle(actor_, ...,
                                            hit_result, actor_, ECC_Visibility);

// Plugins/AirSim/Source/AirBlueprintLib.cpp:615-626
trace_params.AddIgnoredActor(actor_);        // only the ego actor
if (ignore_actor != nullptr) trace_params.AddIgnoredActor(ignore_actor);
return actor->GetWorld()->LineTraceSingleByChannel(hit, start, end, collision_channel, trace_params);
```

这条路径上没有任何 `Pawn` 过滤。射线想穿过另一架无人机，**唯一**的办法就是那架无人机的网格体在
`ECC_Visibility` 上回答 `Ignore` —— 这正是 `Visibility` 这一格所做的事。把它设为 `Block`，LiDAR
就开始测量另外九架无人机，而不是世界。

### 2. 无人机之间不能互相碰撞 → `Pawn = Ignore`

机体间碰撞是 actor 对 actor 的交互，UE 是通过**对象响应**矩阵而非 trace 通道来判定的。每架无人机
的网格体都是 `Object Type = Pawn`，因此把 `Pawn` 这一行设为 `Ignore` 后，无人机之间的重叠会直接
穿透；而 `WorldStatic` / `WorldDynamic` / `PhysicsBody` 仍保持 `Block`，无人机照常会撞上墙体、地面
和障碍物 actor。

## 哪些东西**不能**改

- **`Object Type` 必须保持 `Pawn`。** 一旦漂移成 `WorldDynamic`，`Pawn` 这一行就不再适用，无人机会
  重新开始互相碰撞。
- **`Collision Enabled` 必须保持 `Query and Physics`。** `Query only` 会去掉环境交互和碰撞检测所
  依赖的碰撞。
- **`Visibility` 必须保持 `Ignore`。** 正是这一格让 LiDAR 对无人机"透明"；它也常常是人们在纳闷
  "为什么我的 LiDAR 忽略了自己的无人机"时不小心"修正"掉的那一格。

## 验证补丁是否生效

1. 打开打包好的仿真器，让十架无人机停在生成点 —— 它们应当在视觉上互相重叠，而不是把彼此推开。
2. 把第二架无人机飞到第一架前方，观察第一架的点云：点必须出现在障碍物**之后**，而不是落在其表面上。
3. 在 PIE 中，`BP_FlyingPawn → BodyMesh → Collision` 应显示 `Custom`、`Pawn`、`Visibility: Ignore`、
   `Pawn: Ignore`。

## 补丁已经在哪些地方

| 位置 | 是否含补丁 |
|---|---|
| `airsim-blocks2-standalone-win64.zip`（release） | ✅ 已编译进去 |
| `airsim-blocks2-project-win64.zip`（release） | ✅ 以 `Plugins/AirSim/Content/Blueprints/BP_FlyingPawn.uasset` 形式存在，因此重新构建后仍保留 |
| 本仓库 | ✅ [`project/Content/BP_FlyingPawn.uasset`](project/Content/BP_FlyingPawn.uasset)，字节完全相同 |

想要继承已发布的行为，就保持该资产原样。从[上游 AirSim](https://github.com/microsoft/AirSim)
1.7.0 恢复原版 pawn，会在无声无息中把补丁撤销掉。
