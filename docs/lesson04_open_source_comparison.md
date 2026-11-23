# 第四讲高校开源方案对比与实际应用

查阅日期：2026-11-15。主要对比 PnP 的输入模型、目标关联和坐标变换。学校归属见 [同济 SuperPower 组织页](https://github.com/TongjiSuperPower) 与 [PnX-HKUSTGZ 组织页](https://github.com/PnX-HKUSTGZ)。访问内容的 URL、文件快照和 SHA-256 保存在 `submission/lesson04_evidence/research/sources.json`；分支会变化，复核时应比较快照哈希。

| 项目 | PnP 方法 | 跟踪方法 | 坐标变换 | 值得借鉴的设计 |
|---|---|---|---|---|
| 同济大学 SuperPower / [sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) | IPPE；分别定义大小板；解算后可利用安装俯仰假设优化 yaw | 状态管理与 EKF 分开；目标状态含位置、速度、角速度及车体几何量 | YAML 管理内外参；相机→云台→稳定系；支持重投影 | 将物理安装约束用于姿态优化；在稳定系预测，再回投图像 |
| 香港科技大学（广州）PnX / [auto-aim](https://github.com/PnX-HKUSTGZ/auto-aim) | IPPE Generic 保留候选；将灯条端点组合成上下左右中点，采用相应三维菱形模型 | EKF；按编号、位置与 yaw 关联，管理四种状态；代码还支持双板更新 | 检测/跟踪节点分工，使用惯性系状态 | 显式保持多解；同类目标连续关联，避免仅按图像中心反复重选 |

同济的相关实现是 [`tasks/auto_aim/solver.cpp`](https://github.com/TongjiSuperPower/sp_vision_25/blob/main/tasks/auto_aim/solver.cpp)、[`tracker.cpp`](https://github.com/TongjiSuperPower/sp_vision_25/blob/main/tasks/auto_aim/tracker.cpp) 和 [`target.cpp`](https://github.com/TongjiSuperPower/sp_vision_25/blob/main/tasks/auto_aim/target.cpp)。本次采用物理安装约束和预测重投影的思路，跟踪仍使用基础匀速模型。物点尺寸按本仿真资产计算。

PnX 的相关实现是 [`armor_detector/src/pnp_solver.cpp`](https://github.com/PnX-HKUSTGZ/auto-aim/blob/main/armor_detector/src/pnp_solver.cpp) 和 [`armor_tracker/src/tracker.cpp`](https://github.com/PnX-HKUSTGZ/auto-aim/blob/main/armor_tracker/src/tracker.cpp)。它的中点模型有自己的对应关系，不能直接套用到本工程 TL/TR/BR/BL 数据。这里主要参考多候选处理和连续关联的设计。

另核对了上游 [`chenjunnn/rm_auto_aim` 的 PnP 源码](https://github.com/chenjunnn/rm_auto_aim/blob/main/armor_detector/src/pnp_solver.cpp) 与 [跟踪源码](https://github.com/chenjunnn/rm_auto_aim/blob/main/armor_tracker/src/tracker.cpp)，用于核对分支差异。上游采用灯条四端点矩形，PnX 当前文件采用菱形中点，对应点的定义不同，移植前需重新核对。

## 已应用及实测

1. `src/lesson04_runtime.cpp` 在 PnP 前选板，锁定后按预测投影、尺寸和可用类别保持关联。受控双目标回归中，中心重选发生 1 次切换，连续关联为 0 次。真实实验主要跟踪单块前板，多目标效果仍需补充实测。
2. 保留原始 IPPE 双候选；借鉴物理安装约束思路，结合本资产灯条网格，拟合与检测器一致的轮廓端点统计量。这里使用的是灯条网格观测修正，输出需与原始 PnP 候选分开记录。
3. 对 run02 相同的 240 张真实仿真图像，关闭网格修正时，位置平均误差 8.861 cm、姿态 19.195°、到期预测 9.936 cm；开启后分别为 2.562 cm、4.261°、5.690 cm。两组均使用相同安装先验、相同图像、同一真值和相同 CV 参数；区别仅为网格观测修正。数据在 `run02/no_mesh_*` 与 `run02/live_*`。该结果对应当前资产和测试序列，实车需要重新验证。

当前方案保持基础匀速模型，以真实时间的短窗直线拟合估速度；未引入 KF/EKF、弹道补偿或自动发射。
