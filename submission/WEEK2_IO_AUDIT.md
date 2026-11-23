# Week 2 IO 初始审计（2026-11-01）

仓库 `/home/minsu/RM/hnurmvision_edu`，分支 `feature/minsuhnu-agent-practice`，HEAD `1aa9b7e`。
初始工作区仅 `io/io_context.hpp`、`lessons/lesson02_io.cpp` 已修改，439 行新增、3 行删除；已完整保存在 `/tmp/week2_io_original/preexisting.patch`，原文件同目录备份。未执行提交、远程操作或破坏性 Git 命令。

依据：桌面《第二讲：从业务分析到程序 IO.pdf》5.1–6.4、实际消息定义及仿真器 `src/ros2/plugin.rs`。交接报告仅作为历史背景。

| 分类 | 审计结论 |
| --- | --- |
| 已完成且有证据 | 当前源码包含独立 Context、节点、三个正确 topic、SensorDataQoS、Executor 线程；RGB/BGR/mono 转独立 BGR；读等待上限10ms；main 有信号处理；第三至第五讲仍为原始骨架。以上为源码证据，尚非当前运行验收。 |
| 已实现但尚未验证 | 单帧消费、四元数 ZYX、ROS/steady 时间映射、rad→deg、AUTO_AIM/allow_fire 门控、Close STOP/cancel/join、重复关闭。 |
| 存在缺陷 | Write 仅检查反馈，没有图像有效性门控；Read 未限制图像年龄；时间回退用了处理时刻而非接收时刻；Close 后缓存仍保留；未限制坏输入警告频率。控制时 target_distance=0 与讲义5.4不符。 |
| 尚未实现 | 每100轮IO日志、每秒姿态/弹速日志、预览限频缩放；当前独立测试、性能数据、动态键盘验收、最终录屏。 |

协议核对：仿真器将 Bevy YXZ 提取为 yaw/负pitch/roll，反馈单位deg，Write反转pitch由仿真器内部完成；内部按项目 ZYX yaw-pitch-roll 组装四元数。消息文件本身没有描述旋转顺序，讲义也没有进一步公式，因此记录该约定与来源，使用数值测试验证。

讲义存在距离说明冲突：4.2称-1停止，5.4称无测距填-1、停止填0；实际仿真器 state=0 或 distance=-1 都停止。本次遵循具体任务5.4：控制但无测距=-1，停止=0。故只证明Write协议转换，不声称未知距离的指令能使仿真器运动；主程序第五讲未实现，应始终STOP。

修复范围：两个IO文件、必要的main日志/预览；不改第三讲及之后算法，不改仿真器源码/配置。全部运行保持allow_fire=false。

## 修复阶段补充发现

进一步读取 `simulation/simulator/src/ros2/prelude.rs` 的 `M_ALIGN_MAT3` 后，确认原实现的roll符号错误：Bevy的X/Y/Z轴换基到ROS的-Y/Z/-X，因此由消息重建的四元数应为 `Rz(yaw) Ry(pitch) Rx(-roll)`。已修正，并用独立的 `M * R_Bevy_YXZ * M.transpose()` 数值测试验证含非零roll的组合姿态。此为在初始审计基础上进一步追踪源码发现的缺陷。
