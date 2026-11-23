# Week 2 IO 验收报告

验收日期：2026-11-01（本地时区 Asia/Shanghai）。所有数字来自本次真实运行，历史交接记录不作为当前验收证据。

## 1. 环境、范围与工作保护

- 仓库：`/home/minsu/RM/hnurmvision_edu`；分支：`feature/minsuhnu-agent-practice`；HEAD：`1aa9b7e`。
- Ubuntu 22.04.5 LTS，Linux 6.8.0-138-generic，ROS 2 Humble，rclcpp 16.0.21，OpenCV 4.5.4，C++17 Release；X11桌面2560×1600；NVIDIA RTX 5070系列本机GPU。
- 正式仿真Domain89；协议测试隔离在Domain90。所有运行配置 `allow_fire=false`，无真实设备接入，无FIRE消息的仿真演示。
- 官方依据：本机[第二讲讲义](</home/minsu/桌面/RM 第二阶段培训/第二讲：从业务分析到程序 IO.pdf>)，重点5.1–6.4；消息定义和仿真器实际源码。未使用互联网或远程Git操作。
- 保留原有两个IO文件的全部既有工作；原始差异见 `evidence/preexisting.patch`，原文件额外备份 `/tmp/week2_io_original/`。初始审计见 `WEEK2_IO_AUDIT.md`。
- 产品代码只修改 `io/io_context.hpp`、`lessons/lesson02_io.cpp`、`src/main.cpp`。后者仅补充日志、预览缩放限频和退出时关闭窗口。第三至第五讲算法、类型接口、根CMake、仿真器源码与配置均未修改。没有commit、push、reset或clean。

## 2. Open / Read / Write / Close

| 功能 | 实现及实际验证 | 结果 |
| --- | --- | --- |
| Open | 独立Context；`/training_io`；Image/Recv订阅和Send发布器；SingleThreadedExecutor后台线程实际执行回调；重复open保护。真实DDS发现、节点列表和topic详细信息已核对。 | 通过 |
| Read图像 | RGB8转换BGR；BGR8深拷贝；mono8扩展3通道；保留step；输出CV_8UC3并独立拥有内存；坏buffer/尺寸/encoding拒绝；最新帧槽+pending消费；非零重复stamp拒绝。真实仿真为rgb8、1440×1080、step4320。 | 通过 |
| Read姿态 | 角度deg→rad，按仿真器YXZ和M_ALIGN换基重建 `Rz(yaw) Ry(pitch) Rx(-roll)`，单位四元数。非零yaw/pitch/roll组合由独立矩阵换基测试核对。bullet_speed读取m/s，零弹速/非有限姿态拒绝。 | 通过 |
| Read时间 | 图像在回调接收时以ROS now与steady now配对映射，保留消息年龄；没有把ROS epoch直接转换为steady epoch。等待最多10ms；反馈/图像过期500ms拒绝；未来超过100ms拒绝；无输入nullopt。负sec/非法nanosec被安全拒绝。 | 通过 |
| Write | 绝对rad→deg；ROS header时间；frame `gimbal_link`；无有效新输入、过期、非AUTO_AIM、control=false、NaN/溢出角均STOP；allow_fire=false时shoot=true仍只CONVERGING。独立订阅者核对30°/-15°数值及距离字段。 | 通过（无发射配置） |
| Close | 先发布STOP，保留30ms发送窗口，再cancel/shutdown/join、移除节点、释放订阅/发布器和缓存；重复close、同对象重新open已测试。q和SIGINT均实际退出码0，独立观察到最后STOP，关闭后持续静默且发布器归零。 | 通过 |

共享回调状态由mutex保护；公开四接口由主线程顺序调用，没有宣称可由多个调用线程并发open/close/read。GUI始终在主线程；回调只存最新消息，不做图像拷贝和显示。Executor异常被捕获并失效输入，避免线程异常直接terminate。

### 姿态协议的依据

`simulation/simulator/src/ros2/plugin.rs`：Bevy按YXZ提取旋转，消息为 `(yaw, -pitch_Bevy, roll_Bevy)`。`src/ros2/prelude.rs` 的M_ALIGN把Bevy轴X/Y/Z映到ROS -Y/Z/-X，因此roll也必须取负。修正前直接使用正roll是此次审计发现的缺陷。独立测试先在Bevy坐标生成旋转，再计算 `M * R_Bevy * M.transpose()`，避免仅把被测公式抄成期望值。

### 距离字段的已知冲突

讲义4.2称-1停止，5.4明确“无测距填-1，停止填0”；仿真器实际 `state==0 || distance==-1` 时停止。本实现遵循具体任务5.4：CONVERGING且无测距填-1；STOP填0。**因此无测距的CONVERGING只能证明消息转换，不能声称能驱动仿真器云台。** 未伪造距离、瞄准目标或实际发射。本次演示第五讲仍为空骨架，正式输出应始终是STOP。

### 时间对齐的实际边界

基础实现使用最近收到的反馈，没有姿态插值。日志 `feedback_rx_delta` 是反馈接收时刻减图像映射时刻，包含传输/排队因素，不等于严格的两个传感器采样时差。方向误差近似 `角速度 × 时间差`：2rad/s且差30ms时约3.44°，在3m处横向约18cm。本程序不把这种近似包装成精确同步。

零图像stamp或ROS时基差超过5s时使用接收时刻（后者限频警告），此时不能恢复真实采集时刻。正常仿真使用系统ROS时间，已实测映射有效。零stamp无法借助时间戳识别发布端重复发送的同一帧，但同一回调保存的消息只交付一次。

## 3. 编译与独立验证

`./scripts/build_training.sh` 实际成功：hnurm_interfaces、training_io、training_lessons、training均构建完成。最终产物 `build/training`。无需重编仿真器，既有 `simulation/simulator/bin/daedalus` 已启动并真实运行。原始构建日志：`evidence/build.log`，最终代码/二进制哈希：`evidence/final_sha256.txt`。

隔离Domain90协议探针：`submission/tools/run_probe.sh`，完整源码在 `tools/io_probe.cpp`，独立CMake不改主工程。最终 44 条PASS断言，覆盖图像编码/stride/所有权、姿态换基、时间映射、重复帧、无输入、模式、禁止发射、非法数据、过期、Close/重启。最终结果见 `evidence/probe.log`。未运行allow_fire=true分支，不宣称测试过FIRE。

测试工具开发时出现过DDS最佳努力首包丢失、不同topic回调先后顺序导致的断言失败；在探针中加入发现后的稳定时间和两条回调的调度等待后解决。关闭观察器最初误用rclpy Subscription不存在的方法，已改为 `node.count_publishers` 并重测成功。上述是测试工具修复，未隐瞒为已通过，也未更改产品逻辑来迁就断言。多个独立Context重复初始化日志的rclcpp WARN仍保留，没有屏蔽。

## 4. 真实ROS通信、显示与动态数据

- 本次实际节点：`/robomaster/simulator`、`/training_io`；三个IO topic存在。见 `evidence/nodes.txt`、`topics.txt`。
- topic详细类型和QoS：`topic_image_info.txt`、`topic_feedback_info.txt`、`topic_command_info.txt`。图像发布端实际RELIABLE，培训SensorDataQoS BEST_EFFORT订阅可兼容；反馈/控制实际与BEST_EFFORT配置兼容。IO保持depth1避免积压。
- `ros2 topic hz` 三条命令已实际执行；原始 `hz_0.txt`、`hz_1.txt`、`hz_2.txt`，最后窗口约94.63/151.05/128.17Hz。它们是中间版本联调采样，最终版本的独立测量另列下表。
- `ros2 topic echo /vision_send_data --once` 实际返回state0、distance0、零角、frame gimbal_link和ROS时间。见 `command_echo.txt` 和录屏终端日志 `demo_ros_terminal.log`。
- 实际独立相机窗口 `RoboMaster training` 显示实时图像。算法输入保持1440×1080；仅显示副本缩到最大960宽，至多约30次/秒刷新；waitKey即使断流仍处理q/Esc。每100轮打印状态，每秒左右打印姿态/弹速，未逐帧刷屏。
- 初次实际键盘操作：yaw从-90°到-115.72°，pitch从25°到7.89°，弹速25m/s。F5将auto-aim切到OFF后，真实方向键事件操纵云台。仿真器auto-aim开启时主动禁用手动云台键，不能直接按箭头就误判输入不变。
- 原版录屏运行日志自身的yaw范围 -103.41°–-90.00°，pitch范围 15.37°–25.34°，bullet_speed始终25m/s（`demo_training.log`）。

## 5. 性能与内存（实测，不保证固定FPS）

方法：只读rclpy观察器订阅三话题；先预热2s，再采样20s或30s。Hz由消息到达首尾的间隔计算；消息年龄是消息ROS header到观察器回调的时间，**不是完整自瞄算法延迟或物理曝光延迟**。`image_age`日志另提供Read转换完成时的消息年龄。RSS读取/proc，表中排除启动前3s。全程仿真器自身preview配置未改。采样未进行CPU/GPU资源隔离，频率受场景、焦点、桌面合成和观察器开销影响。

| 条件 | 图像Hz | 反馈Hz | 控制Hz | 图像最大间隔ms | 图像年龄均值/p95 ms | training RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 最终代码，仿真器聚焦，培训预览开（20s） | 102.73 | 165.30 | 136.88 | 157.0 | 28.4 / 33.0 | 116.9–116.9 |
| 最终代码，仿真器聚焦，培训预览关（20s） | 92.46 | 169.71 | 145.19 | 168.8 | 28.5 / 35.2 | 96.6–96.6 |
| 最终代码，培训窗口聚焦/仿真器失焦，预览开（30s） | 11.33 | 59.67 | 93.11 | 930.1 | 34.0 / 40.5 | 116.8–116.8 |
| 修复中间版本，聚焦场景，另有ros2 hz观察器（20s） | 96.19 | 157.20 | 131.80 | 131.5 | 30.1 / 35.8 | 117.1–117.5 |
| 最终代码，预览关，初轮（30s） | 80.04 | 167.80 | 140.18 | 179.6 | 33.7 / 45.5 | 91.9–91.9 |
| 中间版本，包含窗口调整和键盘操作（20s） | 56.17 | 124.97 | 116.57 | 950.5 | 33.5 / 41.9 | 116.8–117.2 |

聚焦、开/关培训预览两轮，图像消息年龄均值均约28.5ms；控制吞吐约137/145Hz。图像吞吐没有稳定地呈现关闭预览一定更高，**不能根据这些短测声称显示带来某个固定倍率改善**。开启预览的最终聚焦采样图像间隔p95约27ms、最大157ms；关闭约32ms、最大169ms。没有观察到持续图像排队或RSS单调增长：最终聚焦开预览约119740KiB，关预览约98872KiB，20s采样稳定。短测不等于长期泄漏证明。

失焦的最终代码样本出现11Hz图像、最大930ms间隔，反馈约60Hz；恢复仿真器焦点后约103Hz图像、165Hz反馈。窗口焦点影响已通过真实聚焦操作重测确认，但尚未在仿真器内部定位到具体调度/渲染根因；不改底层代码。窗口调整的早期样本还有950ms间隔。这些性能异常仍然存在，不删除样本，也不声称已经消除。录屏应保留仿真器焦点并避免拖动窗口。

原始JSON/日志：`perf_focused_display.*`、`perf_focused_headless.*`、`perf_final_display.*`、`perf_display_stable.*`、`perf_headless.*`、`perf_display.*`。原版录像的画面索引（保留作对照）：约12s Open节点与topic，18s Read动态姿态，28s Write安全STOP，65s Close退出与静默。

每份保留采样数量、间隔分布、消息年龄、姿态、STOP计数和RSS序列，便于重新核对。

## 6. Close与重启验收

- q（原版录屏）：独立观察器 `close_q.json`，退出前最后消息state0/distance0，触发后4s以后的消息数为0，发布器数量0；`demo_training.log`记录STOP及资源释放，真实终端显示Training-exit=0。
- SIGINT（与Ctrl+C同一信号）：`close_sigint.json`、`close_focused_display.json`，同样最终STOP、静默、发布器0，实际进程退出码0。通过终端发送SIGINT，没有用SIGKILL模拟正常退出。
- 培训已多次真实重启；同一个IoContext对象的重复close与重新open由协议探针验证。
- DDS采用BEST_EFFORT，30ms发送窗口不保证任何网络下STOP必达；本机测试由独立订阅者实际收到。本次未模拟网络完全断开，不能宣称网络故障时仍保证最后一包送达。

## 7. 录屏与复现（单条粘贴版为当前主提交视频）

当前主文件`submission/week2_io_demo.mp4`，同内容独立版本`week2_io_demo_paste.mp4`。**55.20秒，1600×1000，15fps，H.264，828帧全部解码，约3.74MiB。** 真实X11桌面连续采集，无剪辑、拼接或伪造ROS数据。79秒逐字输入版`week2_io_demo_hands_on.mp4`及82秒原版`week2_io_demo_baseline.mp4`均保留。

按用户最新操作偏好：较快移动实际鼠标并点击窗口；通过真实剪贴板和Ctrl+Shift+V一次粘贴一条完整命令，等待后单独按Enter，再查看结果；同一终端连续执行命令时无需重复点击。共11次单条粘贴，没有多行批量输入。实际复制的每条命令由动作记录和终端原始日志保留，录制后恢复原剪贴板图片并核对像素，输入法恢复rime。未更改产品IO/算法代码，自动操作来源保留，不标称为用户本人亲自操作。

关键索引：约17.9s Open节点，27.8s Read动态姿态，35.1s单条命令整行粘贴，36.9s Write实际安全STOP，48.3–54.3s Close退出码及节点消失。人工抽帧检查确认结果可辨认、任务窗口无遮挡、没有无关聊天窗口露出。首次粘贴试录因背景露出无关窗口被弃用，保留在`/tmp/week2_paste_background_trial.mp4`，未作为提交材料；正式版临时最小化该窗口，录后恢复原可见状态。

独立只读观察器`evidence/paste_demo_observer.json`收到4224条消息，全部state0；最后state0/distance0；退出后发布器0，静默约7.04s，发布器消失后消息数0。yaw范围-104.68°至-89.98°，pitch范围14.20°至25.00°，弹速25m/s。`paste_training.terminal.log`确认STOP、资源释放和真实退出码0。

最新完整复查见`WEEK2_SUBMISSION_CHECKLIST.md`；复审历史`WEEK2_VIDEO_REVIEW.md`；操作指南`WEEK2_RECORDING_GUIDE.md`。视频详细数据`evidence/video_validation.json`，单条粘贴动作`paste_demo_actions.json`，抽帧`paste_video_*.jpg`。复现工具为`tools/record_paste_demo.py`，应在没有仿真器/培训进程且桌面空闲时运行。

```bash
cd /home/minsu/RM/hnurmvision_edu
./scripts/build_training.sh
# 终端1
./scripts/run_simulator.sh
# 终端2
./scripts/run_training.sh
# 终端3
submission/tools/ros_env.sh ros2 node list
submission/tools/ros_env.sh ros2 topic echo /vision_send_data --once
```

## 8. 最终结论与限制

Week2要求的四接口、必要日志、真实图像、动态姿态/弹速、安全STOP、退出和重启均有当前运行证据，代码/报告/真实运行录屏三个交付已形成，可用于Week2基础提交。未进行Git提交或推送。未实现或修改第三至第五讲算法，未测试或演示发射。

仍需如实保留的边界：仿真器失焦时明显降帧及偶发长图像间隔；最近反馈近似对齐；零stamp/异时基只能退回接收时间；无测距-1与仿真器停止规则冲突；内存仅短测；BEST_EFFORT最终STOP只验证本机实际送达。若考核要求稳定硬实时帧率、精确同步或主动控制云台闭环，需要后续单独解决，当前不能宣称这些扩展要求已验收。
