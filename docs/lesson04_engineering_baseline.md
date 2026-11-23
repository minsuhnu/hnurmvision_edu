# 第四讲工程检查与开发基线

检查日期：2026-11-15，Asia/Shanghai。此文件记录本轮实际检查，不将历史报告中的测试计入本轮验收。

## 资料

已找到并完整阅读 `/home/minsu/桌面/RM 第二阶段培训/week4/第四讲 从观测到目标状态：位姿、坐标与基础跟踪 .zip` 的 Markdown，检查正文、相机附录、旋转表示附录、装甲板图纸，以及相关代码和公式图片。ZIP 哈希见 `submission/lesson04_evidence/baseline/materials.json`。讲义的欧拉角轴名依赖其坐标约定；实现以工程的右手坐标系与明确的变换方向为准。

## 工程及基线

工程根目录 `/home/minsu/RM/hnurmvision_edu`，CMake 项目名 `hnurmvision_training`。结构为 `include/` 数据结构、`io/` ROS 输入输出、`lessons/` 各讲算法、`src/` 实时调度和显示、`configs/` 标定与通信、`simulation/simulator/` Bevy 仿真器、`simulation/ros2/` 自定义消息、`test/` 独立 PnP 测试、`submission/` 历史提交材料。原 README 描述的 IO 骨架已经过时。

- 当前分支 `feature/minsuhnu-agent-practice`，存在大量用户未提交修改；保留分支和原有文件，不重置、不覆盖已有材料。
- `./scripts/build_training.sh` 本轮成功：ROS2 消息构建、CMake 配置、C++ 编译与链接全部通过。C++17；OpenCV 4.5.4；Eigen3、yaml-cpp；ROS2 Humble；CMake 3.22.1。
- 在独立 `/tmp/hnurm-lesson04-baseline` 开启 `BUILD_LESSON04_PNP_TEST`，已有 CTest 通过。它是合成输入的数值测试，不能证明真实检测位姿准确。
- `./scripts/run_simulator.sh` 实际启动成功；只读订阅 5 秒收到 image=302、camera_info=302、tf=300 条消息。图像为 1440×1080 RGB8。
- `timeout --signal=INT --kill-after=3s 6s ./scripts/run_training.sh` 启动、读取图像与反馈、正常释放 ROS 资源。timeout 返回 124 表示按计划到时退出；日志有 STOP 和 executor joined。
- 仿真资源加载有已有的 `Unknown vertex attribute COLOR_1` 警告，未将其写成“零警告”。

## 接口和缺失功能

第三讲入口 `std::vector<Armor> lesson03_detection(const cv::Mat&)`，`Armor.corners` 顺序左上、右上、右下、左下；还有 `large_armor`、`class_id`、`confidence`。OpenCV 后端提供灯条轮廓长轴端点，Pose 提供预测关键点，Hybrid 保留 OpenCV 几何并增加类别。类别置信度不能作为物理角点对应已验证的依据。

第四讲入口 `lesson04_tracking(armors, timestamp, imu_q, calibration, state)` 当前始终返回空。已有 IPPE 双候选 PnP 内核、输入检查、参考点转换与离线测试；尚缺真实检测接入、选板、关联、速度估计、状态管理、未来预测、同帧真实值误差、显示与本周 MP4。旧 `submission/week2_*.mp4` 不能作为第四讲视频。

`configs/simulation.yaml` 已有 K、5 项零畸变、相机到云台 R/t。配置 fx=fy=1303.6753236814714，主点 (720,540)；实际 camera_info 的 fx=fy=1303.675283386667，差约 4.0e-5 px。分辨率在 `io/config.cpp` 目前固定 1440×1080；更改分辨率时必须同步内参，不能只缩放画面。

`SensorData.timestamp` 是由图像 ROS SystemTime 映射到 steady_clock 的时间；IMU 使用最新接收反馈，并非严格图像时刻插值。仿真运动用 FixedTime：静止 kind0、正弦平移 kind1、旋转 kind2。UI 按钮或 F6 选靶；平移幅值 0.5m、周期 4s，不是全程匀速。

## Ground Truth 与几何风险

运行中 `/tf` 确有 `map→armor_0…armor_21` 及相机链，提供装甲 CENTER 的位置和旋转；不能把车体根位置、可视化 Marker 的改转朝向或图像时间最近的 TF 直接当同帧装甲真值。原图 header 和 TF 采样时刻不同。

已有未提交 `capture_audit.rs` 可在渲染提取阶段冻结实际 CaptureCamera、CENTER、MARKER 和 FixedTime，并随 GPU copy ticket 导出 PNG/JSON。该新源码尚未通过本轮运行验收；`bin/daedalus` 是旧版本。历史 `/tmp` 审核产物已不存在，本轮按当前授权重新构建，不沿用旧 SHA 或旧运行结论。

MARKER 的小板宽高约 133.77×53.89mm，而灯条网格包围盒中心线间距/长度约 125.80×59.61mm。它们是不同观测定义。禁止直接把现有 `Armor.corners` 宣称为 MARKER 固定点。后续需要使用明确定义的网格观测近似，并用独立真值实测其偏差；低重投影误差不证明物理准确。

## 后续计划

1. 重建当前 ROS2 仿真源码，验证 PNG/JSON 同包采集与实际渲染投影，先打通可信实验输入。
2. 复用 IPPE 内核；为灯条端点近似模型单独声明来源与适用域，保留多解及重投影诊断，不伪装为固定 MARKER 对应已验证。
3. 先按实际图像中心选板，再以位置、尺寸和类别约束保持关联；在已有 TrackingState 增加简单状态机和基于真实时间的匀速估计。
4. 增加可配置预测时域、非阻塞预测队列、按仿真时刻插值的同 ID 真实值对照和不同颜色投影框。
5. 查阅高校官方源码，选取适合本工程的改进，测量其效果并如实记录。
6. 完成静止/平移场景的可重复测试、统计、实际程序视频和提交文档。发生用户规定的关键阻塞时暂停，不用合成数据替代验收。
