# 真实录屏复现

本次使用本机已有的 GStreamer ximagesrc 和 OpenCV/FFmpeg H.264 编码器，不安装 OBS/FFmpeg 命令行工具。`week2_io_demo.mp4` 是实际 X11 桌面采集，不是图片拼接或虚构 ROS 数据。

## 手动复现（3个终端）

所有终端先 `cd /home/minsu/RM/hnurmvision_edu`。配置保持 `display: true`、`allow_fire: false`。

1. 终端1输入 `./scripts/run_simulator.sh`。等训练场景出现。
2. 终端2输入 `./scripts/run_training.sh`。把独立 `RoboMaster training` 相机窗口和终端2摆在可见位置。
3. 终端3输入 `source /opt/ros/humble/setup.bash`、`source simulation/install_interfaces/setup.bash`、`export ROS_DOMAIN_ID=89`。
4. 录屏从两个程序启动前开始，展示Open。终端3执行 `ros2 node list` 和 `ros2 topic list`，显示 `/training_io` 和三个IO topic。
5. 点击仿真器窗口使其获得焦点。底部若显示 `auto-aim:ON`，按一次F5切换到OFF，此时方向键才能手动转动云台。轻按左右/上下方向键，观察相机和每秒yaw/pitch/弹速日志。不要按Space。
6. 终端3执行 `ros2 topic echo /vision_send_data --once`，展示真实 `target_state.data: 0`、`target_distance: 0`、`gimbal_link` 与ROS时间戳。第五讲仍为空骨架，STOP就是正确结果。
7. 点击相机窗口，按q或Esc；或在终端2按Ctrl+C。展示 `STOP published`、`executor joined; ROS resources released` 和退出码0。终端3再次 `ros2 node list`，`/training_io`应消失。
8. 停止录屏，建议保存 `submission/week2_io_demo.mp4`。先关培训，再关仿真器。

## 当前单条粘贴版

当前主视频采用真实剪贴板，每次Ctrl+Shift+V只粘贴一条命令、按Enter执行，查看结果后再继续。使用较快鼠标移动；同一终端内连续命令无需重复点击。运行工具：

```bash
/usr/bin/python3 submission/tools/record_paste_demo.py
```

运行前确保仿真器/培训程序未启动，录制期间保持桌面空闲约60–90秒。工具使用本机已有GTK剪贴板，不安装新软件；结束后恢复剪贴板原图片或文字、输入法及暂时最小化的背景窗口。新文件为`week2_io_demo_paste.mp4`，审查通过后主文件`week2_io_demo.mp4`已同步。

## 鼠标操作版复现工具

上一版为79秒逐字输入版，保留作对照。`tools/record_mouse_demo.py` 会打开三个真实交互终端，移动实际鼠标点击窗口，逐字键入命令，并操作仿真器和相机退出。后台只读观察器保存安全与退出证据。只在没有现存仿真器/培训程序时运行：

```bash
/usr/bin/python3 submission/tools/record_mouse_demo.py
```

录制时约90秒不要操作键盘、鼠标或切换窗口。工具会临时选择英文键盘，完成后恢复原输入法；不会修改输入源配置或用户剪贴板。录制结束可正常操作桌面。脚本保留仿真器窗口，培训程序已正常关闭。

新版输出 `week2_io_demo_hands_on.mp4`，人工复审和完整解码通过后更新主文件 `week2_io_demo.mp4`。旧版保留为 `week2_io_demo_baseline.mp4`。新版是代理受授权操作的真实录屏，不标称为用户本人亲自操作。原版82秒自动演示脚本 `tools/record_demo.py` 仍可复现，但不含新版完整鼠标点击和逐字输入过程。

## 独立测试和性能复现

```bash
submission/tools/run_probe.sh  # 独立Domain 90；绝不连接Domain89仿真器
submission/tools/ros_env.sh ros2 topic info /image_raw -v
submission/tools/ros_env.sh ros2 topic hz /image_raw
submission/tools/ros_env.sh ros2 topic hz /vision_receive_data
submission/tools/ros_env.sh ros2 topic hz /vision_send_data
submission/tools/ros_env.sh /usr/bin/python3 submission/tools/measure_topics.py /tmp/week2_measurement.json --seconds 20
```

协议测试配置始终 `allow_fire=false`。它会发布测试图像/姿态和 `state=0/1` 的测试输出，仅限Domain90；不启动仿真器，不测试真实发射。
