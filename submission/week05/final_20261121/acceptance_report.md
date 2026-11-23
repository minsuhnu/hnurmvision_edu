# 第五周验收记录

2026-11-21，项目分支 `feature/minsuhnu-agent-practice`，HEAD `1aa9b7e2b200d4483622e5d67b53300a2b393d30`。

**判断：达到第五讲官方提交要求。当前教学场景表现较好；没有其他方案的同场数据，不能认定竞争排名或泛化性能。**

## 官方要求与证据

第五讲要求“提交自瞄程序在仿真下击打静止靶、平移靶的录屏”，格式 MP4；没有最低命中率、DPS 或连续命中次数门槛。两段最终视频均直接录制实际 X11 仿真窗口，未剪辑。截图、约200字概要按本次委托书的周提交要求补齐；本地未找到培训总览 PPT 原件，未声称独立核验该 PPT。

| 场景 | 计分时间 | 发射 | 命中 | 未命中 | 命中率 | 教学DPS |
|---|---:|---:|---:|---:|---:|---:|
| 静止靶 | 30.200s | 273 | 273 | 0 | 100.0% | 90.40 |
| 平移靶 | 30.216s | 244 | 227 | 17 | 93.03% | 75.13 |

统计取自同次运行的 `static_target_scores.json` 和 `moving_target_scores.json`。单发伤害10仅为仿真教学设置。按场景计分面板统计，不使用画面底部包含预热/暂停时射击的全局计数；命中由原仿真器物理碰撞判定，原始轨迹保留在 `hits.jsonl`。两个进程均正常退出，见 `run.json`。

视频：`static_target.mp4` 30.95s、`moving_target.mp4` 30.85s，均1600×900、20fps、H.264/avc1。分别619帧和617帧，全部解码通过，见 `verification.json`。`*_video_check.png` 为视频第5秒的抽查帧；`*_detection.png` 为实际检测窗口截图。检测窗口原有250ms显示属于第四讲预测诊断，不代表第五讲实际使用固定250ms提前量。实际提前量随图像延迟与弹道飞行时间变化，双向样本见 `prediction_evidence.json`。

## 本轮修复与优化

- `lessons/lesson05_aiming.cpp`、`include/ballistics.hpp`：独立弹道接口、水平距离、低伸解、数值边界、1ms不动点收敛与30次上限；仿真发弹延迟为0，不加入云台滞后；yaw绝对角、pitch=25°−theta。
- `lessons/lesson02_io.cpp`、`io/io_context.hpp`、`include/pose_history.hpp`：反馈按时间保存，图像时刻四元数插值；边界只接受30ms内最近样本，插值段超过100ms拒绝。避免将相机转动误算成靶速。
- `include/fire_control.hpp`、IO配置：实测角对比上一条已发布控制指令，并要求当前命令无突跳，正确处理环绕；0.01rad容差可配置，是工程选择。失效请求清除上一指令。
- `src/realtime_processor.cpp`：只对当前观测且估速有效的目标解算；增加按需遥测日志。
- `configs/lesson05_tracking.yaml`：第五讲使用80ms估速窗口，原第四讲配置保留。
- `scripts/run_lesson05_acceptance.sh`、`submission/week05/run_acceptance.py`：启动既有仿真器，使用其正常F6/F7/F9界面切换、计分、导出并录制；不改变命中规则。

上一轮已补的预测距离用于IO发送，避免 -1 被仿真器解释为停止。代码保留在原源码位置，没有复制源码交付包。未修改受保护的 `src/main.cpp`，未提交或推送；原有其他工作区修改及旧材料保留。

200ms窗口调试轮：平移118/134（88.06%）；80ms窗口调试轮：130/136（95.59%）；最终录制轮：227/244（93.03%）。这三轮并非相同图像输入的严格消融，不能将差异全部归因于单一参数。保留两次调试轮于 `../review_20261121/`。

## 测试与限制

完整构建成功，三个CTest可执行均通过：第四讲PnP、第四讲跟踪、第五讲弹道/预测/火控。第五讲覆盖轨迹残差、高低目标、不同距离、最大射程边界、无解、异常输入、处理时延与双向运动、慢收敛拒绝、姿态插值和角度环绕。构建与测试日志在 `../review_20261121/`。

最终静止/平移窗口中，处理耗时中位数13.29/13.90ms，p95为15.82/16.64ms；观测至处理结束的年龄中位数43.57/45.56ms。这是检测、PnP、跟踪与瞄准合计，含日志环境，不是单独弹道函数耗时。

本次条件：观察位至靶车约3m，前板距离约2.6m；平移幅值0.5m、周期4s；弹速25m/s、g=9.81；噪声sigma=2，云台限速2.5rad/s，原 swept 碰撞模式。采用讲义忽略枪口与云台中心偏移的假设，未加入未经验证的偏移常量。验证范围为当前直立小板场景；远距离、遮挡、多目标、大板、倾斜车体及实机未充分验证。未以目标真值驱动控制；算法只接收图像与云台反馈。录制中平移仍有17次未命中，不描述为全场景高命中率。

## 复现

在项目根目录，使用已有依赖：

```bash
source /opt/ros/humble/setup.bash
source simulation/install_interfaces/setup.bash
cmake -S . -B build-week05 -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LESSON04_PNP_TEST=ON -DBUILD_LESSON05_TEST=ON \
  -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build build-week05 -j2
ctest --test-dir build-week05 --output-on-failure
bash scripts/run_lesson05_acceptance.sh submission/week05/new_run \
  --seconds 30 --record --tracking-config configs/lesson05_tracking.yaml
```

`new_run`必须不存在。入口启动的是现有 `simulation/simulator/target/release/daedalus` 与 `build-week05/training`；复制仿真配置到新目录，生成允许仿真开火的独立IO配置。X11/GStreamer为本机现有工具，已获本次用户明确授权。运行期间F8清零，静止F7计时，暂停后F6切平移，再F7计时，F9导出；最终正常关闭。原项目默认IO配置仍关闭开火。
