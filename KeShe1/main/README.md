# Tello智能目标追踪系统

## 项目结构
```
tello_tracking/
├── config/
│   └── settings.py       # 配置文件
├── core/
│   ├── __init__.py
│   ├── detector.py       # 目标检测模块
│   ├── controller.py     # 飞行控制模块  
│   ├── mission.py        # 任务执行模块
│   └── video.py          # 视频处理模块
├── utils/
│   ├── __init__.py
│   ├── data_collector.py # 数据收集器
│   ├── pid_controller.py # PID控制器
│   └── helpers.py        # 辅助函数
├── resources/
│   └── fonts/            # 中文字体文件
├── main.py               # 主程序入口
└── requirements.txt      # 依赖项
```

## 模块说明

1. **配置模块 (config/)**
   - 集中管理所有配置参数

2. **核心模块 (core/)**
   - detector.py: 处理YOLOv8目标检测
   - controller.py: 飞行控制逻辑
   - mission.py: 任务规划和执行
   - video.py: 视频处理和显示

3. **工具模块 (utils/)**
   - data_collector.py: 数据收集和保存
   - pid_controller.py: PID控制器实现
   - helpers.py: 通用辅助函数

4. **资源 (resources/)**
   - 存放字体等资源文件