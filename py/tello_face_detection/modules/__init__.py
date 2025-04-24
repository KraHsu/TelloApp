"""
Tello无人机人脸识别系统
====================

这个模块化项目实现了基于DJI Tello无人机的实时人脸检测与识别功能。
项目设计用于《2025智能无人系统综合实践II》课程的技术分享。

模块结构:
-------
- config.py: 配置参数管理
- detectors.py: 人脸检测器实现
- video_handler.py: 视频处理
- utils.py: 通用工具函数
- ui.py: 用户界面组件

作者: [您的姓名]
版本: 1.0.0
日期: 2025-04-02
"""

# 版本号
__version__ = "1.0.0"

# 导出主要组件供直接使用
from .config import Config
from .detectors import FaceDetectorFactory
from .video_handler import VideoHandler
from .ui import UserInterface
from .utils import Logger, FaceUtils, ImageUtils
