# -*- coding: utf-8 -*-
"""
配置文件：存储所有配置参数
"""
import os
import torch
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 模型配置 =====
MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(
        ROOT_DIR, "resources/models/best.pt"
    ),  # YOLOv8 模型文件路径
    "TARGET_CLASSES": ["bit", "drone", "card"],  # 需要检测并获取坐标的目标类别
    "CONFIDENCE_THRESHOLD": 0.8,  # 置信度阈值
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# ===== 视频处理配置 =====
VIDEO_CONFIG = {
    "CROP_WIDTH": 320,
    "CROP_HEIGHT": 240,
    "SAVE_DIR": os.path.join(ROOT_DIR, "recordings"),  # 视频保存目录
    "FONT_PATH": os.path.join(
        ROOT_DIR, "resources", "fonts", "SimHei.ttf"
    ),  # 中文字体路径
    "ANNOTATION_HEIGHT": 120,  # 注释区域高度
}

# ===== 控制配置 =====
CONTROL_CONFIG = {
    "TARGET_HEIGHT": 100,  # 目标飞行高度 (cm)
    "TARGET_CENTER_X": VIDEO_CONFIG["CROP_HEIGHT"] // 2,  # 目标中心X坐标
    "TARGET_CENTER_Y": VIDEO_CONFIG["CROP_WIDTH"] // 2,  # 目标中心Y坐标
    "MAX_SPEED": 100,  # 最大速度值
    "MIN_TARGET_DISTANCE": 4,  # 视为到达目标的最小距离
}

# ===== PID控制器参数 =====
PID_CONFIG = {
    "NORMAL": {
        "P": 0.12,
        "I": 0.07,
        "D": 0,
        "LIMIT": (-100, 100),
        "OUTPUT_LIMIT": (-7, 7),
    },
    "SLOW": {
        "P": 0.1,
        "I": 0.07,
        "D": 0,
        "LIMIT": (-100, 100),
        "OUTPUT_LIMIT": (-5, 5),
    },
    "HEIGHT": {
        "P": 1.2,
        "I": 0.1,
        "D": 0,
        "LIMIT": (-30, 30),
        "OUTPUT_LIMIT": (-10, 10),
    },
}

# ===== 任务规划配置 =====
MISSION_CONFIG = {
    "TARGET_SEQUENCE": ["card", "bit", "drone"],  # 目标检测顺序
    "MISSION_MOVES": [
        {"direction": "F", "duration": 5},  # 前进
        {"direction": "R", "duration": 5},  # 右移
        {"direction": "B", "duration": 5},  # 后退
    ],
    "RETURN_HOME": [
        {"command": "move_left", "distance": 100},
        {"command": "move_left", "distance": 100},
        {"command": "move_left", "distance": 100},
    ],
}

# ===== 数据收集配置 =====
DATA_CONFIG = {
    "DATA_FILE": os.path.join(ROOT_DIR, "mid_data.csv"),
}
