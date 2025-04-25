# -*- coding: utf-8 -*-
"""
目标检测模块
"""
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config.settings import MODEL_CONFIG, VIDEO_CONFIG, CONTROL_CONFIG


class ObjectDetector:
    """
    基于YOLOv8的目标检测器

    参数:
        model_path (str): YOLO模型文件路径
        confidence_threshold (float): 置信度阈值
        target_classes (list): 目标类别列表
        device (torch.device): 计算设备
    """

    def __init__(
        self,
        model_path=MODEL_CONFIG["MODEL_PATH"],
        confidence_threshold=MODEL_CONFIG["CONFIDENCE_THRESHOLD"],
        target_classes=MODEL_CONFIG["TARGET_CLASSES"],
        device=MODEL_CONFIG["DEVICE"],
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.device = device
        self.model = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        """
        加载YOLO模型
        """
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"成功加载模型: {self.model_path}")

            # 检查模型类别名称是否包含目标类别
            model_classes = self.model.names
            print(f"模型包含的类别: {list(model_classes.values())}")

            for target in self.target_classes:
                if target not in model_classes.values():
                    print(f"警告: 目标类别 '{target}' 可能不在模型类别中!")

        except Exception as e:
            print(f"加载YOLO模型时出错: {e}")
            raise

    def detect(self, frame, target_type=None):
        """
        在图像中检测目标

        参数:
            frame (ndarray): 输入图像
            target_type (str, optional): 特定目标类型，如果为None则检测所有目标类别

        返回:
            tuple: (处理后的图像, 检测结果字典)
                结果字典包含：
                - center_x, center_y: 目标中心坐标
                - confidence: 置信度
                - bbox: 边界框 (x1, y1, x2, y2)
                - distance: 到图像中心的距离
                - class_name: 类别名称
        """
        if self.model is None:
            raise ValueError("模型未加载")

        # 检测对象
        results = self.model(
            frame, stream=False, verbose=False, conf=self.confidence_threshold
        )

        # 初始化结果
        detection_result = {
            "center_x": None,
            "center_y": None,
            "confidence": 0,
            "bbox": None,
            "distance": -1,
            "class_name": None,
            "detected": False,
        }

        # 处理检测结果
        if results and len(results) > 0:
            boxes = results[0].boxes  # 获取Boxes对象

            # 定义图像中心点
            img_center_x = CONTROL_CONFIG["TARGET_CENTER_X"]
            img_center_y = CONTROL_CONFIG["TARGET_CENTER_Y"]

            # 遍历每个检测到的边界框
            for box in boxes:
                # 提取置信度
                confidence = float(box.conf[0])

                # 提取类别ID
                cls_id = int(box.cls[0])

                # 获取类别名称
                class_name = self.model.names[cls_id]

                # 如果指定了目标类型且不匹配，则跳过
                if target_type and class_name != target_type:
                    continue

                # 提取边界框坐标 (xyxy 格式)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 计算中心点坐标
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # FUCK BEGIN
                for i in range(2):
                    if center_x < img_center_x:
                        center_x += 1
                        x1 += 1
                        x2 += 1
                    elif center_x > img_center_x:
                        center_x -= 1
                        x1 -= 1
                        x2 -= 1
                    if center_y < img_center_y:
                        center_y += 1
                        y1 += 1
                        y2 += 1
                    elif center_y > img_center_y:
                        center_y -= 1
                        y1 -= 1
                        y2 -= 1
                # FUCK END

                # 计算到图像中心的距离
                dx = img_center_x - center_x
                dy = img_center_y - center_y
                distance = np.sqrt(dx**2 + dy**2)

                # 在图像上绘制检测结果
                frame = self._draw_detection(
                    frame, x1, y1, x2, y2, center_x, center_y, class_name, confidence
                )

                # 更新结果
                detection_result.update(
                    {
                        "center_x": center_x,
                        "center_y": center_y,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "distance": distance,
                        "class_name": class_name,
                        "detected": True,
                    }
                )

                # 只处理第一个符合条件的目标
                break

        # 添加中心十字线
        cv2.circle(
            frame,
            (CONTROL_CONFIG["TARGET_CENTER_X"], CONTROL_CONFIG["TARGET_CENTER_Y"]),
            3,
            (255, 0, 0),
            -1,
        )

        return frame, detection_result

    def _draw_detection(
        self, frame, x1, y1, x2, y2, center_x, center_y, class_name, confidence
    ):
        """
        在图像上绘制检测结果

        参数:
            frame (ndarray): 输入图像
            x1, y1, x2, y2 (int): 边界框坐标
            center_x, center_y (int): 中心点坐标
            class_name (str): 类别名称
            confidence (float): 置信度

        返回:
            ndarray: 绘制后的图像
        """
        # 绘制边界框 (绿色)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 准备标签文字
        label = f"{class_name}: {confidence:.2f}"

        # 绘制标签背景
        (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - h_label - 5),
            (x1 + w_label, y1),
            (0, 255, 0),
            -1,
        )

        # 绘制标签文字 (白色)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # 绘制中心点 (红色)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        return frame
