"""
用户界面模块 (modules/ui.py)
-----------------------

负责处理系统的用户界面展示，包括：
- 视频帧的标注和叠加信息
- 操作指引的显示
- 检测结果的可视化
"""

import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont


class UserInterface:
    """用户界面类，处理所有UI相关功能"""

    def __init__(self):
        """初始化UI组件"""
        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.text_color = (0, 255, 0)  # 绿色文本
        self.text_thickness = 2

        # 框颜色设置
        self.box_color = (0, 255, 0)  # 绿色框
        self.box_thickness = 2

        # UI状态
        self.show_help = True
        self.show_fps = True
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # 添加中文字体支持
        self.chinese_font_path = "font/SMILEYSANS-OBLIQUE.TTF"  # 字体文件路径
        self.chinese_font_sizes = {"small": 18, "medium": 24, "large": 30}

    def draw_results(
        self,
        frame,
        faces,
        names,
        method,
        enable_recognition,
        processing,
        reused_result,
        results_frame_count,
    ):
        """
        在帧上绘制检测结果和状态信息

        参数:
            frame: 输入视频帧
            faces: 检测到的人脸列表 [(x,y,w,h),...]
            names: 人脸对应的名称列表
            method: 当前使用的检测方法
            enable_recognition: 是否启用了人脸识别
            processing: 当前是否正在处理中
            reused_result: 是否使用了缓存结果
            results_frame_count: 当前结果已使用的帧数

        返回:
            result_frame: 绘制结果后的帧
        """
        # 创建帧副本以避免修改原始帧
        result = frame.copy()

        # 更新FPS计算
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_frame_time > 1.0:  # 每秒更新一次
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time

        # 在画面上显示人脸框和名称
        for i, (x, y, w, h) in enumerate(faces):
            # 如果有名字，显示名字
            if i < len(names) and names[i]:
                self._add_chinese_text(
                    result, names[i], (x, y - 20), "small", self.text_color
                )
                cv2.rectangle(
                    result, (x, y), (x + w, y + h), self.box_color, self.box_thickness
                )
            else:
                self._add_chinese_text(
                    result, "未知人员", (x, y - 20), "small", self.text_color
                )
                cv2.rectangle(
                    result, (x, y), (x + w, y + h), (255, 0, 0), self.box_thickness
                )

        # 添加状态信息 - 使用中文
        self._add_status_info(result, f"检测方法: {method}", 10, 30)

        # 显示识别模式 - 使用中文
        if method == "face_recognition" and enable_recognition:
            self._add_status_info(result, "模式: 特定人员识别", 10, 60)
            self._add_status_info(result, f"已识别 {len(faces)} 个已知人脸", 10, 90)
        else:
            self._add_status_info(result, "模式: 全部人脸检测", 10, 60)
            self._add_status_info(result, f"检测到 {len(faces)} 个人脸", 10, 90)

        # 显示处理状态
        processing_status = "处理中..." if processing else "就绪"
        self._add_status_info(result, f"状态: {processing_status}", 10, 120)

        # 显示缓存状态
        if reused_result:
            self._add_status_info(
                result,
                f"使用缓存结果 (帧 {results_frame_count})",
                10,
                150,
                color=(0, 255, 255),  # 黄色警告
            )

        # 显示FPS
        if self.show_fps:
            self._add_status_info(
                result, f"FPS: {self.fps:.1f}", 10, result.shape[0] - 100
            )

        # 显示操作指引
        if self.show_help:
            self._add_help_text(result)

        return result

    def _add_chinese_text(self, frame, text, position, size="small", color=None):
        """添加中文文本到图像"""
        if color is None:
            color = self.text_color

        font_size = self.chinese_font_sizes[size]

        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)

        # 加载中文字体
        try:
            font = ImageFont.truetype(self.chinese_font_path, font_size)
        except IOError:
            # 如果找不到指定字体，尝试使用系统默认字体
            try:
                # Linux系统默认中文字体
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                    font_size,
                )
            except IOError:
                try:
                    # Windows系统默认中文字体
                    font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
                except IOError:
                    # 如果仍然失败，使用默认字体
                    font = ImageFont.load_default()

        # 在PIL图像上绘制中文
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

        # 将PIL图像转换回OpenCV格式并直接修改原始帧
        cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR, frame)

    def _add_status_info(self, frame, text, x, y, color=None, use_chinese=True):
        """
        添加状态信息文本

        参数:
            frame: 输入帧
            text: 文本内容
            x, y: 文本位置
            color: 文本颜色，默认使用self.text_color
            use_chinese: 是否使用中文渲染（支持中文字符）
        """
        if color is None:
            color = self.text_color

        if use_chinese and any("\u4e00" <= char <= "\u9fff" for char in text):
            # 如果启用中文支持且文本包含中文字符，使用中文渲染
            self._add_chinese_text(frame, text, (x, y), "small", color)
        else:
            # 否则使用OpenCV原生方法
            cv2.putText(
                frame,
                text,
                (x, y),
                self.font,
                self.font_scale,
                color,
                self.text_thickness,
            )

    def _add_help_text(self, frame):
        """
        添加操作帮助文本

        参数:
            frame: 输入帧
        """
        # 底部半透明黑条
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 添加操作指引文本
        white_color = (255, 255, 255)
        self._add_status_info(
            frame, "空格键: 切换检测方法", 10, h - 90, color=white_color
        )
        self._add_status_info(frame, "R键: 切换识别模式", 10, h - 60, color=white_color)
        self._add_status_info(frame, "ESC键: 退出程序", 10, h - 30, color=white_color)

    def create_statistics_frame(self, stats):
        """
        创建统计信息帧

        参数:
            stats: 统计信息字典

        返回:
            stats_frame: 统计信息帧
        """
        # 创建统计信息画布
        stats_frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # 添加标题
        title = "检测统计信息"
        cv2.putText(
            stats_frame,
            title,
            (20, 40),
            self.font,
            1.2,
            (255, 255, 255),
            2,
        )

        # 添加统计信息
        y_pos = 100
        for key, value in stats.items():
            info_text = f"{key}: {value}"
            cv2.putText(
                stats_frame,
                info_text,
                (30, y_pos),
                self.font,
                0.8,
                (200, 200, 200),
                2,
            )
            y_pos += 40

        return stats_frame

    def toggle_help(self):
        """切换帮助信息显示状态"""
        self.show_help = not self.show_help

    def toggle_fps(self):
        """切换FPS显示状态"""
        self.show_fps = not self.show_fps
