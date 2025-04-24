"""
工具模块 (modules/utils.py)
-----------------------

提供通用工具类和函数，包含:
- 日志处理
- 人脸工具函数
- 图像处理工具
"""

import os
import time
import cv2
import numpy as np
from datetime import datetime


class Logger:
    """日志处理类，提供统一的日志输出格式和级别控制"""

    def __init__(self, log_to_file=False, log_file="tello_face_recognition.log"):
        """
        初始化日志器

        参数:
            log_to_file: 是否将日志写入文件
            log_file: 日志文件路径
        """
        self.log_to_file = log_to_file
        self.log_file = log_file
        self.log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
        }
        self.current_level = "INFO"  # 默认日志级别

        # 初始化日志文件
        if log_to_file:
            with open(log_file, "w") as f:
                f.write(f"[{self._get_timestamp()}] 日志开始\n")

    def _get_timestamp(self):
        """获取当前时间戳"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _log(self, level, message):
        """
        记录日志

        参数:
            level: 日志级别
            message: 日志消息
        """
        if self.log_levels[level] >= self.log_levels[self.current_level]:
            log_message = f"[{self._get_timestamp()}] {level}: {message}"
            print(log_message)

            if self.log_to_file:
                with open(self.log_file, "a") as f:
                    f.write(log_message + "\n")

    def set_level(self, level):
        """
        设置日志级别

        参数:
            level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        if level in self.log_levels:
            self.current_level = level

    def debug(self, message):
        """记录调试信息"""
        self._log("DEBUG", message)

    def info(self, message):
        """记录一般信息"""
        self._log("INFO", message)

    def warning(self, message):
        """记录警告信息"""
        self._log("WARNING", message)

    def error(self, message):
        """记录错误信息"""
        self._log("ERROR", message)


class FaceUtils:
    """人脸处理工具类，提供人脸相关的实用功能"""

    @staticmethod
    def crop_face(image, face_rect, margin=0.2):
        """
        从图像中裁剪人脸

        参数:
            image: 输入图像
            face_rect: 人脸矩形 (x, y, w, h)
            margin: 边距比例，用于裁剪更大区域

        返回:
            face_image: 裁剪后的人脸图像
        """
        if image is None or len(face_rect) != 4:
            return None

        x, y, w, h = face_rect

        # 计算边距
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # 计算裁剪区域（考虑边界）
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(image.shape[1], x + w + margin_x)
        end_y = min(image.shape[0], y + h + margin_y)

        # 裁剪图像
        face_image = image[start_y:end_y, start_x:end_x]

        return face_image

    @staticmethod
    def align_face(image, landmarks):
        """
        对齐人脸（基于眼睛位置）

        参数:
            image: 输入图像
            landmarks: 人脸关键点，至少需要两眼坐标

        返回:
            aligned_face: 对齐后的人脸图像
        """
        if image is None or landmarks is None:
            return None

        # 获取左眼和右眼中心点
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        # 计算两眼间的角度
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # 计算两眼中心
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)

        # 应用旋转
        height, width = image.shape[:2]
        aligned_face = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC)

        return aligned_face

    @staticmethod
    def enhance_face(image, equalize_hist=True, denoise=True):
        """
        增强人脸图像质量

        参数:
            image: 输入图像
            equalize_hist: 是否进行直方图均衡化
            denoise: 是否进行降噪

        返回:
            enhanced_face: 增强后的人脸图像
        """
        if image is None:
            return None

        # 转为灰度图（如果是彩色图像）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 直方图均衡化
        if equalize_hist:
            # 使用CLAHE进行局部自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # 降噪
        if denoise:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 如果输入是彩色图像，则返回彩色结果
        if len(image.shape) == 3:
            enhanced_face = image.copy()
            # 只改变亮度通道
            hsv = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = gray
            enhanced_face = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return enhanced_face

        return gray

    @staticmethod
    def save_face(image, face_rect, output_dir, filename_prefix="face", max_faces=None):
        """
        保存检测到的人脸

        参数:
            image: 输入图像
            face_rect: 人脸矩形 (x, y, w, h) 或矩形列表
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            max_faces: 最大保存的人脸数量

        返回:
            saved_paths: 保存的文件路径列表
        """
        if image is None:
            return []

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 将单个人脸矩形转换为列表
        if len(np.array(face_rect).shape) == 1:
            face_rects = [face_rect]
        else:
            face_rects = face_rect

        # 限制人脸数量
        if max_faces is not None and len(face_rects) > max_faces:
            face_rects = face_rects[:max_faces]

        saved_paths = []
        timestamp = int(time.time())

        for i, rect in enumerate(face_rects):
            # 裁剪人脸
            face_img = FaceUtils.crop_face(image, rect)

            if face_img is not None:
                # 生成文件名
                filename = f"{filename_prefix}_{timestamp}_{i}.jpg"
                filepath = os.path.join(output_dir, filename)

                # 保存图像
                cv2.imwrite(filepath, face_img)
                saved_paths.append(filepath)

        return saved_paths


class ImageUtils:
    """图像处理工具类，提供通用图像处理功能"""

    @staticmethod
    def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
        """
        调整图像大小，保持宽高比

        参数:
            image: 输入图像
            width: 目标宽度
            height: 目标高度
            inter: 插值方法

        返回:
            resized: 调整大小后的图像
        """
        if image is None:
            return None

        # 获取初始尺寸
        (h, w) = image.shape[:2]

        # 如果宽度和高度都是None，直接返回原图
        if width is None and height is None:
            return image

        # 计算宽高比
        if width is None:
            # 根据高度计算宽度
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            # 根据宽度计算高度
            r = width / float(w)
            dim = (width, int(h * r))

        # 调整大小
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    @staticmethod
    def overlay_transparent(background, overlay, x, y, overlay_size=None):
        """
        在背景图像上叠加透明图像

        参数:
            background: 背景图像
            overlay: 要叠加的图像
            x, y: 叠加位置
            overlay_size: 叠加图像的目标大小 (width, height)

        返回:
            result: 叠加后的图像
        """
        if background is None or overlay is None:
            return background

        # 调整叠加图像大小
        if overlay_size:
            overlay = cv2.resize(overlay, overlay_size)

        h, w = overlay.shape[:2]

        # 确保叠加图像完全在背景内
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + w > background.shape[1]:
            w = background.shape[1] - x
        if y + h > background.shape[0]:
            h = background.shape[0] - y

        # 提取ROI
        if h <= 0 or w <= 0:
            return background

        # 处理叠加部分
        if overlay.shape[2] < 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        overlay_image = overlay[:h, :w]
        mask = overlay_image[:, :, 3:] / 255.0
        background_part = background[y : y + h, x : x + w]

        # 确保背景和叠加尺寸一致
        if background_part.shape[:2] != overlay_image.shape[:2]:
            return background

        # 转换为BGRA（如果不是）
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # 叠加操作
        result = background.copy()
        bg_part = result[y : y + h, x : x + w, :3]
        overlay_part = overlay_image[:, :, :3]

        # 应用alpha混合
        result[y : y + h, x : x + w, :3] = bg_part * (1 - mask) + overlay_part * mask

        return result

    @staticmethod
    def draw_text_with_background(
        image,
        text,
        position,
        font_scale=1.0,
        color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        margin=5,
        alpha=0.5,
    ):
        """
        在图像上绘制带有半透明背景的文本

        参数:
            image: 输入图像
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_scale: 字体大小
            color: 文本颜色
            bg_color: 背景颜色
            thickness: 文本粗细
            font: 字体
            margin: 文本边距
            alpha: 背景透明度 (0-1)

        返回:
            image: 绘制后的图像
        """
        if image is None or not text:
            return image

        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # 计算背景矩形
        x, y = position
        bg_rect = (
            x - margin,
            y - text_height - margin,
            text_width + 2 * margin,
            text_height + baseline + 2 * margin,
        )

        # 创建副本
        result = image.copy()

        # 绘制半透明背景
        sub_img = result[
            bg_rect[1] : bg_rect[1] + bg_rect[3], bg_rect[0] : bg_rect[0] + bg_rect[2]
        ]

        # 创建背景色矩形
        color_rect = np.ones(sub_img.shape, dtype=np.uint8)
        color_rect[:] = bg_color

        # 混合背景和图像
        cv2.addWeighted(color_rect, alpha, sub_img, 1 - alpha, 0, sub_img)

        # 绘制文本
        cv2.putText(result, text, position, font, font_scale, color, thickness)

        return result
