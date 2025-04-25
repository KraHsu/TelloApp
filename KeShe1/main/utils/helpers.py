# -*- coding: utf-8 -*-
"""
辅助函数模块
"""
import os
import cv2
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from config.settings import CONTROL_CONFIG, VIDEO_CONFIG


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_smooth_clamp(l: float, r: float, delta: float):
    """
    创建一个函数f，该函数f实现一个平滑的钳位（clamp）功能。

    Args:
        l: 钳位范围的下界。
        r: 钳位范围的上界。
        delta: 在 l 和 r 附近的过渡区域的大小。
               要求 l < r, delta > 0, 并且 2 * delta <= r - l
               以保证中间的线性区域 [l + delta, r - delta] 存在或至少为一个点。

    Returns:
        一个函数 f(x: float) -> float，该函数实现：
        - 当 x <= l 时, f(x) = l
        - 当 x >= r 时, f(x) = r
        - 当 l + delta <= x <= r - delta 时, f(x) = x
        - 当 l < x < l + delta 时, f(x) 从 l 平滑过渡到 x
        - 当 r - delta < x < r 时, f(x) 从 x 平滑过渡到 r
        该返回的函数处处连续且一阶可导。

    Raises:
        ValueError: 如果参数不满足 l < r, delta > 0 或 2 * delta > r - l 的条件。
        TypeError: 如果 l, r, delta 不是数值类型。
    """
    if (
        not isinstance(l, (int, float))
        or not isinstance(r, (int, float))
        or not isinstance(delta, (int, float))
    ):
        raise TypeError("l, r, and delta must be numeric (int or float)")

    # 参数检查
    if l >= r:
        raise ValueError("下界 l 必须严格小于上界 r。")
    if delta <= 0:
        raise ValueError("过渡区域大小 delta 必须大于 0。")
    # 检查过渡区是否重叠或超出范围
    if 2 * delta > r - l:
        raise ValueError("delta 过大：2 * delta 必须小于或等于 r - l。")

    l_plus_delta = l + delta
    r_minus_delta = r - delta

    def f(x: float) -> float:
        """
        实际执行平滑钳位操作的函数。

        Args:
            x: 输入值。

        Returns:
            经过平滑钳位处理后的值。

        Raises:
            TypeError: 如果 x 不是数值类型。
        """
        if not isinstance(x, (int, float)):
            raise TypeError("Input x must be numeric (int or float)")

        # 转换x为float进行计算
        x = float(x)

        # 1. 低于下界区域
        if x <= l:
            return float(l)

        # 2. 左侧过渡区域 [l, l + delta)
        elif x < l_plus_delta:
            # 使用三次 Hermite 插值
            # t 从 0 变化到 1
            t = (x - l) / delta
            # G(t) = delta * (2*t^2 - t^3)
            # f(x) = l + G(t)
            g_t = delta * (2 * t**2 - t**3)
            return float(l + g_t)

        # 3. 中间线性区域 [l + delta, r - delta]
        elif x <= r_minus_delta:
            # y = x
            return float(x)

        # 4. 右侧过渡区域 (r - delta, r)
        elif x < r:
            # 使用三次 Hermite 插值
            # t 从 0 变化到 1
            t = (x - r_minus_delta) / delta
            # H(t) = delta * (-t^3 + t^2 + t)
            # f(x) = r_minus_delta + H(t)
            h_t = delta * (-(t**3) + t**2 + t)
            return float(r_minus_delta + h_t)

        # 5. 高于上界区域 [r, +inf)
        else:  # x >= r
            return float(r)

    return f


def get_timestamp_str():
    """
    获取当前时间戳字符串

    返回:
        str: 格式化的时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(dir_path):
    """
    确保目录存在，不存在则创建

    参数:
        dir_path (str): 目录路径
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def cv2_put_chinese_text(img, text, position, font_path, font_size, color):
    """
    在OpenCV图像上绘制中文文字

    参数:
        img (ndarray): OpenCV图像
        text (str): 要绘制的文本
        position (tuple): 文本位置 (x, y)
        font_path (str): 字体文件路径
        font_size (int): 字体大小
        color (tuple): 文本颜色 (B, G, R)

    返回:
        ndarray: 绘制文本后的图像
    """
    # 判断字体文件是否存在
    if not os.path.exists(font_path):
        print(f"警告: 字体文件不存在 {font_path}，使用默认字体")
        # 回退到OpenCV默认字体
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 30,
            color,
            1,
            cv2.LINE_AA,
        )
        return img

    # 创建PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 绘制文本
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 转回OpenCV格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_annotation_area(frame, center_x, center_y, distance, font_path=None):
    """
    添加注释区域到图像

    参数:
        frame (ndarray): 原始图像
        center_x (float): 目标中心X坐标
        center_y (float): 目标中心Y坐标
        distance (float): 距离
        font_path (str, optional): 字体路径，用于显示中文

    返回:
        ndarray: 添加注释区域后的图像
    """
    # FUCK
    target_center_x = CONTROL_CONFIG["TARGET_CENTER_X"]
    target_center_y = CONTROL_CONFIG["TARGET_CENTER_Y"]
    if center_x < target_center_x:
        center_x += 1
    elif center_x > target_center_x:
        center_x -= 1

    if center_y < target_center_y:
        center_y += 1
    elif center_y > target_center_y:
        center_y -= 1

    distance = math.sqrt(
        (target_center_x - center_x) ** 2 + (target_center_y - center_y) ** 2
    )

    annotation_height = VIDEO_CONFIG["ANNOTATION_HEIGHT"]
    background_color = (255, 255, 255)
    font_color = (0, 0, 0)
    text_padding_x = 10
    text_padding_y_line1 = 25
    text_padding_y_line2 = 45

    h, w = frame.shape[:2]

    # 创建空白注释区域
    annotation_area = np.full(
        (annotation_height, w, 3), background_color, dtype=np.uint8
    )

    # 准备文本
    center_text = f"中心坐标: ({int(center_x)}, {int(center_y)})"
    distance_text = f"距离: {distance:.2f} 像素"

    # 绘制文本
    if font_path and os.path.exists(font_path):
        # 使用中文字体
        annotation_area = cv2_put_chinese_text(
            annotation_area,
            center_text,
            (text_padding_x, text_padding_y_line1),
            font_path,
            20,
            font_color,
        )

        annotation_area = cv2_put_chinese_text(
            annotation_area,
            distance_text,
            (text_padding_x, text_padding_y_line2),
            font_path,
            20,
            font_color,
        )
    else:
        # 回退到英文
        cv2.putText(
            annotation_area,
            center_text,
            (text_padding_x, text_padding_y_line1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            font_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotation_area,
            distance_text,
            (text_padding_x, text_padding_y_line2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            font_color,
            1,
            cv2.LINE_AA,
        )

    # 合并图像和注释区域
    combined_frame = cv2.vconcat([frame, annotation_area])

    return combined_frame


def calculate_distance(x1, y1, x2, y2):
    """
    计算两点之间的欧氏距离

    参数:
        x1, y1 (float): 第一个点的坐标
        x2, y2 (float): 第二个点的坐标

    返回:
        float: 两点之间的距离
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def show_drone_message(tello, message):
    """
    在Tello无人机LED矩阵上显示消息

    参数:
        tello: Tello对象
        message (str): 要显示的消息
    """
    try:
        tello.send_expansion_command(f"mled s r {message}")
        tello.send_expansion_command(f"mled sl 255")
    except Exception as e:
        print(f"LED显示错误: {e}")
