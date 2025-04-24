import time


class PIDController:
    """A simple Proportional-Integral-Derivative (PID) controller."""

    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-100, 100)):
        """
        Initialize the PID controller.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): The target value the controller tries to achieve.
            output_limits (tuple): A tuple (min_output, max_output) to clamp the controller output.
                                   Defaults to Tello's typical RC range (-100, 100).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_error = 0

        self.reset()  # Initialize properly

    def update(self, current_value):
        """
        Calculate the PID output based on the current value.

        Args:
            current_value (float): The current measured value from the system.

        Returns:
            float: The calculated control output, clamped within the output limits.
        """
        current_time = time.monotonic()  # Use monotonic clock for time differences
        try:
            dt = (
                current_time - self._last_time
                if self._last_time is not None
                else 1.0 / 30.0
            )  # Estimate dt for first run
        except TypeError:  # Handle potential None subtraction if reset wasn't perfect
            dt = 1.0 / 30.0  # Default to ~30fps timeframe if last_time is invalid

        if dt <= 0:  # Avoid division by zero or negative time travel
            # Reuse last output or return 0 if no output yet
            return self.clamp(self._proportional + self._integral + self._derivative)

        error = self.setpoint - current_value

        # Proportional term
        self._proportional = self.Kp * error

        # Integral term (with anti-windup via clamping)
        self._integral += self.Ki * error * dt
        self._integral = self.clamp(
            self._integral, self.output_limits
        )  # Basic anti-windup

        # Derivative term
        if self._last_time is not None:
            delta_error = error - self._last_error
            self._derivative = self.Kd * (delta_error / dt)
        else:
            self._derivative = 0  # No derivative on the first run

        # Calculate total output
        output = self._proportional + self._integral + self._derivative

        # Clamp output to limits
        output = self.clamp(output)

        # Update state for next iteration
        self._last_error = error
        self._last_time = current_time

        return output

    def reset(self):
        """Reset the PID controller's integral sum and timing."""
        self._proportional = 0
        self._integral = 0
        self._derivative = 0
        self._last_error = 0
        self._last_time = None  # Reset time to force dt estimation on next update

    def set_setpoint(self, new_setpoint):
        """Update the target setpoint."""
        self.setpoint = new_setpoint
        self.reset()  # Reset controller state when setpoint changes significantly

    def set_tunings(self, Kp, Ki, Kd):
        """Update the PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def clamp(self, value, limits=None):
        """Clamp the value within the specified limits."""
        if limits is None:
            limits = self.output_limits
        lower, upper = limits
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value


# =============================================
# Placeholder Tello Control Functions
# =============================================
# You need to implement these using tello.send_rc_control(lr, fb, ud, yaw)


def tello_send_rc_command(left_right_speed, fwd_bwd_speed, up_down_speed, yaw_speed):
    """
    Placeholder function to send RC control commands to Tello.
    IMPLEMENT THIS using tello.send_rc_control.
    Speeds should be integers between -100 and 100.
    """
    lr = int(left_right_speed)
    fb = int(fwd_bwd_speed)
    ud = int(up_down_speed)
    yw = int(yaw_speed)
    # Clamp values just in case PID output was slightly outside after clamping
    lr = max(-100, min(100, lr))
    fb = max(-100, min(100, fb))
    ud = max(-100, min(100, ud))
    yw = max(-100, min(100, yw))

    print(f"[SIM] Sending RC -> LR: {lr}, FB: {fb}, UD: {ud}, YAW: {yw}")
    # --- YOUR ACTUAL TELLO CODE HERE ---
    # Example:
    # global tello # Make sure tello object is accessible here
    # tello.send_rc_control(lr, fb, ud, yw)
    # ---------------------------------
    pass  # Remove pass when implementing


def tello_stop_movement():
    """Placeholder function to stop all Tello movement."""
    print("[SIM] Sending RC -> Stop all movement")
    tello_send_rc_command(0, 0, 0, 0)


import threading
import cv2
from djitellopy import Tello
import time
import os
import numpy as np
from ultralytics import YOLO  # 导入YOLO
import math  # 导入 math 以便后续可能需要

# --- 配置 ---
MODEL_PATH = "best.pt"  # YOLOv8 模型文件路径
TARGET_CLASSES = ["bit", "drone", "card"]  # 需要检测并获取坐标的目标类别
CONFIDENCE_THRESHOLD = 0.8  # 置信度阈值 (与原代码一致)
# --- Tello 设置 ---
CROP_WIDTH = 320
CROP_HEIGHT = 240
TARGET_CENTER_X = CROP_WIDTH // 2
TARGET_CENTER_Y = CROP_HEIGHT // 2
# --- 加载模型 ---
try:
    model = YOLO(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
    # 检查模型类别名称是否包含目标类别 (可选但推荐)
    model_classes = model.names
    print(f"模型包含的类别: {list(model_classes.values())}")
    for target in TARGET_CLASSES:
        if target not in model_classes.values():
            print(f"警告: 目标类别 '{target}' 可能不在模型类别中!")
except Exception as e:
    print(f"加载 YOLO 模型时出错: {e}")
    exit()

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print("Tello 电量: ", tello.get_battery(), "%")

    # --- 启动视频流 ---
    tello.streamon()
    # 确保获取 frame_read 对象在 streamon 之后
    frame_read = tello.get_frame_read()

    # --- 设置 Tello 参数 ---
    # 注意：向下摄像头可能不支持所有分辨率/FPS设置，720p可能不是向下摄像头的原生分辨率
    tello.set_video_direction(tello.CAMERA_DOWNWARD)  # 如果需要向下摄像头，取消注释
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_bitrate(tello.BITRATE_5MBPS)  # 可以根据网络情况调整
    tello.set_video_resolution(tello.RESOLUTION_720P)

    print("等待视频流稳定...")
    time.sleep(2)  # 等待视频流建立

    print("开始检测... 按 ESC 键退出。")

    pid_vx = PIDController(0.01, 0, 0, TARGET_CENTER_X, (-100, 100))
    pid_vy = PIDController(0.01, 0, 0, TARGET_CENTER_Y, (-100, 100))

    while True:
        # --- 获取帧 ---
        frame = frame_read.frame
        if frame is None:
            print("未能获取到帧，可能连接中断。")
            time.sleep(0.5)  # 短暂等待后重试
            continue

        # --- 裁剪帧 ---
        # 确保裁剪不会超出图像边界 (虽然 [:240] 通常是安全的)
        h, w, _ = frame.shape
        actual_crop_height = min(CROP_HEIGHT, h)  # 防止裁剪高度超过实际高度
        cropped_frame = frame[
            :actual_crop_height, :, :
        ].copy()  # 裁剪帧并创建副本以进行修改
        cropped_frame = cv2.flip(cv2.transpose(cropped_frame), 1)

        # --- 使用YOLOv8进行检测 (在裁剪后的帧上) ---
        results = model(
            cropped_frame, stream=False, verbose=False, conf=CONFIDENCE_THRESHOLD
        )

        # --- 处理检测结果 ---
        # results 是一个列表，通常只包含一个结果对象 (对应输入的单张图片)
        if results and results[0]:
            boxes = results[0].boxes  # 获取Boxes对象

            # 遍历每个检测到的边界框
            for box in boxes:
                # 提取置信度
                confidence = box.conf[0]
                # 提取类别ID
                cls_id = int(box.cls[0])
                # 获取类别名称
                class_name = model.names[cls_id]

                # 检查是否是我们关心的目标类别
                if class_name in TARGET_CLASSES:
                    # 提取边界框坐标 (xyxy 格式)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # --- 计算中心点坐标 ---
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # --- 计算pid --
                    vx = pid_vx.update(center_y)
                    vy = pid_vy.update(center_x)

                    # --- 在裁剪后的帧上绘制 ---
                    # 绘制边界框 (绿色)
                    cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 准备标签文字 (英文)
                    label = f"{class_name}: {confidence:.2f}"
                    # 绘制标签背景
                    (w_label, h_label), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        cropped_frame,
                        (x1, y1 - h_label - 5),
                        (x1 + w_label, y1),
                        (0, 255, 0),
                        -1,
                    )  # 填充背景
                    # 绘制标签文字 (白色)
                    cv2.putText(
                        cropped_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    # 绘制中心点 (红色)
                    cv2.circle(cropped_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # --- 输出中心坐标 (中文) ---
                    # print(
                    #     f"检测到目标: {class_name} | 中心坐标: ({center_x}, {center_y})"
                    # )
                    print(f"vx: {vx}, vy: {vy}")
                cv2.circle(
                    cropped_frame,
                    (TARGET_CENTER_Y, TARGET_CENTER_X),
                    5,
                    (0, 0, 255),
                    -1,
                )

        # --- 显示处理后的帧 ---
        cv2.imshow("Tello Cropped Detection", cropped_frame)  # 显示带检测结果的裁剪帧
        # cv2.imshow("Tello Original", frame) # 如果需要，取消注释以显示原始帧

        # --- 退出条件 ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键
            print("检测到 ESC 键，正在退出...")
            break
        # 添加一个 'l' 键来降落（可选）
        elif key == ord("l"):
            print("准备降落...")
            tello.land()
            print("已发送降落指令。")

    # --- 清理 ---
    tello.streamoff()
    time.sleep(1)

    cv2.destroyAllWindows()
    print("视频流已关闭，窗口已销毁。")
