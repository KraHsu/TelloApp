# -*- coding: utf-8 -*-
"""
视频处理模块
"""
import os
import cv2
import time
import numpy as np
from threading import Thread, Event
from config.settings import VIDEO_CONFIG
from utils.helpers import add_annotation_area, get_timestamp_str, ensure_dir


class VideoProcessor:
    """
    视频处理器类，处理视频流和录制功能

    参数:
        frame_read: Tello帧读取对象
        detector: 目标检测器对象
        save_video (bool): 是否保存视频
        font_path (str): 字体文件路径
    """

    def __init__(
        self, frame_read, detector, save_video=True, font_path=VIDEO_CONFIG["FONT_PATH"]
    ):
        self.frame_read = frame_read
        self.detector = detector
        self.save_video = save_video
        self.font_path = font_path
        self.running = False
        self.target_type = None
        self.stop_event = Event()

        # 视频帧处理参数
        self.crop_width = VIDEO_CONFIG["CROP_WIDTH"]
        self.crop_height = VIDEO_CONFIG["CROP_HEIGHT"]

        # 视频保存器
        self.video_writer = None
        if save_video:
            self._init_video_writer()

        # 处理结果
        self.current_result = {
            "center_x": -1,
            "center_y": -1,
            "distance": -1,
            "detected": False,
        }

        # 处理线程
        self.process_thread = None

    def _init_video_writer(self):
        """
        初始化视频写入器
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4格式
        timestamp = get_timestamp_str()

        save_dir = VIDEO_CONFIG["SAVE_DIR"]
        ensure_dir(save_dir)

        video_path = os.path.join(save_dir, f"res_{timestamp}.mp4")

        # 获取完整的视频尺寸 注意转置
        frame_height = self.crop_width + VIDEO_CONFIG["ANNOTATION_HEIGHT"]
        frame_width = self.crop_height

        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, 30.0, (frame_width, frame_height)
        )
        print(f"视频将保存到: {video_path}")

    def start(self, target_type=None):
        """
        启动视频处理

        参数:
            target_type (str, optional): 目标类型
        """
        if self.running:
            print("视频处理器已在运行")
            return

        self.running = True
        self.target_type = target_type
        self.stop_event.clear()

        self.process_thread = Thread(target=self._process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        print("视频处理器已启动")

    def stop(self):
        """
        停止视频处理
        """
        self.running = False
        self.stop_event.set()

        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        cv2.destroyAllWindows()
        print("视频处理器已停止")

    def set_target_type(self, target_type):
        """
        设置目标类型

        参数:
            target_type (str): 目标类型
        """
        self.target_type = target_type
        print(f"目标类型已设置为: {target_type}")

    def get_current_result(self):
        """
        获取当前处理结果

        返回:
            dict: 当前处理结果
        """
        return self.current_result.copy()

    def is_target_detected(self):
        """
        检查是否检测到目标

        返回:
            bool: 是否检测到目标
        """
        return self.current_result["detected"]

    def save_current_frame(self, file_name=None):
        """
        保存当前帧

        参数:
            file_name (str, optional): 文件名，如不指定则使用时间戳
        """
        if not hasattr(self, "current_frame") or self.current_frame is None:
            print("没有可用的帧")
            return

        if file_name is None:
            timestamp = get_timestamp_str()
            file_name = f"frame_{timestamp}.jpg"

        save_dir = VIDEO_CONFIG["SAVE_DIR"]
        ensure_dir(save_dir)

        file_path = os.path.join(save_dir, file_name)
        cv2.imwrite(file_path, self.current_frame)
        print(f"已保存当前帧到: {file_path}")

    def _process_video(self):
        """
        视频处理主循环
        """
        no_target_count = 0

        while self.running and not self.stop_event.is_set():
            # 获取帧
            frame = self.frame_read.frame
            if frame is None:
                print("未能获取到帧，可能连接中断")
                time.sleep(0.5)  # 短暂等待后重试
                continue

            # 裁剪帧
            h, w, _ = frame.shape
            actual_crop_height = min(self.crop_height, h)  # 防止裁剪高度超过实际高度
            cropped_frame = frame[
                :actual_crop_height, :, :
            ].copy()  # 裁剪帧并创建副本以进行修改
            cropped_frame = cv2.flip(cv2.transpose(cropped_frame), 1)

            # 目标检测
            processed_frame, result = self.detector.detect(
                cropped_frame, self.target_type
            )

            # 更新当前结果
            self.current_result = result

            # 更新连续未检测计数
            if result["detected"]:
                no_target_count = 0
            else:
                no_target_count += 1

            # 添加注释区域
            annotated_frame = add_annotation_area(
                processed_frame,
                result["center_x"] if result["detected"] else -1,
                result["center_y"] if result["detected"] else -1,
                result["distance"] if result["detected"] else -1,
                self.font_path,
            )

            # 保存当前帧（用于可能的截图）
            self.current_frame = annotated_frame

            # 显示视频
            cv2.imshow("Tello目标检测", annotated_frame)

            # 保存视频
            if self.video_writer:
                self.video_writer.write(annotated_frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                self.running = False
                break
            elif key == ord("s"):  # 's'键保存当前帧
                self.save_current_frame()

            # 处理帧率控制
            time.sleep(0.01)  # 限制帧率

        # 清理资源
        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()
        print("视频处理线程已结束")
