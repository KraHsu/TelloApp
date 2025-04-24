"""
视频处理模块 (modules/video_handler.py)
--------------------------------

负责从Tello无人机获取视频流并进行基础处理。
该模块封装了视频获取和预处理操作，提供简洁的接口给其他组件使用。
"""

import cv2
import numpy as np
import time
from .utils import Logger

# 初始化日志器
logger = Logger()


class VideoHandler:
    """视频处理类，用于处理从Tello获取的视频流"""

    def __init__(self, tello, frame_resize=None):
        """
        初始化视频处理器

        参数:
            tello: 已连接的Tello实例
            frame_resize: 可选的调整大小元组 (width, height)
        """
        self.tello = tello
        self.frame_resize = frame_resize
        self.frame_read = tello.get_frame_read()
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0

        # 每30帧更新一次FPS
        self.fps_update_interval = 30

        logger.info("视频处理器初始化完成")

    def get_frame(self):
        """
        获取当前视频帧并进行基础处理

        返回:
            processed_frame: 处理后的视频帧，如果无法获取则为None
        """
        if self.frame_read.frame is None:
            logger.warning("无法获取视频帧")
            return None

        # 获取原始帧
        frame = self.frame_read.frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 计算FPS
        self.frame_count += 1
        if self.frame_count % self.fps_update_interval == 0:
            current_time = time.time()
            time_diff = current_time - self.last_frame_time
            self.fps = self.fps_update_interval / time_diff if time_diff > 0 else 0
            self.last_frame_time = current_time

        # 如果需要调整大小
        if self.frame_resize:
            frame = cv2.resize(frame, self.frame_resize)

        return frame

    def apply_basic_enhancement(self, frame):
        """
        应用基本图像增强

        参数:
            frame: 输入视频帧

        返回:
            enhanced_frame: 增强后的视频帧
        """
        if frame is None:
            return None

        # 自动对比度增强
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_frame

    def get_enhanced_frame(self):
        """
        获取增强后的视频帧

        返回:
            enhanced_frame: 增强后的视频帧
        """
        frame = self.get_frame()
        if frame is None:
            return None

        return self.apply_basic_enhancement(frame)

    def get_frame_with_info(self):
        """
        获取带有信息叠加的视频帧

        返回:
            info_frame: 带有信息的视频帧
        """
        frame = self.get_frame()
        if frame is None:
            return None

        # 添加FPS信息
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # 添加电池信息
        try:
            battery = self.tello.get_battery()
            cv2.putText(
                frame,
                f"电池: {battery}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if battery > 20 else (0, 0, 255),
                1,
            )
        except:
            pass

        return frame

    def record_video(self, output_file, duration=None, fps=30.0):
        """
        录制视频

        参数:
            output_file: 输出文件路径
            duration: 录制时长（秒），如果为None则一直录制直到被中断
            fps: 帧率

        返回:
            bool: 成功返回True，否则返回False
        """
        if self.frame_read.frame is None:
            logger.error("无法获取视频帧，无法录制")
            return False

        # 获取第一帧以确定大小
        frame = self.get_frame()
        if frame is None:
            logger.error("无法获取视频帧，无法录制")
            return False

        height, width = frame.shape[:2]

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        logger.info(f"开始录制视频到 {output_file}")
        start_time = time.time()

        try:
            while True:
                # 获取帧
                frame = self.get_frame()
                if frame is None:
                    continue

                # 写入帧
                out.write(frame)

                # 如果指定了时长且已达到时长，则停止录制
                if duration is not None and time.time() - start_time >= duration:
                    break

                # 检查是否请求中断
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break

                # 限制帧率
                time.sleep(1 / fps)

        except KeyboardInterrupt:
            logger.info("用户中断录制")
        except Exception as e:
            logger.error(f"录制过程中出错: {e}")
            return False
        finally:
            # 释放视频写入器
            out.release()
            logger.info(f"视频录制完成: {output_file}")

        return True

    def take_photo(self, output_file):
        """
        拍摄照片并保存

        参数:
            output_file: 输出文件路径

        返回:
            bool: 成功返回True，否则返回False
        """
        frame = self.get_enhanced_frame()
        if frame is None:
            logger.error("无法获取视频帧，无法拍照")
            return False

        try:
            cv2.imwrite(output_file, frame)
            logger.info(f"已保存照片: {output_file}")
            return True
        except Exception as e:
            logger.error(f"保存照片时出错: {e}")
            return False
