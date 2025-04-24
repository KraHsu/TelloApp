"""
Tello无人机人脸识别系统
====================

项目概述:
-------
本项目利用DJI Tello无人机实现实时人脸检测与识别功能，可用于智能无人系统中的人员识别、
目标跟踪等场景。项目支持多种人脸检测方法，包括传统的OpenCV Haar级联分类器、
更先进的深度学习方法如OpenCV DNN、MTCNN和face_recognition库。

主要功能:
-------
1. 实时视频流获取与处理
2. 多种人脸检测算法支持与切换
3. 特定人员识别功能（基于face_recognition）
4. 多线程设计，提高系统响应速度
5. 用户友好的交互界面

技术栈:
------
- DJITelloPy: 与Tello无人机通信
- OpenCV: 图像处理与显示
- face_recognition: 高精度人脸识别
- MTCNN: 深度学习人脸检测
- NumPy: 数据处理
- Threading: 多线程支持

使用方法:
-------
python tello_face_recognition.py [--method METHOD] [--recognize] [--images DIR]

参数说明:
  -m, --method    选择人脸检测方法: opencv, opencv_dnn, mtcnn, face_recognition
  -r, --recognize 启用特定人员识别模式(仅适用于face_recognition方法)
  -i, --images    包含已知人脸图像的文件夹路径

开发者: [您的姓名]
版本: 1.0.0
日期: 2025-04-02
"""

# 导入所需的库
import os
import cv2
import time
import threading
import numpy as np
import argparse
import re
from datetime import datetime
import queue
from djitellopy import Tello

# 项目模块导入
from modules.config import Config
from modules.detectors import FaceDetectorFactory
from modules.video_handler import VideoHandler
from modules.utils import Logger, FaceUtils
from modules.ui import UserInterface


def parse_arguments():
    """
    解析命令行参数

    返回:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Tello无人机人脸检测与识别系统")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="opencv",
        choices=["opencv", "opencv_dnn", "mtcnn", "face_recognition"],
        help="选择人脸检测方法: opencv, opencv_dnn, mtcnn, 或 face_recognition",
    )
    parser.add_argument(
        "-r",
        "--recognize",
        action="store_true",
        help="启用特定人员识别模式(仅适用于face_recognition方法)",
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        default="known_faces",
        help="包含已知人脸图像的文件夹路径",
    )
    return parser.parse_args()


class TelloFaceRecognitionSystem:
    """
    Tello无人机人脸识别系统主类

    负责协调各模块工作，包括无人机控制、视频处理、人脸检测与识别等
    """

    def __init__(self, args):
        """
        初始化系统

        参数:
            args: 命令行参数
        """
        # 配置初始化
        self.config = Config(args)
        self.logger = Logger()

        self.logger.set_level("DEBUG")

        # 状态变量
        self.running = True
        self.frame = None
        self.latest_frame = None
        self.latest_results = None
        self.processing = False
        self.results_frame_count = 0
        self.last_process_time = 0
        self.results_lock = threading.Lock()

        # 初始化检测器工厂
        self.detector_factory = FaceDetectorFactory()
        self.face_detector = self.detector_factory.create_detector(
            self.config.current_method,
            self.config.enable_recognition,
            self.config.known_faces_dir,
        )

        # 用户界面
        self.ui = UserInterface()

        # 无人机和视频处理器初始化为None，后续连接
        self.tello = None
        self.video_handler = None

    def connect_drone(self):
        """连接到Tello无人机并初始化视频流"""
        try:
            self.tello = Tello()
            self.tello.connect()
            self.logger.info(f"连接成功! 电池电量: {self.tello.get_battery()}%")

            # 初始化视频流
            self.tello.streamoff()
            self.tello.streamon()
            self.logger.info("视频流已启动")

            # 初始化视频处理器
            self.video_handler = VideoHandler(self.tello)

            # 等待视频流初始化
            time.sleep(2)
            return True
        except Exception as e:
            self.logger.error(f"连接无人机失败: {e}")
            return False

    def start_video_thread(self):
        """启动视频处理线程"""
        video_thread = threading.Thread(target=self._video_thread_function)
        video_thread.daemon = True
        video_thread.start()

        # 等待视频流第一帧
        timeout = 10  # 10秒超时
        start_time = time.time()
        while self.latest_frame is None:
            if time.time() - start_time > timeout:
                self.logger.warning("等待视频流超时，可能无法正常工作")
                break
            time.sleep(0.1)

        if self.latest_frame is not None:
            self.logger.info("视频流已准备就绪")
            return True
        return False

    def _video_thread_function(self):
        """视频线程函数，从无人机获取视频帧"""
        try:
            while self.running:
                current_frame = self.video_handler.get_frame()
                if current_frame is not None:
                    # 创建新的副本以避免引用问题
                    self.frame = current_frame.copy()
                    self.latest_frame = current_frame.copy()
                time.sleep(0.03)  # 限制帧率以减少CPU使用
        except Exception as e:
            self.logger.error(f"视频流错误: {e}")
        finally:
            self.logger.info("视频线程结束")

    def start_detection_thread(self):
        """启动人脸检测线程"""
        detection_thread = threading.Thread(target=self._face_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        self.logger.info("人脸检测线程已启动")

    def _face_detection_thread(self):
        """人脸检测线程，负责处理视频帧中的人脸检测"""
        while self.running:
            # 检查是否有帧可处理
            current_frame = None

            with self.results_lock:
                # 处理超时检测，如果处理时间超过3秒，重置处理状态
                if self.processing and time.time() - self.last_process_time > 3:
                    self.logger.warning("检测到处理超时，重置处理状态")
                    self.processing = False

                # 只有当不在处理中且有可用帧时才处理
                if not self.processing and self.latest_frame is not None:
                    current_frame = self.latest_frame.copy()
                    self.processing = True
                    self.last_process_time = time.time()

            # 如果没有需要处理的帧，等待一会儿
            if current_frame is None:
                time.sleep(0.01)
                continue

            try:
                # 执行人脸检测
                faces, names = self.face_detector.detect_faces(current_frame)

                # 更新结果
                with self.results_lock:
                    self.latest_results = (faces, names, current_frame.shape)
                    self.results_frame_count = 0
                    self.processing = False
            except Exception as e:
                self.logger.error(f"人脸检测错误: {e}")
                with self.results_lock:
                    self.processing = False

            # 控制处理频率，避免CPU过载
            time.sleep(0.01)

    def process_frame_loop(self):
        """主循环，处理并显示视频帧"""
        try:
            while self.running:
                if self.frame is None:
                    time.sleep(0.1)
                    continue

                # 制作显示帧的副本
                display_frame = self.frame.copy()

                # 获取当前的识别结果
                faces = []
                names = []
                reused_result = False

                with self.results_lock:
                    if self.latest_results is not None:
                        # 检查是否需要重用结果
                        if (
                            self.processing
                            and self.results_frame_count < self.config.max_reuse_frames
                        ):
                            faces, names, frame_shape = self.latest_results

                            # 检查帧尺寸是否匹配
                            if (
                                frame_shape[0] == display_frame.shape[0]
                                and frame_shape[1] == display_frame.shape[1]
                            ):
                                self.results_frame_count += 1
                                reused_result = True
                            else:
                                faces = []
                                names = []
                        elif not self.processing:
                            faces, names, _ = self.latest_results
                            self.results_frame_count = 0

                # 绘制UI和识别结果
                result_frame = self.ui.draw_results(
                    display_frame,
                    faces,
                    names,
                    self.config.current_method,
                    self.config.enable_recognition,
                    self.processing,
                    reused_result,
                    self.results_frame_count,
                )

                # 显示帧
                cv2.imshow("Tello人脸识别系统", result_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break
                elif key == 32:  # 空格键
                    self._toggle_detection_method()
                elif key == ord("r"):  # 'r'键
                    self._toggle_recognition_mode()
        finally:
            cv2.destroyAllWindows()

    def _toggle_detection_method(self):
        """切换人脸检测方法"""
        # 获取可用方法列表
        available_methods = self.detector_factory.get_available_methods()

        # 当前方法索引
        current_idx = (
            available_methods.index(self.config.current_method)
            if self.config.current_method in available_methods
            else 0
        )

        # 切换到下一个方法
        self.config.current_method = available_methods[
            (current_idx + 1) % len(available_methods)
        ]

        # 更新检测器
        self.face_detector = self.detector_factory.create_detector(
            self.config.current_method,
            self.config.enable_recognition,
            self.config.known_faces_dir,
        )

        self.logger.info(f"切换到检测方法: {self.config.current_method}")

    def _toggle_recognition_mode(self):
        """切换识别模式"""
        if (
            self.config.current_method == "face_recognition"
            and self.detector_factory.has_face_recognition
        ):
            self.config.enable_recognition = not self.config.enable_recognition
            mode = "启用" if self.config.enable_recognition else "禁用"
            self.logger.info(f"{mode}特定人员识别模式")

            # 更新检测器
            self.face_detector = self.detector_factory.create_detector(
                self.config.current_method,
                self.config.enable_recognition,
                self.config.known_faces_dir,
            )
        else:
            self.logger.warning("特定人员识别仅在face_recognition模式下可用")

    def run(self):
        """运行系统主流程"""
        try:
            self.logger.info(
                f"启动Tello人脸识别系统 - 使用{self.config.current_method}方法"
            )

            # 连接无人机
            if not self.connect_drone():
                return

            # 启动视频线程
            if not self.start_video_thread():
                return

            # 启动人脸检测线程
            self.start_detection_thread()

            # 进入主循环
            self.process_frame_loop()

        except KeyboardInterrupt:
            self.logger.info("用户中断程序")
        except Exception as e:
            self.logger.error(f"系统错误: {e}")
        finally:
            # 清理资源
            self.cleanup()

    def cleanup(self):
        """清理系统资源"""
        self.running = False
        if self.tello:
            try:
                self.logger.info("关闭视频流...")
                self.tello.streamoff()
            except:
                pass
        self.logger.info("系统已关闭")


def main():
    """主函数"""
    args = parse_arguments()
    system = TelloFaceRecognitionSystem(args)
    system.run()


if __name__ == "__main__":
    main()
