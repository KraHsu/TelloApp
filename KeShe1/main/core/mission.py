# -*- coding: utf-8 -*-
"""
任务执行模块
"""
import time
import logging
from threading import Thread

from djitellopy import Tello

from config.settings import MISSION_CONFIG
from core.video import VideoProcessor
from core.controller import DroneController


class MissionExecutor:
    """
    任务执行器类，执行预定义的飞行任务

    参数:
        tello: Tello对象
        controller: DroneController对象
        video_processor: VideoProcessor对象
    """

    def __init__(
        self, tello: Tello, controller: DroneController, video_processor: VideoProcessor
    ):
        self.tello = tello
        self.controller = controller
        self.video_processor = video_processor
        self.running = False
        self.current_target = None
        self.target_sequence = MISSION_CONFIG["TARGET_SEQUENCE"]
        self.mission_complete = False

    def start_mission(self):
        """
        开始执行任务

        返回:
            bool: 任务是否成功启动
        """
        if self.running:
            print("任务已在执行中")
            return False

        self.running = True
        self.mission_complete = False

        # 创建并启动任务线程
        mission_thread = Thread(target=self._execute_mission)
        mission_thread.daemon = True
        mission_thread.start()

        return True

    def stop_mission(self):
        """
        停止当前任务
        """
        self.running = False
        print("任务已停止")

    def is_mission_complete(self):
        """
        检查任务是否完成

        返回:
            bool: 任务是否完成
        """
        return self.mission_complete

    def _execute_mission(self):
        """
        执行预定义的任务序列
        """
        try:
            print("开始执行任务...")

            # 初始化连接
            self._init_drone()

            # 起飞并稳定高度
            self._takeoff_and_stabilize()

            # 按顺序执行目标搜索任务
            for i, target_type in enumerate(self.target_sequence):
                if not self.running:
                    break

                print(f"开始执行第{i+1}阶段任务: 寻找{target_type}")
                self._locate_and_approach_target(target_type, i)

            # 返航
            if self.running:
                self._return_home()
                self.mission_complete = True

        except Exception as e:
            print(f"任务执行出错: {e}")
            logging.exception("任务执行异常")
        finally:
            # 确保安全降落
            self._ensure_safe_landing()
            self.running = False
            print("任务执行结束")

    def _init_drone(self):
        """
        初始化无人机连接和设置
        """
        print("初始化无人机...")

        # 启动控制器
        self.controller.start()

        # 启动视频处理器
        self.video_processor.start()

        print(f"无人机电池电量: {self.tello.get_battery()}%")

    def _takeoff_and_stabilize(self):
        """
        起飞并稳定高度
        """
        print("准备起飞...")

        # 起飞
        takeoff_start = time.time()
        self.tello.takeoff()
        print(f"起飞耗时: {time.time() - takeoff_start:.2f}秒")

        # 上升到指定高度
        self.tello.move_up(50)

        # 悬停并稳定
        self.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(1)

        print("高度稳定，准备执行任务")

    def _locate_and_approach_target(self, target_type, stage_index):
        """
        定位并接近目标

        参数:
            target_type (str): 目标类型
            stage_index (int): 阶段索引
        """
        # 设置当前目标类型
        self.current_target = target_type
        self.video_processor.set_target_type(target_type)

        # 关闭目标模式，执行开环移动
        self.controller.set_target_mode(False)

        # 获取此阶段的移动方向
        if stage_index < len(MISSION_CONFIG["MISSION_MOVES"]):
            move_config = MISSION_CONFIG["MISSION_MOVES"][stage_index]
            direction = move_config["direction"]
            duration = move_config["duration"]

            print(f"执行开环{direction}方向移动")
            self.controller.execute_move(direction, duration)

        # 搜索目标
        self.controller.search_for_target(direction)

        # 如果找到目标，启用目标跟踪
        if self.controller.target_got:
            print(f"已找到目标: {target_type}，开始追踪")
            self.controller.set_target_mode(True)

            # 等待接近目标
            approached = self.controller.wait_for_target_approach()

            if approached:
                # 保存图像
                self.video_processor.save_current_frame(f"result_{target_type}.jpg")
                print(f"已成功接近并拍照: {target_type}")
            else:
                print(f"未能成功接近目标: {target_type}")
        else:
            print(f"未找到目标: {target_type}")

    def _return_home(self):
        """
        执行返航程序
        """
        print("开始返航...")

        # 关闭目标跟踪模式
        self.controller.set_target_mode(False)

        # 执行返航动作
        for action in MISSION_CONFIG["RETURN_HOME"]:
            if not self.running:
                break

            command = action["command"]
            if command == "move_left":
                self.tello.move_left(action["distance"])
            elif command == "move_right":
                self.tello.move_right(action["distance"])
            elif command == "move_forward":
                self.tello.move_forward(action["distance"])
            elif command == "move_back":
                self.tello.move_back(action["distance"])

        # 稳定悬停
        self.tello.send_rc_control(0, 0, 0, 0)
        print("返航完成")

    def _ensure_safe_landing(self):
        """
        确保安全降落
        """
        try:
            # 停止控制器
            self.controller.stop()

            # 停止视频处理
            self.video_processor.stop()

            # 发送停止命令
            self.tello.send_rc_control(0, 0, 0, 0)

            # 降落
            self.tello.land()

            # 关闭视频流
            self.tello.streamoff()

            print("无人机已安全降落")

        except Exception as e:
            print(f"降落过程出错: {e}")
            # 尝试紧急降落
            try:
                self.tello.emergency()
            except:
                pass
