# -*- coding: utf-8 -*-
"""
飞行控制模块
"""
import logging
import time
import math
from threading import Thread, Event
from djitellopy import Tello
from config.settings import CONTROL_CONFIG, PID_CONFIG
from utils.pid_controller import PIDController
from utils.helpers import sigmoid, create_smooth_clamp


class DroneController:
    """
    无人机控制器类

    参数:
        tello (Tello): Tello对象
    """

    def __init__(
        self, tello: Tello, logger: logging.Logger = logging.getLogger("tello_tracking")
    ):
        self.tello: Tello = tello
        self.running = False
        self.stop_event = Event()
        self.logger = logger

        # 控制模式
        self.control_mode_target = False

        # 目标检测状态
        self.target_got = False

        # 输出控制信号
        self.output = {
            "vx": 0,  # 前后速度，前+
            "vy": 0,  # 左右速度，左+
            "vz": 0,  # 上下速度，上+
            "vr": 0,  # yaw速度，逆时针+
            "d": 0,  # 距离
        }

        # PID控制器
        self._init_pid_controllers()

        # 控制线程
        self.height_thread = None
        self.control_thread = None

    def _init_pid_controllers(self):
        """
        初始化PID控制器
        """
        # 普通速度PID
        self.pid_vx = PIDController(
            PID_CONFIG["NORMAL"]["P"],
            PID_CONFIG["NORMAL"]["I"],
            PID_CONFIG["NORMAL"]["D"],
            0,
            PID_CONFIG["NORMAL"]["LIMIT"],
            PID_CONFIG["NORMAL"]["OUTPUT_LIMIT"],
        )

        self.pid_vy = PIDController(
            PID_CONFIG["NORMAL"]["P"],
            PID_CONFIG["NORMAL"]["I"],
            PID_CONFIG["NORMAL"]["D"],
            0,
            PID_CONFIG["NORMAL"]["LIMIT"],
            PID_CONFIG["NORMAL"]["OUTPUT_LIMIT"],
        )

        # 慢速PID
        self.pid_vx_slow = PIDController(
            PID_CONFIG["SLOW"]["P"],
            PID_CONFIG["SLOW"]["I"],
            PID_CONFIG["SLOW"]["D"],
            0,
            PID_CONFIG["SLOW"]["LIMIT"],
            PID_CONFIG["SLOW"]["OUTPUT_LIMIT"],
        )

        self.pid_vy_slow = PIDController(
            PID_CONFIG["SLOW"]["P"],
            PID_CONFIG["SLOW"]["I"],
            PID_CONFIG["SLOW"]["D"],
            0,
            PID_CONFIG["SLOW"]["LIMIT"],
            PID_CONFIG["SLOW"]["OUTPUT_LIMIT"],
        )

        # 高度PID
        self.pid_height = PIDController(
            PID_CONFIG["HEIGHT"]["P"],
            PID_CONFIG["HEIGHT"]["I"],
            PID_CONFIG["HEIGHT"]["D"],
            CONTROL_CONFIG["TARGET_HEIGHT"],
            PID_CONFIG["HEIGHT"]["LIMIT"],
            PID_CONFIG["HEIGHT"]["OUTPUT_LIMIT"],
        )

    def start(self):
        """
        启动控制器
        """
        if self.running:
            self.logger.info("控制器已在运行")
            return

        self.running = True
        self.stop_event.clear()

        # 启动高度控制线程
        self.height_thread = Thread(target=self._keep_height)
        self.height_thread.daemon = True
        self.height_thread.start()

        # 启动主控制线程
        self.control_thread = Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        self.logger.info("无人机控制器已启动")

    def stop(self):
        """
        停止控制器
        """
        self.running = False
        self.stop_event.set()

        # 停止无人机移动
        self.tello.send_rc_control(0, 0, 0, 0)

        # 等待线程结束
        if self.height_thread and self.height_thread.is_alive():
            self.height_thread.join(timeout=2.0)

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        self.logger.info("无人机控制器已停止")

    def set_target_mode(self, enabled):
        """
        设置目标跟踪模式

        参数:
            enabled (bool): 是否启用目标跟踪模式
        """
        self.control_mode_target = enabled
        self.logger.info(f"目标跟踪模式: {'已启用' if enabled else '已禁用'}")

    def update_target_status(
        self, detected, center_x=None, center_y=None, distance=None
    ):
        """
        更新目标状态

        参数:
            detected (bool): 是否检测到目标
            center_x (int, optional): 目标中心X坐标
            center_y (int, optional): 目标中心Y坐标
            distance (float, optional): 距离
        """
        self.target_got = detected

        if detected and center_x is not None and center_y is not None:
            # 使用PID控制器更新输出
            target_center_x = CONTROL_CONFIG["TARGET_CENTER_X"]
            target_center_y = CONTROL_CONFIG["TARGET_CENTER_Y"]

            dx = target_center_x - center_x
            dy = target_center_y - center_y

            d = math.sqrt(dx**2 + dy**2)

            f = lambda x: x**3 / 21000 + 97 * x / 105

            if d > 30:
                # 使用普通PID
                self.output["vy"] = self.pid_vx.update(f(dx))
                self.output["vx"] = -self.pid_vy.update(f(dy))
                self.pid_vx_slow.reset()
                self.pid_vy_slow.reset()
            else:
                # 使用慢速PID
                self.pid_vx.reset()
                self.pid_vy.reset()
                self.output["vy"] = self.pid_vx_slow.update(f(dx))
                self.output["vx"] = -self.pid_vy_slow.update(f(dy))

            if distance is not None:
                self.output["d"] = distance

        else:
            # 如果没有检测到目标，重置控制
            self.pid_vx.reset()
            self.pid_vy.reset()
            self.pid_vx_slow.reset()
            self.pid_vy_slow.reset()

    def execute_move(self, direction, duration=5):
        """
        执行平滑移动

        参数:
            direction (str): 移动方向，'F'前，'B'后，'L'左，'R'右
            duration (float): 移动持续时间（秒）
        """
        # 保存当前控制模式
        original_mode = self.control_mode_target
        self.control_mode_target = False

        # 设置平滑曲线参数
        a = 20
        b = 50
        delta = 5

        g = lambda t: 4 * sigmoid(t) * (1 - sigmoid(t))
        h = lambda t: 50 * g(t / 1.5) + 10
        clamp = create_smooth_clamp(a, b, delta)

        self.logger.info(f"执行{direction}方向移动，持续{duration}秒")

        # 执行移动
        begin = time.time()
        # while time.time() - begin < 1 and self.running:
        #     v = 50
        #     if direction == "F":
        #         self.tello.send_rc_control(0, int(v), int(self.output["vz"]), 0)
        #     elif direction == "R":
        #         self.tello.send_rc_control(int(v), 0, int(self.output["vz"]), 0)
        #     elif direction == "B":
        #         self.tello.send_rc_control(0, -int(v), int(self.output["vz"]), 0)
        #     elif direction == "L":
        #         self.tello.send_rc_control(-int(v), 0, int(self.output["vz"]), 0)
        
        # DEBUG 开环
        v = 50
        duration = 2.2
        while time.time() - begin < duration and self.running:
            # t = time.time() - begin
            # v = clamp(h(t))

            if direction == "F":
                self.tello.send_rc_control(0, int(v), int(self.output["vz"]), 0)
            elif direction == "R":
                self.tello.send_rc_control(int(v), 0, int(self.output["vz"]), 0)
            elif direction == "B":
                self.tello.send_rc_control(0, -int(v), int(self.output["vz"]), 0)
            elif direction == "L":
                self.tello.send_rc_control(-int(v), 0, int(self.output["vz"]), 0)

            time.sleep(0.001)

        # 恢复原始控制模式
        self.control_mode_target = original_mode
        self.logger.info("移动完成")

    def search_for_target(self, direction="F", speed=20):
        """
        搜索目标

        参数:
            direction (str): 搜索方向，'F'前，'B'后，'L'左，'R'右
            speed (int): 搜索速度
        """
        # 保存当前控制模式
        original_mode = self.control_mode_target
        self.control_mode_target = False

        self.logger.info(f"开始搜索目标，方向: {direction}")

        while not self.target_got and self.running:
            if direction == "F":
                self.tello.send_rc_control(0, speed, int(self.output["vz"]), 0)
            elif direction == "R":
                self.tello.send_rc_control(speed, 0, int(self.output["vz"]), 0)
            elif direction == "B":
                self.tello.send_rc_control(0, -speed, int(self.output["vz"]), 0)
            elif direction == "L":
                self.tello.send_rc_control(-speed, 0, int(self.output["vz"]), 0)

            time.sleep(0.001)

        # 恢复原始控制模式
        self.control_mode_target = original_mode
        self.logger.info("目标已找到" if self.target_got else "搜索结束")

    def wait_for_target_approach(
        self, min_distance=CONTROL_CONFIG["MIN_TARGET_DISTANCE"]
    ):
        """
        等待接近目标

        参数:
            min_distance (float): 最小目标距离

        返回:
            bool: 是否成功接近目标
        """
        self.logger.info(f"等待接近目标 (距离 < {min_distance})")

        while self.running:
            # if not self.target_got:
            #     self.logger.warning("目标丢失，接近终止")
            #     return False

            if self.output["d"] > 0 and self.output["d"] < min_distance:
                self.logger.info(f"成功接近目标，距离: {self.output['d']:.2f}")
                return True

            time.sleep(0.1)

        return False

    def _keep_height(self):
        """
        保持高度控制线程
        """
        while self.running and not self.stop_event.is_set():
            try:
                current_height = self.tello.get_distance_tof()
                self.output["vz"] = self.pid_height.update(current_height)
            except Exception as e:
                self.logger.error(f"高度控制异常: {e}")

            time.sleep(0.01)

        self.output["vz"] = 0
        self.logger.info("高度控制线程已结束")

    def _control_loop(self):
        """
        主控制循环线程
        """
        while self.running and not self.stop_event.is_set():
            if self.control_mode_target:
                # DEBUG PID OUTPUT
                # 目标跟踪模式，使用PID输出控制

                min = 7
                zero = 0

                if zero < self.output["vy"] < min + 1:
                    self.output["vy"] = min
                elif -min - 1 < self.output["vy"] < -zero:
                    self.output["vy"] = -min
                # elif -zero <= self.output["vy"] <= zero:
                #     self.output["vy"] = 0

                if 0 < self.output["vx"] < min + 1:
                    self.output["vx"] = min
                elif -min - 1 < self.output["vx"] < 0:
                    self.output["vx"] = -min
                # elif -zero <= self.output["vx"] <= zero:
                #     self.output["vx"] = 0

                self.tello.send_rc_control(
                    int(self.output["vy"]),
                    int(self.output["vx"]),
                    int(self.output["vz"]),
                    0,
                )

            time.sleep(0.01)

        # 停止移动
        self.tello.send_rc_control(0, 0, 0, 0)
        self.logger.info("控制线程已结束")
