# -*- coding: utf-8 -*-
"""
Tello智能目标追踪系统 - 主程序入口
"""
import os
import time
import logging
import sys
from threading import Thread
from djitellopy import Tello

# 导入自定义模块
from config.settings import MODEL_CONFIG, VIDEO_CONFIG, DATA_CONFIG
from core.detector import ObjectDetector
from core.controller import DroneController
from core.mission import MissionExecutor
from core.video import VideoProcessor
from utils.data_collector import DataCollector
from utils.helpers import ensure_dir


def setup_logging() -> logging.Logger:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    ensure_dir(log_dir)

    log_file = os.path.join(log_dir, f"tello_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 中文
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler],
    )

    logging.getLogger("djitellopy").setLevel(logging.WARNING)
    return logging.getLogger("tello_tracking")


def initialize_tello() -> Tello:
    """
    初始化Tello对象

    返回:
        Tello: 初始化后的Tello对象
    """
    tello = Tello()

    # DEBUG 注释这部分内容以启用飞行
    # no_func = lambda *args: 1

    # tello.takeoff = no_func
    # tello.send_rc_control = no_func

    # tello.move_up = no_func
    # tello.move_down = no_func
    # tello.move_forward = no_func
    # tello.move_back = no_func
    # tello.move_right = no_func
    # tello.move_left = no_func

    # tello.land = no_func
    # tello.emergency = no_func

    try:
        tello.connect()
        logging.info(f"成功连接到Tello，电池电量: {tello.get_battery()}%")
        tello.streamon()
        tello.set_video_fps(Tello.FPS_30)
        tello.set_video_bitrate(Tello.BITRATE_2MBPS)
        tello.set_video_direction(Tello.CAMERA_DOWNWARD)
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        logging.info(f"Tello 相机已打开")

        return tello
    except Exception as e:
        logging.error(f"连接Tello时出错: {e}")
        raise


def run_mission(tello: Tello):
    """
    运行完整的任务

    参数:
        tello (Tello): Tello对象
    """
    # 创建数据收集器
    data_collector = DataCollector(DATA_CONFIG["DATA_FILE"])

    try:
        # 初始化目标检测器
        detector = ObjectDetector(
            MODEL_CONFIG["MODEL_PATH"],
            MODEL_CONFIG["CONFIDENCE_THRESHOLD"],
            MODEL_CONFIG["TARGET_CLASSES"],
            MODEL_CONFIG["DEVICE"],
        )

        # 初始化视频处理器
        video_processor = VideoProcessor(
            tello.get_frame_read(),
            detector,
            save_video=True,
        )

        # 初始化飞行控制器
        controller = DroneController(tello)

        # 视频处理回调函数，用于更新控制器状态
        def process_detection_result():
            while True:
                result = video_processor.get_current_result()
                controller.update_target_status(
                    result["detected"],
                    result.get("center_x"),
                    result.get("center_y"),
                    result.get("distance"),
                )

                # 收集数据
                if result["detected"]:
                    data_collector.collect_datas(
                        [
                            ("x", result["center_x"]),
                            ("y", result["center_y"]),
                            ("distance", result["distance"]),
                            ("vx", controller.output["vx"]),
                            ("vy", controller.output["vy"]),
                            ("vz", controller.output["vz"]),
                        ]
                    )

                # 50Hz
                time.sleep(0.02)

        # 启动处理线程
        process_thread = Thread(target=process_detection_result)
        process_thread.daemon = True
        process_thread.start()

        # 创建任务执行器
        mission = MissionExecutor(tello, controller, video_processor)

        # 启动任务
        mission.start_mission()

        # 等待任务完成
        while not mission.is_mission_complete():
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("接收到终止信号，正在停止...")
    except Exception as e:
        logging.error(f"任务执行出错: {e}")
        logging.exception("异常详情")
    finally:
        # 保存收集的数据
        data_collector.save_to_csv()

        # 确保安全降落
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
        tello.streamoff()


def main():
    """
    主程序入口
    """
    # 设置日志
    logger = setup_logging()
    logger.info("===== Tello智能目标追踪系统启动 =====")

    try:
        # 初始化Tello
        tello = initialize_tello()

        # 运行任务
        begin = time.time()
        run_mission(tello)
        print(f"==== END TIME: {time.time()}")
        print(f"总耗时：{time.time() - begin} s")

    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.exception("异常详情")

    logger.info("===== 程序结束 =====")


if __name__ == "__main__":
    main()
