import logging
from typing import *
from djitellopy import Tello
import time
import threading
import cv2
import os
import numpy as np
from datetime import datetime


def get_depth(tl: Tello):
    """获取TOF传感器距离数据"""
    try:
        return int(tl.send_read_command("EXT tof?")[4:])
    except:
        return 500  # 如果读取失败，返回一个较大的默认值


if __name__ == "__main__":

    tl = Tello()

    Tello.LOGGER.setLevel(logging.ERROR)

    tl.connect()
    print("电池电量:", tl.get_battery(), "%")

    try:
        while True:
            print(get_depth(tl))
            time.sleep(0.1)
            print("direct: ", tl.get_distance_tof())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("停止")

    finally:
        print("任务结束")
