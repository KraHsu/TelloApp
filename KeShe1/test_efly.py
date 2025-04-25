import logging
import cv2
from djitellopy import Tello
import time
import os
import numpy as np
from DataCollector import DataCollector
from PIDController import PIDController

from Utils import *

TARGET = 100


def e_fly(tl: Tello):
    a = 20
    b = 50
    delta = 5
    g = lambda t: 4 * sigmoid(t) * (1 - sigmoid(t))
    h = lambda t: 50 * g(t) + 10
    clamp = create_smooth_clamp(a, b, delta)

    time.sleep(0.1)

    begin = time.time()
    while time.time() - begin < 5:
        t = time.time() - begin

        v = clamp(h(t))

        tl.send_rc_control(0, int(v), 0, 0)

        time.sleep(0.001)


if __name__ == "__main__":
    tl = Tello()

    Tello.LOGGER.setLevel(logging.WARNING)

    tl.connect()
    print(f"电池：{tl.query_battery()}%")
    print(f"状态：{tl.get_current_state()}")
    tl.streamon()
    tl.set_video_fps(Tello.FPS_30)
    tl.set_video_bitrate(Tello.BITRATE_5MBPS)
    tl.set_video_direction(Tello.CAMERA_DOWNWARD)
    tl.set_video_resolution(Tello.RESOLUTION_720P)

    tl.takeoff()

    begin = time.time()
    tl.move_up(50)
    print(f"升空50cm：{time.time() - begin}")

    tl.send_rc_control(0, 0, 0, 0)

    e_fly(tl)
    
    # begin = time.time()
    # while time.time() - begin < 5:
    #     tl.send_rc_control(0, 20, 0, 0)

    tl.send_rc_control(0, 0, 0, 0)

    tl.land()

    tl.streamoff()
