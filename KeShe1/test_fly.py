import logging
import cv2
from djitellopy import Tello
import time
import os
import numpy as np
from DataCollector import DataCollector
from PIDController import PIDController

TARGET = 100

if __name__ == "__main__":

    dc = DataCollector("height_data.csv")
    pid_height = PIDController(1.7, 0.1, 0, TARGET, (-100, 100))

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

    # frame_read = tl.get_frame_read()

    tl.takeoff()

    begin = time.time()
    tl.move_up(50)
    print(f"升空50cm：{time.time() - begin}")

    begin = time.time()
    tl.send_rc_control(0, 0, 0, 0)
    while time.time() - begin < 20:
        height = tl.query_distance_tof()
        output = pid_height.update(height)

        if time.time() - begin > 12:
            TARGET = 150
            pid_height.set_setpoint(TARGET)
        elif time.time() - begin > 6:
            TARGET = 60
            pid_height.set_setpoint(TARGET)

        dc.collect_datas([("Target", TARGET), ("Tof", height), ("Output", output)])
        tl.send_rc_control(0, 0, int(output), 0)
        time.sleep(0.001)

    tl.send_rc_control(0, 0, 0, 0)

    tl.land()

    dc.save_to_csv()

    tl.streamoff()
