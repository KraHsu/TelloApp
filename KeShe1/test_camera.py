import threading
import cv2
from djitellopy import Tello
import time
import os
import numpy as np


if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print("Tello battery: ", tello.get_battery())
    tello.streamon()
    tello.set_video_direction(tello.CAMERA_DOWNWARD)
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_bitrate(tello.BITRATE_5MBPS)
    tello.set_video_resolution(tello.RESOLUTION_720P)

    time.sleep(2)

    frame_read = tello.get_frame_read()

    while cv2.waitKey(1) & 0xFF != 27:
        cv2.imshow("Name", frame_read.frame)
        cv2.imshow("Name-", frame_read.frame[:240, :, :])

    # tello.land()
    tello.streamoff()
    tello.end()
    time.sleep(1)

    cv2.destroyAllWindows()
    print("视频已保存，无人机已安全降落。")
