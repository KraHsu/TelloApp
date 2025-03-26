from typing import *
from djitellopy import Tello
import keyboard
import time
import cv2


SPEED = 30

def show_cmd(tl: Tello, c: str):
    tl.send_expansion_command(f"mled s b {c}")

if __name__ == "__main__":
    tl = Tello()

    tl.connect()
    tl.takeoff()

    while True:
        if keyboard.is_pressed("a"):
            tl.send_rc_control(-SPEED, 0, 0, 0)
            show_cmd(tl, "A")
            continue

        if keyboard.is_pressed("d"):
            tl.send_rc_control(SPEED, 0, 0, 0)
            show_cmd(tl, "D")
            continue

        if keyboard.is_pressed("w"):
            tl.send_rc_control(0, SPEED, 0, 0)
            show_cmd(tl, "W")
            continue

        if keyboard.is_pressed("s"):
            tl.send_rc_control(0, -SPEED, 0, 0)
            show_cmd(tl, "S")
            continue

        if keyboard.is_pressed("q"):
            tl.send_rc_control(0, 0, 0, SPEED)
            show_cmd(tl, "Q")
            continue

        if keyboard.is_pressed("e"):
            tl.send_rc_control(0, 0, 0, -SPEED)
            show_cmd(tl, "E")
            continue

        # 默认显示 T
        show_cmd("T")

        if keyboard.is_pressed("esc"):
            break

        time.sleep(0.01)

    tl.land()
