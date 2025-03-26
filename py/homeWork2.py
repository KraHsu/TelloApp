from typing import *
from djitellopy import Tello
import keyboard
import time
import cv2


SPEED = 20

def show_cmd(tl: Tello, c: str):
    tl.send_expansion_command(f"mled s r {c}")
    tl.send_expansion_command(f"mled sl 255")

def mled_off(tl: Tello):
    tl.send_expansion_command(f"mled sl 0")

def get_depth(tl: Tello):
    return int(tl.send_read_command('EXT tof?')[4:])

def led(tl: Tello, c: str):
    if c == "r":
        c = "255 0 0"
    elif c == "g":
        c = "0 255 0"
    elif c == "b":
        c = "0 0 255"
    tl.send_expansion_command(f"led {c}")

if __name__ == "__main__":
    tl = Tello()

    tl.connect()

    mled_off(tl)
    led(tl, "b")

    tl.enable_mission_pads()
    tl.set_mission_pad_detection_direction(0)
    tl.takeoff()

    for i in range(0, 3):
        while get_depth(tl) >= 1000:
            tl.send_rc_control(0, SPEED, 0, 0)
            time.sleep(0.01)

        tl.send_rc_control(0, 0, 0, 0)

        time.sleep(0.5)
        
        id = tl.get_mission_pad_id()
        if id != -1:
            led(tl, 'g')
            show_cmd(tl, f"{id}")
            tl.go_xyz_speed_mid(0, 0, 40, SPEED, id)
        else:
            led(tl, "r")
            mled_off(tl)

        print(f"识别: {i}:{id}")

        tl.rotate_clockwise(-90)

        if keyboard.is_pressed("esc"):
            break

        time.sleep(0.5)

        mled_off(tl)
        led(tl, "b")

    tl.land()
