from typing import *
from djitellopy import Tello
import keyboard
import time
import threading
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

frame = None
show = True

def show_img():
    while show:
        if frame is not None:
            cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tl = Tello()

    tl.connect()
    tl.takeoff()

    tl.streamoff()
    tl.streamon()

    frame_read = tl.get_frame_read()

    threading.Thread(target=show_img).start()

    try:
        while True:
            frame = frame_read.frame

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

            # 默认关闭
            mled_off(tl)

            if keyboard.is_pressed("esc"):
                break

    except Exception as e:
        print(e)

    finally:
        print("降落")
        tl.streamoff()
        tl.land()

        show = False
