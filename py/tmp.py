from typing import *
from djitellopy import Tello
import time
import cv2


if __name__ == "__main__":
    tl = Tello()

    tl.connect()
    tl.streamoff()
    tl.streamon()

    frame_read = tl.get_frame_read()

    while True:
        frame = frame_read.frame

        print(frame.shape)

        cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            tl.streamoff()
            cv2.destroyAllWindows()
            break

    # tl.takeoff()

    # time.sleep(3)

    # tl.land()

    # quit()
