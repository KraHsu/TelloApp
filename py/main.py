from typing import *
from djitellopy import Tello
from tcpServer import TCPServer
import time


class FakeTello:
    def connect(self):
        print("connect")

    def takeoff(self):
        print("takeoff")

    def land(self):
        print("land")

    def send_rc_control(self, *x):
        print(*x)


if __name__ == "__main__":
    tl = Tello()
    # tl = FakeTello()

    tl.connect()

    # tl.takeoff()

    # time.sleep(3)

    # tl.land()

    # quit()

    server = TCPServer(host="0.0.0.0", port=8888)

    def start_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        tl.takeoff()
        time.sleep(3)
        return {}

    def key_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        front = payload.get("Front", False)
        back = payload.get("Back", False)
        left = payload.get("Left", False)
        right = payload.get("Right", False)
        up = payload.get("Up", False)
        down = payload.get("Down", False)

        tl.send_rc_control(
            (right - left) * 20, (front - back) * 20, (up - down) * 20, 0
        )

        time.sleep(0.01)
        return {}

    def stop_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        tl.land()
        time.sleep(3)
        server.stop()
        return {}

    server.register_handler("start", start_handler)
    server.register_handler("key", key_handler)
    server.register_handler("stop", stop_handler)

    server.start()
