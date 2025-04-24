import socket
import sys
from djitellopy import Tello
import time

tello = Tello()
tello.connect()
tello.streamon()
tello.set_video_direction(tello.CAMERA_DOWNWARD)
print(tello.get_udp_video_address())
# --- 配置 ---
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 11111
BUFFER_SIZE = 4096
TIMEOUT_SECONDS = 5.0
# -------------

sock = None
try:
    # socket.SOCK_DGRAM 表示使用 UDP 协议
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Socket created.")

    server_address = (LISTEN_IP, LISTEN_PORT)
    sock.bind(server_address)
    print(f"Socket bound to {LISTEN_IP}:{LISTEN_PORT}")

    # --- 设置超时 ---
    sock.settimeout(TIMEOUT_SECONDS)
    print(f"Socket timeout set to {TIMEOUT_SECONDS} seconds.")
    # ----------------

    print("Listening for incoming UDP packets...")

    while True:
        try:
            data, address = sock.recvfrom(BUFFER_SIZE)
            print(f"\nReceived {len(data)} bytes from {address}:")
            print(f"  Data (first 100 bytes): {data[:100]}")

        except socket.timeout:
            print(f".", end="", flush=True)
            time.sleep(0.1)
            continue
        except Exception as e:
            print(f"Error during recvfrom: {e}")


except socket.error as e:
    print(f"Socket error: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\nCtrl+C detected. Exiting...")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if sock:
        print("Closing socket.")
        sock.close()
    tello.streamoff()
    tello.end()
