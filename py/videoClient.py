import socket
import threading
import cv2
import numpy as np
import base64
import time
import signal
import sys
import struct

# 定义帧数据分隔符
FRAME_DELIMITER = b"\r\n\r\n"  # 使用四个字节的分隔符


class VideoStreamClient:
    def __init__(self, server_host="localhost", server_port=8888):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.video_info = None

    def connect(self):
        """连接到服务器"""
        try:
            # 创建 socket 对象
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 连接服务器
            self.client_socket.connect((self.server_host, self.server_port))
            self.running = True

            # 设置信号处理，捕获 Ctrl+C
            signal.signal(signal.SIGINT, self.signal_handler)

            print(f"已连接到服务器: {self.server_host}:{self.server_port}")

            # 发送INFO请求获取视频信息
            self.client_socket.sendall(b"INFO")

            # 设置TCP缓冲区大小以提高性能
            self.client_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024
            )  # 4MB接收缓冲区

            # 创建接收线程
            recv_thread = threading.Thread(target=self.receive_frames)
            recv_thread.daemon = True
            recv_thread.start()

            # 创建显示线程
            display_thread = threading.Thread(target=self.display_frames)
            display_thread.daemon = True
            display_thread.start()

            # 创建心跳线程
            heartbeat_thread = threading.Thread(target=self.heartbeat)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()

            # 保持主线程运行
            while self.running:
                time.sleep(0.1)

        except Exception as e:
            print(f"连接服务器失败: {e}")
            self.disconnect()

    def heartbeat(self):
        """定期发送心跳包保持连接"""
        try:
            while self.running:
                try:
                    self.client_socket.sendall(b"PING")
                    time.sleep(5)  # 每5秒发送一次心跳
                except:
                    print("发送心跳失败，可能已断开连接")
                    self.running = False
                    break
        except:
            pass

    def receive_frames(self):
        """接收视频帧线程"""
        buffer = b""

        try:
            while self.running:
                # 接收数据
                data = self.client_socket.recv(
                    1024 * 1024
                )  # 增大接收缓冲区以处理较大的图像
                if not data:
                    print("服务器已断开连接")
                    self.running = False
                    break

                # 添加到缓冲区
                buffer += data

                # 处理所有完整的帧
                while True:
                    # 查找分隔符位置
                    delimiter_pos = buffer.find(FRAME_DELIMITER)
                    if delimiter_pos == -1:
                        break  # 没有找到完整的帧

                    # 提取一个完整的帧
                    frame_data = buffer[:delimiter_pos]
                    buffer = buffer[delimiter_pos + len(FRAME_DELIMITER) :]

                    # 处理接收到的数据
                    self.process_data(frame_data)

        except Exception as e:
            print(f"接收数据异常: {e}")
        finally:
            self.disconnect()

    def process_data(self, data):
        """处理接收到的数据"""
        try:
            # 检查是否是文本命令响应
            if data.startswith(b"PONG"):
                # 心跳响应
                return
            elif data.startswith(b"INFO:"):
                # 视频信息
                info_str = data[5:].decode("utf-8")
                info_parts = info_str.split(",")
                if len(info_parts) >= 6:
                    self.video_info = {
                        "width": int(info_parts[0]),  # 原始宽度
                        "height": int(info_parts[1]),  # 原始高度
                        "fps": float(info_parts[2]),
                        "frame_count": int(info_parts[3]),
                        "stream_fps": float(info_parts[4]),
                        "loop": bool(int(info_parts[5])),
                    }

                    # 如果有传输分辨率信息
                    if len(info_parts) >= 8:
                        self.video_info["stream_width"] = int(info_parts[6])
                        self.video_info["stream_height"] = int(info_parts[7])
                    print(f"视频信息: {self.video_info}")
                return
            elif data.startswith(b"PING"):
                # 服务器心跳
                self.client_socket.sendall(b"PONG" + FRAME_DELIMITER)
                return

            # 如果不是命令响应，则处理为视频帧
            # 帧格式: 帧尺寸(4字节) + 帧数据
            if len(data) < 4:
                return  # 数据太短，不是有效帧

            # 提取帧大小
            frame_size = struct.unpack(">I", data[:4])[0]

            # 检查数据长度
            if len(data) - 4 < frame_size:
                return  # 数据不完整

            # 提取帧数据
            frame_data = data[4 : 4 + frame_size]

            # 解码base64图像
            img_bytes = base64.b64decode(frame_data)

            # 解码JPEG图像
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is not None:
                # 更新最新帧
                with self.frame_lock:
                    self.latest_frame = frame

        except Exception as e:
            print(f"处理数据时出错: {e}")

    def display_frames(self):
        """显示视频帧线程"""
        try:
            # 创建一个可调整大小的窗口
            cv2.namedWindow("视频流", cv2.WINDOW_NORMAL)

            # 初始窗口大小设置
            window_width = 800
            window_height = 600

            # 如果有视频信息，根据原始分辨率设置合适的窗口大小
            if self.video_info:
                aspect_ratio = self.video_info.get("width", 16) / self.video_info.get(
                    "height", 9
                )
                window_height = min(720, self.video_info.get("height", 600))
                window_width = int(window_height * aspect_ratio)

            # 设置窗口大小
            cv2.resizeWindow("视频流", window_width, window_height)

            while self.running:
                # 获取最新的帧
                with self.frame_lock:
                    frame = self.latest_frame

                if frame is not None:
                    # 显示帧，保持图像的原始质量
                    cv2.imshow("视频流", frame)

                    # 检测按键
                    key = cv2.waitKey(1) & 0xFF

                    # 按 'q' 退出
                    if key == ord("q"):
                        self.running = False
                        break
                    # 按 'f' 切换全屏
                    elif key == ord("f"):
                        if (
                            cv2.getWindowProperty("视频流", cv2.WND_PROP_FULLSCREEN)
                            != cv2.WINDOW_FULLSCREEN
                        ):
                            cv2.setWindowProperty(
                                "视频流", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                            )
                        else:
                            cv2.setWindowProperty(
                                "视频流", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                            )
                            cv2.resizeWindow("视频流", window_width, window_height)
                else:
                    # 如果没有帧，短暂休眠
                    time.sleep(0.01)

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"显示视频异常: {e}")
            self.running = False

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号，优雅关闭客户端"""
        print("\n正在关闭客户端...")
        self.disconnect()
        sys.exit(0)

    def disconnect(self):
        """断开与服务器的连接"""
        self.running = False

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        print("已断开与服务器的连接")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频流客户端")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8888, help="服务器端口")

    args = parser.parse_args()

    # 创建并连接视频流客户端
    client = VideoStreamClient(args.host, args.port)
    client.connect()
