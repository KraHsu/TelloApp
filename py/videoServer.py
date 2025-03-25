import socket
import signal
import sys
import threading
import cv2
import numpy as np
import base64
import time
import struct

# 定义帧数据分隔符
FRAME_DELIMITER = b"\r\n\r\n"  # 使用四个字节的分隔符


class VideoStreamServer:
    def __init__(self, video_path, host="localhost", port=8888, fps=30, loop=True):
        self.video_path = video_path
        self.host = host
        self.port = port
        self.fps = fps
        self.loop = loop  # 循环播放标志
        self.server_socket = None
        self.running = False
        self.clients = []
        self.video_thread = None

    def start(self):
        # 创建 socket 对象
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置 socket 选项，允许地址重用
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定地址和端口
        self.server_socket.bind((self.host, self.port))
        # 开始监听
        self.server_socket.listen(5)
        self.running = True

        # 设置信号处理，捕获 Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        print(f"视频流服务器启动，监听于 {self.host}:{self.port}")

        # 启动视频处理线程
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        try:
            # 循环等待客户端连接
            while self.running:
                try:
                    # 使用超时机制，使循环能够检查running标志
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    self.clients.append(client_socket)
                    print(f"客户端连接: {client_address}")

                    # 设置套接字选项以提高性能
                    client_socket.setsockopt(
                        socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024
                    )  # 4MB发送缓冲区

                    # 为每个客户端创建一个新线程处理
                    client_thread = threading.Thread(
                        target=self.handle_client, args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    # 超时，继续循环
                    continue

        except Exception as e:
            print(f"服务器异常: {e}")
        finally:
            self.stop()

    def process_video(self):
        """处理视频并发送到所有连接的客户端"""
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {self.video_path}")
                return

            # 获取视频信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算传输分辨率
            max_width = 1280
            stream_width = width
            stream_height = height

            if width > max_width:
                # 等比例缩放
                scale_factor = max_width / width
                stream_width = int(width * scale_factor)
                stream_height = int(height * scale_factor)

            # 发送视频信息给所有客户端
            video_info = f"INFO:{width},{height},{fps},{frame_count},{self.fps},{1 if self.loop else 0},{stream_width},{stream_height}".encode(
                "utf-8"
            )
            self.broadcast(video_info + FRAME_DELIMITER)

            frame_interval = 1.0 / self.fps

            while self.running:
                start_time = time.time()

                # 读取一帧视频
                ret, frame = cap.read()
                if not ret:
                    if self.loop:
                        # 视频结束，重新开始播放
                        print("视频播放结束，重新播放")
                        cap.release()  # 释放当前视频捕获对象
                        cap = cv2.VideoCapture(self.video_path)  # 重新打开视频文件
                        if not cap.isOpened():
                            print(f"无法重新打开视频文件: {self.video_path}")
                            break
                        ret, frame = cap.read()  # 读取第一帧
                        if not ret:
                            print("无法读取视频帧，退出播放")
                            break
                    else:
                        # 不循环播放，结束视频处理线程
                        print("视频播放结束")
                        break

                # 保持原始分辨率，或者根据需要调整到更高的分辨率
                # 如果原始分辨率太大，可以适当缩小，但保持较高的分辨率
                original_height, original_width = frame.shape[:2]
                max_width = 1280  # 增大最大宽度

                if original_width > max_width:
                    # 等比例缩放
                    scale_factor = max_width / original_width
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))

                # 将帧编码为JPEG格式，提高质量
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 将图像转换为base64字符串
                jpg_as_base64 = base64.b64encode(buffer)

                # 创建帧数据包：帧尺寸(4字节) + 帧数据 + 分隔符
                frame_size = len(jpg_as_base64)
                frame_header = struct.pack(">I", frame_size)  # 4字节无符号整数，大端序
                frame_packet = frame_header + jpg_as_base64 + FRAME_DELIMITER

                # 发送到所有客户端
                self.broadcast(frame_packet)

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()

        except Exception as e:
            print(f"视频处理异常: {e}")

    def broadcast(self, message):
        """向所有客户端广播消息"""
        disconnected_clients = []

        for client in self.clients:
            try:
                client.sendall(message)
            except:
                # 标记断开连接的客户端
                disconnected_clients.append(client)

        # 移除断开连接的客户端
        for client in disconnected_clients:
            if client in self.clients:
                self.clients.remove(client)
                try:
                    client.close()
                except:
                    pass
                print(f"客户端断开连接")

    def handle_client(self, client_socket, client_address):
        """处理客户端连接和命令"""
        try:
            # 发送初始PING以确认连接
            ping_message = b"PING" + FRAME_DELIMITER
            client_socket.sendall(ping_message)

            while self.running:
                # 接收客户端可能发送的命令
                try:
                    client_socket.settimeout(0.5)  # 设置短超时以便定期检查running标志
                    data = client_socket.recv(1024)
                    if not data:
                        break

                    # 处理简单的命令
                    command = data.decode("utf-8").strip()
                    if command == "PING":
                        client_socket.sendall(b"PONG" + FRAME_DELIMITER)
                    elif command == "INFO":
                        # 重新发送视频信息
                        cap = cv2.VideoCapture(self.video_path)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()

                        video_info = f"INFO:{width},{height},{fps},{frame_count},{self.fps},{1 if self.loop else 0}".encode(
                            "utf-8"
                        )
                        client_socket.sendall(video_info + FRAME_DELIMITER)

                except socket.timeout:
                    # 超时，继续循环
                    continue

        except Exception as e:
            print(f"处理客户端 {client_address} 时出错: {e}")
        finally:
            # 关闭与客户端的连接
            try:
                client_socket.close()
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
                print(f"客户端断开连接: {client_address}")
            except:
                pass

    def signal_handler(self, sig, frame):
        """处理Ctrl+C信号，优雅关闭服务器"""
        print("\n正在关闭服务器...")
        self.stop()
        sys.exit(0)

    def stop(self):
        """停止服务器并清理资源"""
        self.running = False

        # 关闭所有客户端连接
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()

        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print("服务器已关闭")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频流服务器")
    parser.add_argument("--video", type=str, default="demo.mp4", help="视频文件路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8888, help="服务器端口")
    parser.add_argument("--fps", type=int, default=30, help="发送帧率")
    parser.add_argument("--no-loop", action="store_true", help="不循环播放视频")

    args = parser.parse_args()

    # 创建并启动视频流服务器
    server = VideoStreamServer(
        args.video, args.host, args.port, args.fps, not args.no_loop
    )
    server.start()
