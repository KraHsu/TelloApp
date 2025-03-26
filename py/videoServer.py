import socket
import signal
import sys
import threading
import cv2
import numpy as np
import base64
import time
import struct
import abc
from typing import Tuple, List, Dict, Optional, Union, Any, TypeVar, cast

from djitellopy import Tello

# 定义帧数据分隔符
FRAME_DELIMITER: bytes = b"\r\n\r\n"  # 使用四个字节的分隔符


class FrameSource(abc.ABC):
    """
    帧源抽象基类，定义了获取帧的接口
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.frame_info: Dict[str, int] = {
            "width": 0,
            "height": 0,
            "fps": 30,
            "frame_count": 0,
        }

    @abc.abstractmethod
    def start(self) -> bool:
        """启动帧源"""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """停止帧源"""
        pass

    @abc.abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取下一帧，返回(success, frame)元组"""
        pass

    def get_frame_info(self) -> Dict[str, int]:
        """获取帧源信息"""
        return self.frame_info


class VideoFileSource(FrameSource):
    """
    从视频文件获取帧的源
    """

    def __init__(
        self, video_path: str, target_fps: int = 30, loop: bool = True
    ) -> None:
        super().__init__()
        self.video_path: str = video_path
        self.target_fps: int = target_fps
        self.loop: bool = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.original_fps: float = 0.0

    def start(self) -> bool:
        """打开视频文件并初始化"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频文件: {self.video_path}")

        # 获取视频信息
        self.frame_info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_info["fps"] = self.target_fps
        self.frame_info["frame_count"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.running = True
        return True

    def stop(self) -> None:
        """关闭视频源"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取下一帧"""
        if not self.running or not self.cap:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                # 视频结束，重新开始播放
                if self.cap:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    return False, None
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
            else:
                return False, None

        return True, frame


class WebcamSource(FrameSource):
    """
    从摄像头获取帧的源
    """

    def __init__(self, camera_id: int = 0, target_fps: int = 30) -> None:
        super().__init__()
        self.camera_id: int = camera_id
        self.frame_info["fps"] = target_fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"无法打开摄像头ID: {self.camera_id}")

        # 获取视频信息
        self.frame_info["width"] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_info["height"] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_info["frame_count"] = -1  # 直播流没有总帧数

        self.running = True
        return True

    def stop(self) -> None:
        """关闭视频源"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """获取下一帧"""
        if not self.running or not self.cap:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame


class StaticImageSource(FrameSource):
    """
    使用静态图像作为帧源
    """

    def __init__(self, image_path: str, target_fps: int = 30) -> None:
        super().__init__()
        self.image_path: str = image_path
        self.frame_info["fps"] = target_fps
        self.image: Optional[np.ndarray] = None

    def start(self) -> bool:
        """加载静态图像"""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise Exception(f"无法加载图像: {self.image_path}")

        # 获取图像信息
        h, w = self.image.shape[:2]
        self.frame_info["width"] = w
        self.frame_info["height"] = h
        self.frame_info["frame_count"] = 1  # 静态图像只有一帧

        self.running = True
        return True

    def stop(self) -> None:
        """停止图像源"""
        self.running = False
        self.image = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """返回静态图像"""
        if not self.running or self.image is None:
            return False, None

        return True, self.image.copy()


class VideoStreamServer:
    def __init__(
        self,
        frame_source: FrameSource,
        host: str = "localhost",
        port: int = 8888,
        max_width: int = 1280,
    ) -> None:
        """
        初始化视频流服务器

        参数:
        frame_source: FrameSource的一个实例，用于获取帧
        host: 服务器主机地址
        port: 服务器端口
        max_width: 传输的最大宽度
        """
        self.frame_source: FrameSource = frame_source
        self.host: str = host
        self.port: int = port
        self.max_width: int = max_width
        self.server_socket: Optional[socket.socket] = None
        self.running: bool = False
        self.clients: List[socket.socket] = []
        self.video_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动视频流服务器"""
        # 启动帧源
        try:
            self.frame_source.start()
        except Exception as e:
            print(f"无法启动帧源: {e}")
            return

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
                    if self.server_socket:
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
                            target=self.handle_client,
                            args=(client_socket, client_address),
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

    def process_video(self) -> None:
        """处理视频并发送到所有连接的客户端"""
        try:
            # 获取帧源信息
            frame_info = self.frame_source.get_frame_info()
            width = frame_info["width"]
            height = frame_info["height"]
            fps = frame_info["fps"]
            frame_count = frame_info["frame_count"]

            # 计算传输分辨率
            stream_width = width
            stream_height = height

            if width > self.max_width:
                # 等比例缩放
                scale_factor = self.max_width / width
                stream_width = int(width * scale_factor)
                stream_height = int(height * scale_factor)

            # 发送视频信息给所有客户端
            is_loop = 0
            if isinstance(self.frame_source, VideoFileSource):
                is_loop = 1 if self.frame_source.loop else 0

            video_info = f"INFO:{width},{height},{fps},{frame_count},{fps},{is_loop},{stream_width},{stream_height}".encode(
                "utf-8"
            )
            self.broadcast(video_info + FRAME_DELIMITER)

            frame_interval = 1.0 / fps

            while self.running:
                start_time = time.time()

                # 从帧源获取一帧视频
                ret, frame = self.frame_source.get_frame()
                if not ret or frame is None:
                    print("无法获取帧，退出处理")
                    break

                # 调整分辨率
                original_height, original_width = frame.shape[:2]
                if original_width > self.max_width:
                    # 等比例缩放
                    scale_factor = self.max_width / original_width
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))

                # 将帧编码为JPEG格式，提高质量
                result, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                if not result:
                    print("无法编码帧")
                    continue

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

        except Exception as e:
            print(f"视频处理异常: {e}")
        finally:
            if self.running:
                self.stop()

    def broadcast(self, message: bytes) -> None:
        """向所有客户端广播消息"""
        disconnected_clients: List[socket.socket] = []

        for client in self.clients:
            try:
                client.sendall(message)
            except Exception:
                # 标记断开连接的客户端
                disconnected_clients.append(client)

        # 移除断开连接的客户端
        for client in disconnected_clients:
            if client in self.clients:
                self.clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass
                print(f"客户端断开连接")

    def handle_client(
        self, client_socket: socket.socket, client_address: Tuple[str, int]
    ) -> None:
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
                        frame_info = self.frame_source.get_frame_info()
                        width = frame_info["width"]
                        height = frame_info["height"]
                        fps = frame_info["fps"]
                        frame_count = frame_info["frame_count"]

                        is_loop = 0
                        if isinstance(self.frame_source, VideoFileSource):
                            is_loop = 1 if self.frame_source.loop else 0

                        video_info = f"INFO:{width},{height},{fps},{frame_count},{fps},{is_loop}".encode(
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
            except Exception:
                pass

    def signal_handler(self, sig: int, frame: Any) -> None:
        """处理Ctrl+C信号，优雅关闭服务器"""
        print("\n正在关闭服务器...")
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """停止服务器并清理资源"""
        self.running = False

        # 停止帧源
        self.frame_source.stop()

        # 关闭所有客户端连接
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()

        # 关闭服务器套接字
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        print("服务器已关闭")


class TelloSource(FrameSource):
    """
    使用Tello作为帧源
    """

    def __init__(self, tl: Tello) -> None:
        super().__init__()

        self.frame_info["fps"] = 30

        tl.streamon()

    def start(self) -> bool:

        self.frame_read = tl.get_frame_read()
        self.frame_cnt = 1

        frame = self.frame_read.frame

        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = self.image.shape[:2]
        self.frame_info["width"] = w
        self.frame_info["height"] = h

        self.frame_cnt = self.frame_cnt + 1
        self.frame_info["frame_count"] = self.frame_cnt

        self.running = True
        return True

    def stop(self) -> None:
        """停止图像源"""
        self.running = False
        self.image = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """返回静态图像"""
        if not self.running or self.image is None:
            return False, None
        
        frame = self.frame_read.frame
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return True, self.image.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频流服务器")
    parser.add_argument(
        "--source",
        type=str,
        choices=["video", "webcam", "image", "tello"],
        default="video",
        help="帧源类型: video, webcam, image 或 tello 无人机",
    )
    parser.add_argument("--video", type=str, default="demo.mp4", help="视频文件路径")
    parser.add_argument("--image", type=str, default="image.jpg", help="静态图像路径")
    parser.add_argument("--webcam", type=int, default=0, help="摄像头ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8888, help="服务器端口")
    parser.add_argument("--fps", type=int, default=30, help="发送帧率")
    parser.add_argument("--no-loop", action="store_true", help="不循环播放视频")
    parser.add_argument("--max-width", type=int, default=1280, help="传输的最大宽度")

    args = parser.parse_args()

    # 根据参数创建帧源
    frame_source: Optional[FrameSource] = None
    if args.source == "video":
        frame_source = VideoFileSource(args.video, args.fps, not args.no_loop)
    elif args.source == "webcam":
        frame_source = WebcamSource(args.webcam, args.fps)
    elif args.source == "image":
        frame_source = StaticImageSource(args.image, args.fps)
    elif args.source == "tello":
        tl = Tello()
        tl.connect()
        tl.streamoff()
        frame_source = TelloSource(tl)

    else:
        print(f"不支持的帧源类型: {args.source}")
        sys.exit(1)

    # 创建并启动视频流服务器
    server = VideoStreamServer(frame_source, args.host, args.port, args.max_width)
    server.start()

    if tl:
        tl.streamoff()
