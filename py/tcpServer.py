import socket
import json
import signal
import sys
import threading
from typing import Dict, Any, Callable, List, Tuple, Optional, Union


# 定义请求处理器类型 - 接收payload参数并返回响应对象
HandlerType = Callable[[Dict[str, Any]], Dict[str, Any]]


class TCPServer:
    def __init__(self, host: str = "localhost", port: int = 8888) -> None:
        """
        初始化TCP服务器

        参数:
        host: 服务器主机地址
        port: 服务器端口
        """
        self.host: str = host
        self.port: int = port
        self.server_socket: Optional[socket.socket] = None
        self.running: bool = False
        self.clients: List[socket.socket] = []
        # 命令处理器字典，用于存储注册的处理器
        self.handlers: Dict[str, HandlerType] = {}

        # 注册默认处理器
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认的命令处理器"""
        self.register_handler("ping", self._ping_handler)

    def _ping_handler(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """内置的ping处理器"""
        return {"status": "pong"}

    def register_handler(self, cmd: str, handler: HandlerType) -> None:
        """
        注册一个命令处理器

        参数:
        cmd: 命令名称
        handler: 处理器函数，接收payload并返回响应
        """
        self.handlers[cmd] = handler
        print(f"注册处理器: {cmd}")

    def unregister_handler(self, cmd: str) -> bool:
        """
        注销一个命令处理器

        参数:
        cmd: 要注销的命令名称

        返回:
        bool: 是否成功注销
        """
        if cmd in self.handlers:
            del self.handlers[cmd]
            print(f"注销处理器: {cmd}")
            return True
        return False

    def start(self) -> None:
        """启动TCP服务器"""
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

        print(f"服务器启动，监听于 {self.host}:{self.port}")
        print(
            f"已注册的处理器: {', '.join(self.handlers.keys()) if self.handlers else '无'}"
        )

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

    def handle_client(
        self, client_socket: socket.socket, client_address: Tuple[str, int]
    ) -> None:
        """处理客户端连接和数据交换"""
        try:
            while self.running:
                # 接收数据
                data = client_socket.recv(4096)
                if not data:
                    break

                try:
                    # 解析收到的JSON数据
                    request = json.loads(data.decode("utf-8"))
                    # print(f"收到请求: {request}")

                    # 处理请求
                    response = self.process_request(request)

                    # 发送响应
                    client_socket.sendall(json.dumps(response).encode("utf-8"))

                except json.JSONDecodeError:
                    # JSON解析错误
                    error_response = {
                        "code": "error",
                        "payload": {"message": "无效的JSON格式"},
                    }
                    client_socket.sendall(json.dumps(error_response).encode("utf-8"))

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

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理客户端请求并返回响应"""
        # 检查请求格式

        if "cmd" not in request or "payload" not in request:
            return {
                "code": "error",
                "payload": {"message": "请求格式错误，缺少必要字段"},
            }

        cmd = request["cmd"]
        payload = request["payload"]

        # 检查命令是否已注册
        if cmd in self.handlers:
            try:
                # 调用注册的处理器
                result = self.handlers[cmd](payload)
                return {"code": "success", "payload": result}
            except Exception as e:
                # 处理器执行出错
                return {
                    "code": "error",
                    "payload": {"message": f"处理请求时出错: {str(e)}"},
                }
        else:
            # 未注册的命令
            return {"code": "error", "payload": {"message": f"未知命令: {cmd}"}}

    def broadcast(self, message: Union[str, Dict[str, Any]]) -> None:
        """向所有客户端广播消息"""
        if isinstance(message, dict):
            message = json.dumps(message)

        message_bytes = message.encode("utf-8") if isinstance(message, str) else message

        disconnected_clients: List[socket.socket] = []

        for client in self.clients:
            try:
                client.sendall(message_bytes)
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

    def signal_handler(self, sig: int, frame: Any) -> None:
        """处理Ctrl+C信号，优雅关闭服务器"""
        print("\n正在关闭服务器...")
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """停止服务器并清理资源"""
        self.running = False

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


# 示例用法
if __name__ == "__main__":
    # 创建服务器实例
    server = TCPServer(host="0.0.0.0", port=8888)

    # 定义自定义处理器
    def echo_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        """回声处理器"""
        return {"message": payload.get("message", "")}

    def add_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        """加法处理器"""
        a = payload.get("a", 0)
        b = payload.get("b", 0)
        result = a + b
        return {"result": result}

    def key_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        front = payload.get("Front", False)
        back = payload.get("Back", False)
        left = payload.get("Left", False)
        right = payload.get("Right", False)
        up = payload.get("Up", False)
        down = payload.get("Down", False)
        return {}

    # 注册自定义处理器
    server.register_handler("echo", echo_handler)
    server.register_handler("add", add_handler)
    server.register_handler("key", key_handler)

    # 启动服务器
    server.start()
