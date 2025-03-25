import socket
import json
import signal
import sys
import threading

class TCPServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []
        
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
        
        print(f"服务器启动，监听于 {self.host}:{self.port}")
        
        try:
            # 循环等待客户端连接
            while self.running:
                try:
                    # 使用超时机制，使循环能够检查running标志
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()
                    self.clients.append(client_socket)
                    print(f"客户端连接: {client_address}")
                    
                    # 为每个客户端创建一个新线程处理
                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    # 超时，继续循环
                    continue
                    
        except Exception as e:
            print(f"服务器异常: {e}")
        finally:
            self.stop()
    
    def handle_client(self, client_socket, client_address):
        """处理客户端连接和数据交换"""
        try:
            while self.running:
                # 接收数据
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    # 解析收到的JSON数据
                    request = json.loads(data.decode('utf-8'))
                    print(f"收到请求: {request}")
                    
                    # 处理请求
                    response = self.process_request(request)
                    
                    # 发送响应
                    client_socket.sendall(json.dumps(response).encode('utf-8'))
                    
                except json.JSONDecodeError:
                    # JSON解析错误
                    error_response = {"code": "error", "payload": {"message": "无效的JSON格式"}}
                    client_socket.sendall(json.dumps(error_response).encode('utf-8'))
                    
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
    
    def process_request(self, request):
        """处理客户端请求并返回响应"""
        # 检查请求格式
        if "cmd" not in request or "payload" not in request:
            return {"code": "error", "payload": {"message": "请求格式错误，缺少必要字段"}}
        
        cmd = request["cmd"]
        payload = request["payload"]
        
        # 根据命令类型处理请求
        if cmd == "echo":
            # 简单的回声服务
            return {"code": "success", "payload": {"message": payload.get("message", "")}}
        
        elif cmd == "add":
            # 加法操作示例
            try:
                a = payload.get("a", 0)
                b = payload.get("b", 0)
                result = a + b
                return {"code": "success", "payload": {"result": result}}
            except:
                return {"code": "error", "payload": {"message": "计算错误"}}
        
        elif cmd == "ping":
            # 简单的心跳检测
            return {"code": "success", "payload": {"status": "pong"}}
            
        else:
            # 未知命令
            return {"code": "error", "payload": {"message": f"未知命令: {cmd}"}}
    
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
    # 创建并启动TCP服务器
    server = TCPServer()
    server.start()