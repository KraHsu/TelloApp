using Godot;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Tello.script;

// 用于请求的类
public record Request(string cmd, object payload);

// 用于响应的类
public record Response(string code, object payload);

// TCP客户端封装类
public class TcpClient(string host = "localhost", int port = 8888) : IDisposable {
    private System.Net.Sockets.TcpClient _tcpClient = new();
    private NetworkStream                _stream;
    public  bool                         IsConnected { get; private set; } = false;

    // 构造函数

    // 连接到服务器
    public async Task<bool> ConnectAsync() {
        try {
            if (IsConnected)
                return true;

            await _tcpClient.ConnectAsync(host, port);
            _stream     = _tcpClient.GetStream();
            IsConnected = true;
            GD.Print($"已连接到服务器 {host}:{port}");
            return true;
        }
        catch (Exception ex) {
            GD.Print($"连接服务器失败: {ex.Message}");
            IsConnected = false;
            return false;
        }
    }

    // 发送请求并获取响应
    private readonly JsonSerializerOptions options = new() {
        PropertyNameCaseInsensitive = true
    };

    public async Task<Response> SendRequestAsync(Request request) {
        if (!IsConnected) {
            var connected = await ConnectAsync();
            if (!connected)
                return null;
        }

        try {
            // 序列化请求
            var jsonRequest  = JsonSerializer.Serialize(request);
            var requestBytes = Encoding.UTF8.GetBytes(jsonRequest);

            // 发送请求
            await _stream.WriteAsync(requestBytes);
            GD.Print($"已发送请求: {jsonRequest}");

            // 接收响应
            var responseBuffer = new byte[4096];
            var bytesRead      = await _stream.ReadAsync(responseBuffer);
            var jsonResponse   = Encoding.UTF8.GetString(responseBuffer, 0, bytesRead);

            GD.Print($"收到响应: {jsonResponse}");

            // 反序列化响应

            Response response = JsonSerializer.Deserialize<Response>(jsonResponse, options);

            // 处理错误响应
            if (response.code != "error") return response;
            try {
                JsonElement jsonElement  = (JsonElement)response.payload;
                var         errorMessage = jsonElement.GetProperty("message").GetString();
                GD.PrintErr($"服务器错误: {errorMessage}");
            }
            catch {
                GD.PrintErr("服务器返回了一个错误，但无法解析错误消息");
            }

            return response;
        }
        catch (Exception ex) {
            GD.PrintErr($"通信错误: {ex.Message}");
            IsConnected = false;
            return null;
        }
    }

    // 发送Echo请求
    public async Task<string> SendEchoAsync(string message) {
        Request request = new(cmd: "echo", payload: new { message });

        Response response = await SendRequestAsync(request);
        if (response is not { code: "success" }) return null;
        try {
            var jsonElement = (JsonElement)response.payload;
            return jsonElement.GetProperty("message").GetString();
        }
        catch {
            return "无法解析响应消息";
        }

        return null;
    }

    // 发送加法请求
    public async Task<int?> SendAddAsync(int a, int b) {
        Request request = new(cmd: "add", payload: new { a, b });

        Response response = await SendRequestAsync(request);
        if (response is not { code: "success" }) return null;
        try {
            JsonElement jsonElement = (JsonElement)response.payload;
            return jsonElement.GetProperty("result").GetInt32();
        }
        catch {
            return null;
        }

        return null;
    }

    // 发送Ping请求
    public async Task<string> SendPingAsync() {
        Request request = new(cmd: "ping", payload: new { });

        Response response = await SendRequestAsync(request);
        if (response is not { code: "success" }) return null;
        try {
            JsonElement jsonElement = (JsonElement)response.payload;
            return jsonElement.GetProperty("status").GetString();
        }
        catch {
            return null;
        }
    }

    // 断开连接
    public void Disconnect() {
        if (_stream != null) {
            _stream.Close();
            _stream = null;
        }

        if (_tcpClient != null) {
            _tcpClient.Close();
            _tcpClient = null;
        }

        IsConnected = false;
        GD.Print("已断开与服务器的连接");
    }

    // 实现IDisposable接口
    public void Dispose() {
        Disconnect();
    }
}
