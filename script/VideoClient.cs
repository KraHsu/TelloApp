using Godot;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tello.script;

/// <summary>
/// 视频信息类，存储从服务器接收的视频元数据
/// </summary>
public class VideoInfo {
    public int    OriginalWidth  { get; set; }
    public int    OriginalHeight { get; set; }
    public double OriginalFps    { get; set; }
    public int    FrameCount     { get; set; }
    public double StreamFps      { get; set; }
    public bool   IsLooping      { get; set; }
    public int    StreamWidth    { get; set; }
    public int    StreamHeight   { get; set; }
}

/// <summary>
/// 帧数据类，包含接收到的视频帧数据
/// </summary>
public class FrameData {
    public byte[] Data { get; set; }
}

/// <summary>
/// 视频流客户端类，用于连接视频流服务器并接收视频数据
/// </summary>
public class VideoClient : IDisposable {
    private readonly string                       _host;
    private readonly int                          _port;
    private          System.Net.Sockets.TcpClient _client;
    private          NetworkStream                _stream;
    private          CancellationTokenSource      _cancellationTokenSource;
    private readonly byte[]                       _frameDelimiter = new byte[] { 13, 10, 13, 10 }; // \r\n\r\n
    private          bool                         _isConnected    = false;
    private          bool                         _isDisposed     = false;

    // 事件定义
    public event EventHandler<VideoInfo> OnVideoInfoReceived;
    public event EventHandler<FrameData> OnFrameReceived;
    public event EventHandler<string>    OnLog;
    public event EventHandler            OnConnected;
    public event EventHandler            OnDisconnected;
    public event EventHandler<Exception> OnError;

    /// <summary>
    /// 初始化视频流客户端
    /// </summary>
    /// <param name="host">服务器主机地址</param>
    /// <param name="port">服务器端口</param>
    public VideoClient(string host = "localhost", int port = 8888) {
        _host   = host;
        _port   = port;
        _client = null;
        _stream = null;
    }

    /// <summary>
    /// 连接到视频流服务器
    /// </summary>
    /// <returns>连接是否成功</returns>
    public async Task<bool> ConnectAsync() {
        if (_isConnected)
            return true;

        try {
            // 创建TCP客户端并连接
            _client = new System.Net.Sockets.TcpClient();
            await _client.ConnectAsync(_host, _port);
            _stream = _client.GetStream();

            // 设置较大的接收缓冲区
            _client.ReceiveBufferSize = 4 * 1024 * 1024; // 4MB

            _isConnected = true;
            Log($"已连接到服务器: {_host}:{_port}");
            OnConnected?.Invoke(this, EventArgs.Empty);

            // 启动接收数据的任务
            _cancellationTokenSource = new CancellationTokenSource();
            _                        = Task.Run(() => ReceiveDataAsync(_cancellationTokenSource.Token));

            return true;
        }
        catch (Exception ex) {
            Log($"连接服务器失败: {ex.Message}");
            OnError?.Invoke(this, ex);
            return false;
        }
    }

    /// <summary>
    /// 断开与服务器的连接
    /// </summary>
    public void Disconnect() {
        if (!_isConnected)
            return;

        try {
            _cancellationTokenSource?.Cancel();
            _stream?.Close();
            _client?.Close();
            _isConnected = false;

            Log("已断开与服务器的连接");
            OnDisconnected?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex) {
            Log($"断开连接时出错: {ex.Message}");
            OnError?.Invoke(this, ex);
        }
    }

    /// <summary>
    /// 发送PING命令到服务器
    /// </summary>
    public async Task SendPingAsync() {
        if (!_isConnected)
            return;

        try {
            byte[] pingData = Encoding.UTF8.GetBytes("PING");
            await _stream.WriteAsync(pingData,        0, pingData.Length);
            await _stream.WriteAsync(_frameDelimiter, 0, _frameDelimiter.Length);
            await _stream.FlushAsync();
            Log("已发送PING命令");
        }
        catch (Exception ex) {
            Log($"发送PING命令时出错: {ex.Message}");
            OnError?.Invoke(this, ex);
        }
    }

    /// <summary>
    /// 请求视频信息
    /// </summary>
    public async Task RequestVideoInfoAsync() {
        if (!_isConnected)
            return;

        try {
            byte[] infoData = Encoding.UTF8.GetBytes("INFO");
            await _stream.WriteAsync(infoData,        0, infoData.Length);
            await _stream.WriteAsync(_frameDelimiter, 0, _frameDelimiter.Length);
            await _stream.FlushAsync();
            Log("已请求视频信息");
        }
        catch (Exception ex) {
            Log($"请求视频信息时出错: {ex.Message}");
            OnError?.Invoke(this, ex);
        }
    }

    /// <summary>
    /// 接收数据的异步方法
    /// </summary>
    private async Task ReceiveDataAsync(CancellationToken cancellationToken) {
        try {
            using MemoryStream bufferStream = new MemoryStream();
            byte[]             buffer       = new byte[8192]; // 读取缓冲区
            int                bytesRead;

            while (!cancellationToken.IsCancellationRequested && _isConnected) {
                bytesRead = await _stream.ReadAsync(buffer, 0, buffer.Length, cancellationToken);
                if (bytesRead == 0) {
                    // 连接已关闭
                    HandleDisconnect();
                    break;
                }

                // 将读取的数据追加到缓冲流中
                bufferStream.Write(buffer, 0, bytesRead);

                // 处理完整的消息包
                ProcessBufferStream(bufferStream);
            }
        }
        catch (OperationCanceledException) {
            // 正常取消操作，不需要处理
        }
        catch (Exception ex) {
            if (!cancellationToken.IsCancellationRequested) {
                Log($"接收数据时出错: {ex.Message}");
                OnError?.Invoke(this, ex);
                HandleDisconnect();
            }
        }
    }

    /// <summary>
    /// 处理缓冲流中的数据，提取完整的消息包
    /// </summary>
    private void ProcessBufferStream(MemoryStream bufferStream) {
        byte[] data     = bufferStream.ToArray();
        int    position = 0;
        int    delimiterIndex;

        // 查找所有完整的消息包
        while ((delimiterIndex = FindDelimiter(data, position)) != -1) {
            int length = delimiterIndex - position;
            if (length > 0) {
                byte[] packet = new byte[length];
                Array.Copy(data, position, packet, 0, length);
                ProcessPacket(packet);
            }

            position = delimiterIndex + _frameDelimiter.Length;
        }

        // 保留未处理的数据
        if (position < data.Length) {
            bufferStream.SetLength(0);
            bufferStream.Write(data, position, data.Length - position);
        }
        else {
            bufferStream.SetLength(0);
        }
    }

    /// <summary>
    /// 查找帧分隔符的位置
    /// </summary>
    private int FindDelimiter(byte[] data, int startIndex) {
        if (data.Length < _frameDelimiter.Length + startIndex)
            return -1;

        for (int i = startIndex; i <= data.Length - _frameDelimiter.Length; i++) {
            bool found = true;
            for (int j = 0; j < _frameDelimiter.Length; j++) {
                if (data[i + j] != _frameDelimiter[j]) {
                    found = false;
                    break;
                }
            }

            if (found)
                return i;
        }

        return -1;
    }

    /// <summary>
    /// 处理接收到的完整数据包
    /// </summary>
    private void ProcessPacket(byte[] packet) {
        try {
            // 检查是否是文本命令
            string textCommand = Encoding.UTF8.GetString(packet);

            if (textCommand == "PONG") {
                Log("收到PONG响应");
                return;
            }

            // 检查是否是INFO消息
            if (textCommand.StartsWith("INFO:")) {
                ProcessVideoInfo(textCommand);
                return;
            }

            // 如果是二进制数据（视频帧），处理帧数据
            if (packet.Length > 4) // 至少需要4字节头部
            {
                // 前4个字节是帧大小（大端序）
                int frameSize = (packet[0] << 24) | (packet[1] << 16) | (packet[2] << 8) | packet[3];

                // 实际帧数据从第5个字节开始
                if (packet.Length >= frameSize + 4) {
                    byte[] frameData = new byte[frameSize];
                    Array.Copy(packet, 4, frameData, 0, frameSize);

                    // 通知帧数据接收
                    OnFrameReceived?.Invoke(this, new FrameData { Data = frameData });
                }
            }
        }
        catch (Exception ex) {
            Log($"处理数据包时出错: {ex.Message}");
            OnError?.Invoke(this, ex);
        }
    }

    /// <summary>
    /// 处理视频信息
    /// </summary>
    private void ProcessVideoInfo(string infoText) {
        try {
            // 格式: INFO:width,height,fps,frameCount,streamFps,isLooping,streamWidth,streamHeight
            string[] parts = infoText.Substring(5).Split(',');

            if (parts.Length >= 6) {
                VideoInfo info = new VideoInfo {
                    OriginalWidth  = int.Parse(parts[0]),
                    OriginalHeight = int.Parse(parts[1]),
                    OriginalFps    = double.Parse(parts[2]),
                    FrameCount     = int.Parse(parts[3]),
                    StreamFps      = double.Parse(parts[4]),
                    IsLooping      = parts[5] == "1"
                };

                // 如果有流的宽高信息
                if (parts.Length >= 8) {
                    info.StreamWidth  = int.Parse(parts[6]);
                    info.StreamHeight = int.Parse(parts[7]);
                }
                else {
                    // 没有传输宽高信息，使用原始宽高
                    info.StreamWidth  = info.OriginalWidth;
                    info.StreamHeight = info.OriginalHeight;
                }

                Log(
                    $"收到视频信息: {info.OriginalWidth}x{info.OriginalHeight}, FPS: {info.OriginalFps}, 帧数: {info.FrameCount}");
                OnVideoInfoReceived?.Invoke(this, info);
            }
        }
        catch (Exception ex) {
            Log($"处理视频信息时出错: {ex.Message}");
            OnError?.Invoke(this, ex);
        }
    }

    /// <summary>
    /// 处理断开连接
    /// </summary>
    private void HandleDisconnect() {
        if (_isConnected) {
            _isConnected = false;
            Log("与服务器的连接已断开");
            OnDisconnected?.Invoke(this, EventArgs.Empty);
        }
    }

    /// <summary>
    /// 记录日志
    /// </summary>
    private void Log(string message) {
        OnLog?.Invoke(this, message);
    }

    /// <summary>
    /// 释放资源
    /// </summary>
    public void Dispose() {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// 释放资源的实现
    /// </summary>
    protected virtual void Dispose(bool disposing) {
        if (!_isDisposed) {
            if (disposing) {
                Disconnect();
                _cancellationTokenSource?.Dispose();
                _stream?.Dispose();
                _client?.Dispose();
            }

            _isDisposed = true;
        }
    }

    /// <summary>
    /// 析构函数
    /// </summary>
    ~VideoClient() {
        Dispose(false);
    }

    /// <summary>
    /// 获取连接状态
    /// </summary>
    public bool IsConnected => _isConnected;
}
