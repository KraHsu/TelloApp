using Godot;
using System;
using System.Text;
using System.Threading.Tasks;

namespace Tello.script;

public partial class Index : Node {
    private TcpClient   tcpClient;
    private VideoClient client;

    [Export] private TextureRect rect = null!;

    [Export] private Button testButton = null!;
    [Export] private Label  testLabel  = null!;

    ImageTexture texture = null;

    public override async void _Ready() {
        try {
            await InitTcpClient();

            await tcpClient.SendRequestAsync(new Request("start", new { }));

            testButton.ButtonDown += async () => {
                // testLabel.Text = echo;
            };
        }
        catch (Exception e) {
            throw; // TODO handle exception
        }

        // // 创建客户端实例
        // client = new VideoClient();
        //
        // // 注册事件处理
        // client.OnConnected    += (s, e) => GD.Print("已连接到服务器");
        // client.OnDisconnected += (s, e) => GD.Print("已断开与服务器的连接");
        // client.OnLog          += (s, m) => GD.Print($"日志: {m}");
        // client.OnError        += (s, ex) => GD.Print($"错误: {ex.Message}");
        // client.OnVideoInfoReceived += (s, info) => {
        //     GD.Print($"视频信息: {info.OriginalWidth}x{info.OriginalHeight}");
        //     // 在这里可以准备UI组件来显示视频
        // };
        // client.OnFrameReceived += (s, frame) => {
        //     try {
        //         // frame.Data 实际上是Base64编码的JPEG数据，需要先解码
        //         var jpegData = frame.Data; // 这已经是服务器发送的Base64编码后的数据
        //
        //         // 创建Image对象
        //         Image image = new();
        //
        //         // 使用正确的数据格式加载图像
        //         var   decodedData = Convert.FromBase64String(Encoding.UTF8.GetString(frame.Data));
        //         Error err         = image.LoadJpgFromBuffer(decodedData);
        //
        //         if (err != Error.Ok) {
        //             GD.PrintErr($"无法加载图像数据(尝试Base64解码后): {err}");
        //             return;
        //         }
        //
        //         CallDeferred(nameof(UpdateFrame), image);
        //
        //         // 成功创建纹理
        //         GD.Print($"成功接收到新帧，尺寸: {image.GetWidth()}x{image.GetHeight()}");
        //     }
        //     catch (Exception ex) {
        //         GD.PrintErr($"处理图像帧时出错: {ex.Message}");
        //     }
        // };
        //
        // // 连接到服务器
        // await client.ConnectAsync();
    }

    private void UpdateFrame(Image image) {
        if (texture == null) {
            texture      = ImageTexture.CreateFromImage(image);
            rect.Texture = texture;
        }
        else {
            texture.Update(image);
        }
    }

    private ulong lastTime = 0;

    public override async void _Process(double delta) {
        var now = Time.GetTicksMsec();
        if (now - lastTime <= 10) return;
        if (tcpClient.IsConnected) {
            await tcpClient.SendRequestAsync(new Request(cmd: "key", new {
                Front = Input.IsActionPressed("Front"),
                Back  = Input.IsActionPressed("Back"),
                Right = Input.IsActionPressed("Right"),
                Left  = Input.IsActionPressed("Left"),
                Up    = Input.IsActionPressed("Up"),
                Down  = Input.IsActionPressed("Down")
            }));
        }

        lastTime = now;
    }

    public override async void _ExitTree() {
        await tcpClient.SendRequestAsync(new Request("stop", new { }));

        tcpClient.Disconnect();
        client.Disconnect();
    }


    private async Task InitTcpClient() {
        try {
            tcpClient = new TcpClient();

            await tcpClient.ConnectAsync();

            // // 注册事件
            // _client.ConnectionStatusChanged += (sender, connected) => {
            //     // 使用Godot的CallDeferred确保UI更新在主线程进行
            //     CallDeferred(nameof(UpdateConnectionStatus), connected);
            // };
            //
            // _client.FrameReceived += (sender, args) => {
            //     // 确保图像更新在Godot的主线程处理
            //     CallDeferred(nameof(UpdateVideoFrame), args.ImageData);
            // };
            //
            // _client.MessageReceived += (sender, response) => {
            //     // 在Godot主线程更新消息状态
            //     CallDeferred(nameof(UpdateMessageStatus), response.code);
            // };
        }
        catch (Exception e) {
            throw; // TODO handle exception
        }
    }
}
