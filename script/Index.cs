using Godot;
using System;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Tello.script;

public partial class Index : Node {
    private TcpClient tcpClient;

    [Export] private Button testButton = null!;
    [Export] private Label  testLabel  = null!;

    [Export] private Button stopButton = null!;

    public override async void _Ready() {
        try {
            await InitTcpClient();

            testButton.ButtonDown += async () => {
                var echo = await tcpClient.SendEchoAsync("你好，服务器！");
                testLabel.Text = echo;
            };

            stopButton.ButtonDown += () => { tcpClient.Disconnect(); };
        }
        catch (Exception e) {
            throw; // TODO handle exception
        }
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
