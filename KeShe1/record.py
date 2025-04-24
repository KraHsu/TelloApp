import threading
import cv2
from djitellopy import Tello
import time
import os
import numpy as np

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    print("Tello battery: ", tello.get_battery())
    tello.streamon()
    tello.set_video_direction(tello.CAMERA_DOWNWARD)
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_bitrate(tello.BITRATE_5MBPS)
    tello.set_video_resolution(tello.RESOLUTION_720P)

    time.sleep(2)

    frame_read = tello.get_frame_read()

    # 获取视频帧的尺寸
    frame_height, frame_width = 240, 320

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用MP4格式
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    save_dir = "recordings"  # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"tello_recording_{timestamp}.mp4")

    out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

    try:
        while cv2.waitKey(1) & 0xFF != 27:
            # 获取原始帧
            frame = frame_read.frame
            # 获取裁剪后的帧
            cropped_frame = frame[:240, :, :]

            # 显示两个窗口
            cv2.imshow("Name-", cropped_frame)

            # 写入裁剪后的帧到视频文件
            out.write(cropped_frame)

    except KeyboardInterrupt:
        print("录制被用户中断")
    finally:
        # 释放资源
        out.release()
        tello.streamoff()
        tello.end()
        time.sleep(1)
        cv2.destroyAllWindows()
        print(f"视频已保存到: {video_path}")
