from typing import *
from djitellopy import Tello
import time
import threading
import cv2
import os
import numpy as np
from datetime import datetime

# 飞行速度和控制参数
SPEED = 20
SEARCH_SPEED = 15
FACE_DISTANCE = 120  # 距离人脸多远停止 (TOF传感器读数，单位cm)
RECOGNITION_TIME = 3  # 识别时间（秒）

# 创建videos目录
if not os.path.exists("videos"):
    os.makedirs("videos")

# 全局变量
frame_read = None
show = True
recording = False
video_writer = None
recording_start_time = None
face_detected = False
current_member = 0  # 当前识别的组员索引
members_recognized = [False, False]  # 两位组员的识别状态

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def show_cmd(tl: Tello, c: str):
    """在点阵屏上显示命令"""
    tl.send_expansion_command(f"mled s r {c}")
    tl.send_expansion_command(f"mled sl 255")


def mled_off(tl: Tello):
    """关闭点阵屏"""
    tl.send_expansion_command(f"mled sl 0")


def get_depth(tl: Tello):
    """获取TOF传感器距离数据"""
    try:
        tl.get_distance_tof()
        return int(tl.send_read_command("EXT tof?")[4:])
    except:
        return 500  # 如果读取失败，返回一个较大的默认值


def led(tl: Tello, c: str):
    """控制LED灯颜色"""
    if c == "r":
        c = "255 0 0"
    elif c == "g":
        c = "0 255 0"
    elif c == "b":
        c = "0 0 255"
    elif c == "y":  # 黄色
        c = "255 255 0"
    tl.send_expansion_command(f"led {c}")


def start_recording():
    """开始录制视频"""
    global recording, video_writer, recording_start_time

    if recording:
        return  # 已经在录制

    # 创建文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"videos/tello_mission_{timestamp}.mp4"

    # 获取帧尺寸
    height, width, _ = frame_read.frame.shape

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

    recording = True
    recording_start_time = datetime.now()
    print(f"开始录制: {filename}")


def stop_recording():
    """停止录制视频"""
    global recording, video_writer

    if not recording:
        return  # 没有在录制

    recording = False
    if video_writer is not None:
        video_writer.release()
        video_writer = None
        print("录制停止")


def detect_faces(image):
    """检测图像中的人脸并返回位置信息"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces


def draw_status(image, faces):
    """在图像上显示状态信息和人脸框"""
    # 绘制状态信息
    status_text = f"Rec progress: {sum(members_recognized)}/2"
    cv2.putText(
        image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # 如果正在录制，显示录制指示器
    if recording:
        elapsed = int((datetime.now() - recording_start_time).total_seconds())
        mins, secs = divmod(elapsed, 60)
        timer_text = f"REC {mins:02d}:{secs:02d}"
        cv2.circle(image, (image.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(
            image,
            timer_text,
            (image.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # 绘制人脸框
    for x, y, w, h in faces:
        # 绘制绿色框表示检测到的人脸
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 如果是当前正在识别的人且已识别成功
        if (
            face_detected
            and current_member < len(members_recognized)
            and members_recognized[current_member]
        ):
            label = f"mem {current_member+1} get!"
            cv2.putText(
                image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    return image


def show_img():
    """显示视频流并录制"""
    global recording, video_writer

    while show:
        if frame_read is not None:
            display_frame = frame_read.frame

            # 检测人脸
            faces = detect_faces(display_frame)

            # 绘制状态和人脸框
            display_frame = draw_status(display_frame, faces)

            # 显示帧
            cv2.imshow("Tello 任务", cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            # 录制处理过的帧
            if recording and video_writer is not None:
                video_writer.write(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            # 按ESC键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break

    # 确保释放视频写入器
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def search_and_recognize(tl: Tello):
    """搜索并识别组员的主要逻辑"""
    global face_detected, current_member, members_recognized

    print("开始任务：寻找两位组员并进行人脸识别")

    # 起飞并设置初始高度
    tl.takeoff()
    # tl.move_up(20)  # 从桌子上起飞不用上升，不然太高
    tl.move_back(20)

    while current_member < len(members_recognized):
        if not members_recognized[current_member]:
            # 旋转搜索人脸
            search_for_face(tl)

            if face_detected:
                # 识别成功，设置LED为绿色
                led(tl, "g")
                show_cmd(tl, str(current_member + 1))
                print(f"组员 {current_member + 1} 识别成功!")

                # 标记当前组员已识别
                members_recognized[current_member] = True

                # 暂停几秒确保提示明显
                time.sleep(RECOGNITION_TIME)

                # 继续到下一个组员
                current_member += 1
                face_detected = False

                # 转身180度寻找下一个组员
                if current_member < len(members_recognized):
                    tl.rotate_clockwise(180)

            # 重置LED
            led(tl, "b")
            mled_off(tl)

    print("任务完成: 所有组员识别成功!")

    # 返回起始位置并降落
    tl.move_back(100)
    tl.land()


def search_for_face(tl: Tello):
    """搜索人脸的逻辑"""
    global face_detected, frame_read

    # 首先尝试按螺旋形搜索
    for _ in range(8):  # 控制搜索的圈数
        if face_detected:
            return

        # 旋转搜索
        for _ in range(8):  # 每圈旋转8次，每次45度
            if frame_read is not None:
                faces = detect_faces(frame_read.frame)

                if len(faces) > 0:
                    # 找到人脸，调整位置
                    face_detected = adjust_position(tl, faces[0])
                    if face_detected:
                        return

            # 没找到，继续旋转
            tl.rotate_counter_clockwise(45)
            time.sleep(1)

        # 完成一圈旋转后，向前移动一些
        tl.move_forward(50)
        time.sleep(1)


def adjust_position(tl: Tello, face):
    """调整无人机位置到人脸前方合适距离"""
    global frame_read

    x, y, w, h = face

    # 水平调整（左右）
    max_attempts = 5
    attempts = 0

    while attempts < max_attempts:
        if frame_read is not None:
            faces = detect_faces(frame_read.frame)
            if len(faces) > 0:
                return True

        # 更新人脸位置
        if frame_read is not None:
            faces = detect_faces(frame_read.frame)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                center_x = x + w // 2
            else:
                # 找不到人脸了
                attempts += 1
        else:
            attempts += 1

    return False


if __name__ == "__main__":
    tl = Tello()

    try:
        # 连接到Tello
        tl.connect()
        print("电池电量:", tl.get_battery(), "%")

        # 启动视频流
        tl.streamoff()
        tl.streamon()

        # 获取视频帧
        frame_read = tl.get_frame_read()

        # 等待视频流初始化
        time.sleep(2)

        # 启动视频显示线程
        video_thread = threading.Thread(target=show_img)
        video_thread.start()

        # 开始录制
        start_recording()

        # 执行主要任务
        search_and_recognize(tl)

    except Exception as e:
        print(f"发生错误: {e}")

    except KeyboardInterrupt:
        print("停止")

    finally:
        print("任务结束")

        # 确保停止录制
        stop_recording()

        # 确保无人机安全降落
        try:
            tl.land()
        except:
            pass

        tl.streamoff()

        # 设置标志通知显示线程退出
        show = False
        if "video_thread" in locals():
            video_thread.join()
