import logging
from threading import Thread
import cv2
from djitellopy import Tello
import time
import os
import numpy as np
import torch
from DataCollector import DataCollector
from PIDController import PIDController
from ultralytics import YOLO  # 导入YOLO
import math  # 导入 math 以便后续可能需要

# --- 配置 ---
MODEL_PATH = "best.pt"  # YOLOv8 模型文件路径
TARGET_CLASSES = ["bit", "drone", "card"]  # 需要检测并获取坐标的目标类别
CONFIDENCE_THRESHOLD = 0.8  # 置信度阈值 (与原代码一致)
device = torch.device("cuda" if torch.cuda.is_available() else exit())
# --- Tello 设置 ---
CROP_WIDTH = 320
CROP_HEIGHT = 240
TARGET_CENTER_Y = CROP_WIDTH // 2
TARGET_CENTER_X = CROP_HEIGHT // 2
# --- 加载模型 ---
try:
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"成功加载模型: {MODEL_PATH}")
    # 检查模型类别名称是否包含目标类别
    model_classes = model.names
    print(f"模型包含的类别: {list(model_classes.values())}")
    for target in TARGET_CLASSES:
        if target not in model_classes.values():
            print(f"警告: 目标类别 '{target}' 可能不在模型类别中!")
except Exception as e:
    print(f"加载 YOLO 模型时出错: {e}")
    exit()

TARGET_HEIGHT = 100
RUNNING = True
CONTROL_MODE_TARGET = False
TARGET_TYPE = "card"


def show_cmd(tl: Tello, c: str):
    tl.send_expansion_command(f"mled s r {c}")
    tl.send_expansion_command(f"mled sl 255")


OUTPUT = {
    # 前后速度，前+
    "vx": 0,
    # 左右速度，左+
    "vy": 0,
    # 上下速度，上+
    "vz": 0,
    # yaw速度，逆时针+
    "vr": 0,
}

# cv: pid
# #→ x 120
# ↓
# y 160

# drone:
# x
# ↑
# #→ y


def add_annotation_area(frame, center_x, center_y, distance):
    annotation_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA
    text_padding_x = 10
    text_padding_y_line1 = 25
    text_padding_y_line2 = 45
    background_color = (255, 255, 255)

    h, w = frame.shape[:2]

    annotation_area = np.full(
        (annotation_height, w, 3), background_color, dtype=np.uint8
    )

    center_text = f"Center: ({int(center_x)}, {int(center_y)})"
    distance_text = f"Distance: {distance:.2f} pixels"

    cv2.putText(
        annotation_area,
        center_text,
        (
            text_padding_x,
            text_padding_y_line1,
        ),
        font,
        font_scale,
        font_color,
        thickness,
        line_type,
    )

    cv2.putText(
        annotation_area,
        distance_text,
        (text_padding_x, text_padding_y_line2),
        font,
        font_scale,
        font_color,
        thickness,
        line_type,
    )

    combined_frame = cv2.vconcat([frame, annotation_area])

    return combined_frame


def detect_task(frame_read):
    global RUNNING, CROP_HEIGHT, CONFIDENCE_THRESHOLD

    no_target_cnt = 0

    print("检测开始")
    dc = DataCollector("mid_data.csv")

    pid_vx = PIDController(0.12, 0.07, 0, 0, (-100, 100), (-7, 7))
    pid_vy = PIDController(0.12, 0.07, 0, 0, (-100, 100), (-7, 7))

    while RUNNING:
        # --- 获取帧 ---
        frame = frame_read.frame
        if frame is None:
            print("未能获取到帧，可能连接中断。")
            time.sleep(0.5)  # 短暂等待后重试
            continue

        # --- 裁剪帧 ---
        # 确保裁剪不会超出图像边界 (虽然 [:240] 通常是安全的)
        h, w, _ = frame.shape
        actual_crop_height = min(CROP_HEIGHT, h)  # 防止裁剪高度超过实际高度
        cropped_frame = frame[
            :actual_crop_height, :, :
        ].copy()  # 裁剪帧并创建副本以进行修改
        cropped_frame = cv2.flip(cv2.transpose(cropped_frame), 1)

        # --- 使用YOLOv8进行检测 (在裁剪后的帧上) ---
        results = model(
            cropped_frame, stream=False, verbose=False, conf=CONFIDENCE_THRESHOLD
        )

        # --- 处理检测结果 ---
        center_x = None
        center_y = None
        distance = 0

        if results and results[0]:
            boxes = results[0].boxes  # 获取Boxes对象

            # 遍历每个检测到的边界框
            for box in boxes:
                # 提取置信度
                confidence = box.conf[0]
                # 提取类别ID
                cls_id = int(box.cls[0])
                # 获取类别名称
                class_name = model.names[cls_id]

                if class_name == TARGET_TYPE:
                    # 提取边界框坐标 (xyxy 格式)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # --- 计算中心点坐标 ---
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # --- 在裁剪后的帧上绘制 ---
                    # 绘制边界框 (绿色)
                    cv2.rectangle(cropped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 准备标签文字 (英文)
                    label = f"{class_name}: {confidence:.2f}"
                    # 绘制标签背景
                    (w_label, h_label), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        cropped_frame,
                        (x1, y1 - h_label - 5),
                        (x1 + w_label, y1),
                        (0, 255, 0),
                        -1,
                    )  # 填充背景
                    # 绘制标签文字 (白色)
                    cv2.putText(
                        cropped_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                    # 绘制中心点 (红色)
                    cv2.circle(cropped_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    # --- 输出中心坐标 (中文) ---
                    # print(
                    #     f"检测到目标: {class_name} | 中心坐标: ({center_x}, {center_y})"
                    # )
                    # print(f"vx: {vx}, vy: {vy}")

                    break

        if center_x != None and center_y != None:
            dc.collect_datas([("x", center_x), ("y", center_y)])

            # --- 计算pid --
            try:
                #     output["vz"] = pid_height.update(tl.query_distance_tof(), 1 / 30)
                dx = TARGET_CENTER_X - center_x
                dy = TARGET_CENTER_Y - center_y

                OUTPUT["vy"] = pid_vx.update(TARGET_CENTER_X - center_x)
                OUTPUT["vx"] = -pid_vy.update(TARGET_CENTER_Y - center_y)
                distance = np.sqrt(dx**2 + dy**2)

            except Exception as e:
                print("Warning: ", e)
                continue
        else:
            distance = -1
            center_x = -1
            center_y = -1

            if no_target_cnt > 6:
                OUTPUT["vz"] = 0
                OUTPUT["vx"] = 0
                OUTPUT["vy"] = 0
                no_target_cnt = 0
            else:
                no_target_cnt += 1

        # --- 显示处理后的帧 ---
        cv2.circle(
            cropped_frame,
            (TARGET_CENTER_X, TARGET_CENTER_Y),
            3,
            (255, 0, 0),
            -1,
        )

        cropped_frame = add_annotation_area(cropped_frame, center_x, center_y, distance)
        cv2.imshow("Tello Cropped Detection", cropped_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            RUNNING = False

    # --- 退出条件 ---
    dc.save_to_csv()
    print("退出检测")


def keep_height(tl: Tello):
    global RUNNING, TARGET_HEIGHT, OUTPUT

    pid_height = PIDController(1.7, 0.1, 0, TARGET_HEIGHT, (-100, 100))
    while RUNNING:
        try:
            current_heigh = tl.query_distance_tof()
            o = pid_height.update(current_heigh)
            OUTPUT["vz"] = o
        except Exception:
            continue
        time.sleep(0.001)

    OUTPUT["vz"] = 0


def control_tl(tl: Tello):
    global RUNNING, TARGET_HEIGHT, OUTPUT

    while RUNNING:
        if CONTROL_MODE_TARGET:
            tl.send_rc_control(
                int(OUTPUT["vy"]), int(OUTPUT["vx"]), int(OUTPUT["vz"]), 0
            )
            print(int(OUTPUT["vy"]), int(OUTPUT["vx"]), int(OUTPUT["vz"]))
        else:
            ...
        time.sleep(0.001)

    OUTPUT["vz"] = 0


def main(tl: Tello):
    global RUNNING, CONTROL_MODE_TARGET, TARGET_TYPE

    Tello.LOGGER.setLevel(logging.WARNING)

    tl.connect()
    print(f"电池：{tl.query_battery()}%")
    print(f"状态：{tl.get_current_state()}")
    tl.streamon()
    tl.set_video_fps(Tello.FPS_30)
    tl.set_video_bitrate(Tello.BITRATE_2MBPS)
    tl.set_video_direction(Tello.CAMERA_DOWNWARD)
    tl.set_video_resolution(Tello.RESOLUTION_480P)

    frame_read = tl.get_frame_read()

    # tl.takeoff = lambda *arg: ...
    # tl.move_up = lambda *arg: ...
    # tl.send_rc_control = lambda *arg: ...
    # tl.land = lambda *arg: ...

    tl.takeoff()

    Thread(target=detect_task, args=(frame_read,)).start()

    begin = time.time()
    tl.move_up(50)
    print(f"升空耗时：{time.time() - begin}")

    begin = time.time()
    tl.send_rc_control(0, 0, 0, 0)

    CONTROL_MODE_TARGET = True
    TARGET_TYPE = "card"

    Thread(target=keep_height, args=(tl,)).start()
    Thread(target=control_tl, args=(tl,)).start()

    while time.time() - begin < 120 and RUNNING:
        time.sleep(0.001)

    tl.send_rc_control(0, 0, 0, 0)
    RUNNING = False
    time.sleep(5)


if __name__ == "__main__":
    tl = Tello()

    try:
        main(tl)

    except KeyboardInterrupt:
        print("^C")
        RUNNING = False
        time.sleep(1)

    finally:
        tl.send_rc_control(0, 0, 0, 0)
        tl.land()
        tl.streamoff()
