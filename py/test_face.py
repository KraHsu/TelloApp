from djitellopy import Tello
import cv2
import time
import threading
import numpy as np
import os
import argparse
from datetime import datetime
import queue
import re  # 添加正则表达式支持

# 命令行参数解析
parser = argparse.ArgumentParser(description="Tello人脸检测测试")
parser.add_argument(
    "-m",
    "--method",
    type=str,
    default="opencv",
    choices=["opencv", "mtcnn", "face_recognition", "opencv_dnn"],  # 添加opencv_dnn选项
    help="选择人脸检测方法: opencv, opencv_dnn, mtcnn, 或 face_recognition",
)
parser.add_argument(
    "-r",
    "--recognize",
    action="store_true",
    help="启用特定人员识别模式(仅适用于face_recognition方法)",
)
parser.add_argument(
    "-i",
    "--images",
    type=str,
    default="known_faces",
    help="包含已知人脸图像的文件夹路径",
)
args = parser.parse_args()

# 全局变量
frame = None
running = True
detection_methods = [
    "opencv",
    "opencv_dnn",
    "mtcnn",
    "face_recognition",
]  # 更新方法列表
current_method = args.method
enable_recognition = args.recognize
known_faces_dir = args.images

known_face_encodings_by_person = {}  # 每个人的多个人脸编码
known_face_names = []  # 已知人名列表

latest_frame = None  # 最新的视频帧
latest_results = None  # 最新的识别结果 (faces, names)
processing = False  # 是否正在处理识别
results_frame_count = 0  # 当前结果已使用的帧数
max_reuse_frames = 60  # 最多重复使用结果的帧数
results_lock = threading.Lock()  # 用于同步访问结果的锁
last_process_time = 0  # 上次处理时间，用于超时检测


# OpenCV Haar级联检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

opencv_dnn_face_detector = None


def init_opencv_dnn_detector():
    """初始化OpenCV DNN人脸检测器 (使用YuNet ONNX模型)"""
    global opencv_dnn_face_detector

    try:
        # 使用YuNet ONNX模型
        model_file = "modules/face_detection_yunet_2023mar.onnx"

        # 检查模型文件是否存在
        if not os.path.exists(model_file):
            print(f"警告: YuNet ONNX模型文件 '{model_file}' 不存在")
            print(f"请下载模型文件到 {os.path.abspath('.')} 目录")
            print(
                "可从 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet 下载"
            )
            return None

        # 使用FaceDetectorYN创建检测器
        detector = cv2.FaceDetectorYN.create(
            model_file,
            "",
            (320, 320),  # 输入大小
            0.9,  # 得分阈值
            0.3,  # NMS阈值
            5000,  # 最大检测数量
        )

        print("YuNet ONNX人脸检测器加载成功")
        return detector
    except Exception as e:
        print(f"加载YuNet ONNX模型失败: {e}")
        return None


def detect_faces_opencv_dnn(image):
    """使用OpenCV YuNet ONNX模型检测人脸"""
    global opencv_dnn_face_detector

    # 如果检测器未初始化，尝试初始化
    if opencv_dnn_face_detector is None:
        opencv_dnn_face_detector = init_opencv_dnn_detector()

    # 如果仍然无法初始化，回退到普通OpenCV方法
    if opencv_dnn_face_detector is None:
        print("警告: YuNet检测器不可用，回退到Haar级联分类器")
        return detect_faces_opencv(image)

    # 设置输入大小
    height, width = image.shape[:2]
    opencv_dnn_face_detector.setInputSize((width, height))

    # 执行检测 - YuNet返回tuple(retval, faces)
    detection_result = opencv_dnn_face_detector.detect(image)
    print(detection_result)

    # 检查人脸信息是否存在
    if detection_result[1] is None:
        return []

    # 处理检测结果，转换为(x, y, w, h)格式
    face_boxes = []
    for face in detection_result[1]:
        # YuNet返回的格式是[x, y, w, h, score, landmarks...]
        x, y, w, h = map(int, face[:4])
        confidence = face[-1]  # 最后一个元素是置信度

        # 只有当置信度大于阈值时才添加
        if confidence > 0.3:  # 可调整置信度阈值
            face_boxes.append((x, y, w, h))

            # 调试信息 - 打印检测到的人脸信息
            print(f"人脸坐标: ({x}, {y}), 宽高: ({w}, {h}), 置信度: {confidence:.2f}")

            # 如果需要处理关键点，可以从face[4:-1]中提取
            # 典型的有5个关键点，每个关键点有x,y两个坐标: eyes(2), nose(1), mouth corners(2)

    return face_boxes


# 检查MTCNN是否可用
try:
    from mtcnn import MTCNN

    mtcnn_detector = MTCNN()
    has_mtcnn = True
    print("MTCNN可用")
except ImportError:
    has_mtcnn = False
    print("MTCNN不可用，请使用 'pip install mtcnn tensorflow' 安装")
    if current_method == "mtcnn":
        print("警告: 已选择MTCNN但未安装，将使用OpenCV替代")
        current_method = "opencv"

# 检查face_recognition是否可用
try:
    import face_recognition

    has_face_recognition = True
    print("face_recognition可用")
except ImportError:
    has_face_recognition = False
    print("face_recognition不可用，请使用 'pip install face_recognition' 安装")
    print(
        "注意: 这可能需要先安装dlib，详见https://github.com/ageitgey/face_recognition"
    )
    if current_method == "face_recognition":
        print("警告: 已选择face_recognition但未安装，将使用OpenCV替代")
        current_method = "opencv"


def face_detection_thread():
    """专门负责人脸检测的线程"""
    global latest_frame, latest_results, processing, results_frame_count, last_process_time

    while running:
        # 检查是否有帧可处理
        current_frame = None

        with results_lock:
            # 添加处理超时检测，如果处理时间超过3秒，重置处理状态
            if processing and time.time() - last_process_time > 3:
                print("检测到处理超时，重置处理状态")
                processing = False

            # 只有当不在处理中且有可用帧时才处理
            if not processing and latest_frame is not None:
                current_frame = latest_frame.copy()
                processing = True
                last_process_time = time.time()

        # 如果没有需要处理的帧，等待一会儿
        if current_frame is None:
            time.sleep(0.01)
            continue

        try:
            # 执行人脸检测
            faces, names = detect_faces(current_frame, method=current_method)

            # 更新结果
            with results_lock:
                latest_results = (faces, names, current_frame.shape)
                results_frame_count = 0
                processing = False
        except Exception as e:
            print(f"人脸检测错误: {e}")
            with results_lock:
                processing = False

        # 控制处理频率，避免CPU过载
        time.sleep(0.01)


def detect_faces_opencv(image, improved=True):
    """使用OpenCV Haar级联分类器检测人脸"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if improved:
        # 改进的参数
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=10, minSize=(120, 120)
        )
    else:
        # 默认参数
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    return faces


def detect_faces_mtcnn(image):
    """使用MTCNN检测人脸"""
    if not has_mtcnn:
        return []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtcnn_results = mtcnn_detector.detect_faces(rgb_image)

    faces = []
    for face in mtcnn_results:
        # 提取边界框坐标
        x, y, w, h = face["box"]
        # 过滤低置信度检测
        if face["confidence"] > 0.9:
            faces.append((x, y, w, h))

    return faces


first = True


def detect_faces_fr(image, recognize_specific=False):
    """使用face_recognition库检测人脸，支持每人多张图片的识别"""
    global first, known_face_encodings_by_person, known_face_names

    if not has_face_recognition:
        return [], [None]

    # 转换为RGB格式（face_recognition需要）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 定位图像中的所有人脸
    face_locations = face_recognition.face_locations(rgb_image, model="cnn")

    if len(face_locations) != 0 and first:
        cv2.imwrite("test.png", rgb_image)
        first = False
        print("获取一张人脸")

    # 如果不需要识别特定人员或没有已知人脸，返回所有人脸
    if not recognize_specific or len(known_face_names) == 0:
        # 转换为OpenCV格式 (x, y, w, h)
        faces = []
        for top, right, bottom, left in face_locations:
            faces.append((left, top, right - left, bottom - top))
        return faces, [None] * len(faces)

    # 提取当前帧中所有人脸的编码
    face_encodings = face_recognition.face_encodings(
        rgb_image, face_locations, model="large"
    )

    # 存储匹配的人脸和姓名
    matched_faces = []
    matched_names = []

    # 比较每个人脸与已知人脸
    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        matched_person = None
        min_distance = 1.0  # 距离范围是0-1，越小越相似

        # 对每个已知人员进行检查
        for person_name in known_face_names:
            # 获取该人员的所有编码
            person_encodings = known_face_encodings_by_person[person_name]

            # 与该人员的所有编码进行比较
            matches = face_recognition.compare_faces(
                person_encodings, face_encoding, tolerance=0.6
            )

            # 如果有任何一个编码匹配
            if True in matches:
                # 计算与所有编码的距离并取最小值
                face_distances = face_recognition.face_distance(
                    person_encodings, face_encoding
                )
                best_match_distance = min(face_distances)

                # 如果这是目前最佳匹配
                if best_match_distance < min_distance:
                    min_distance = best_match_distance
                    matched_person = person_name

        # 如果找到匹配的人
        if matched_person is not None:
            matched_faces.append((left, top, right - left, bottom - top))
            matched_names.append(matched_person)

            # 打印调试信息
            # print(f"识别到: {matched_person}, 距离: {min_distance:.3f}")

    return matched_faces, matched_names


def load_known_faces(directory="known_faces"):
    """加载已知人脸图像并提取特征，支持每人多张图片"""
    global known_face_encodings_by_person, known_face_names

    if not has_face_recognition:
        print("无法加载已知人脸: face_recognition库不可用")
        return

    if not os.path.exists(directory):
        print(f"已知人脸目录 '{directory}' 不存在，创建目录")
        os.makedirs(directory)
        print(f"请在 '{directory}' 目录中放置已知人员的图片")
        print("图片文件名格式: 'name.jpg' 或 'name1.jpg', 'name2.jpg' 等")
        return

    face_images = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not face_images:
        print(f"目录 '{directory}' 中没有找到图片")
        return

    print(f"加载 {len(face_images)} 张已知人脸图片...")

    # 临时存储，用于收集每个人的所有编码
    person_encodings = {}

    for image_file in face_images:
        try:
            # 从文件名提取人名基础部分 (去掉数字后缀和扩展名)
            # 例如: "john1.jpg" -> "john", "mary_smith2.png" -> "mary_smith"
            file_base = os.path.splitext(image_file)[0]
            # 使用正则表达式移除末尾的数字
            person_name = (
                re.sub(r"\d+$", "", file_base).rstrip("_").rstrip("-").rstrip(".")
            )

            # 加载图像
            image_path = os.path.join(directory, image_file)
            image = face_recognition.load_image_file(image_path)

            # 尝试找到并编码人脸
            face_encodings = face_recognition.face_encodings(image, model="large")

            if len(face_encodings) > 0:
                # 使用第一个检测到的人脸
                # 如果这个人已经有编码，则添加到列表中，否则创建新列表
                if person_name not in person_encodings:
                    person_encodings[person_name] = []

                person_encodings[person_name].append(face_encodings[0])
                print(f"成功加载: {image_file} (识别为: {person_name})")
            else:
                print(f"警告: 在图片 {image_file} 中未检测到人脸")

        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")

    # 将收集的编码转换为最终格式
    known_face_encodings_by_person = person_encodings
    known_face_names = list(person_encodings.keys())

    # 打印统计信息
    total_encodings = sum(len(encodings) for encodings in person_encodings.values())
    print(f"已加载 {len(known_face_names)} 个人的 {total_encodings} 个人脸编码:")
    for name, encodings in person_encodings.items():
        print(f"  - {name}: {len(encodings)} 张照片")


def detect_faces(image, method="opencv"):
    """使用指定方法检测人脸"""
    global enable_recognition

    if method == "opencv":
        return detect_faces_opencv(image, improved=True), [None]
    elif method == "opencv_dnn":  # 添加opencv_dnn方法
        return detect_faces_opencv_dnn(image), [None]
    elif method == "mtcnn" and has_mtcnn:
        return detect_faces_mtcnn(image), [None]
    elif method == "face_recognition" and has_face_recognition:
        return detect_faces_fr(image, recognize_specific=enable_recognition)
    # 默认回退到OpenCV
    return detect_faces_opencv(image, improved=True), [None]


def process_frame():
    """处理帧并显示结果（不再执行人脸检测）"""
    global frame, current_method, enable_recognition, latest_results, results_frame_count

    while running:
        if frame is None:
            time.sleep(0.1)
            continue

        # 制作显示帧的副本
        display_frame = frame.copy()

        # 获取当前的识别结果
        faces = []
        names = []
        reused_result = False

        with results_lock:
            if latest_results is not None:
                # 检查是否需要重用结果
                if processing and results_frame_count < max_reuse_frames:
                    faces, names, frame_shape = latest_results

                    # 检查帧尺寸是否匹配，如果不匹配则不使用缓存结果
                    if (
                        frame_shape[0] == display_frame.shape[0]
                        and frame_shape[1] == display_frame.shape[1]
                    ):
                        results_frame_count += 1
                        reused_result = True
                    else:
                        faces = []
                        names = []
                elif not processing:
                    faces, names, _ = latest_results
                    results_frame_count = 0

        # 在画面上显示人脸框
        for i, (x, y, w, h) in enumerate(faces):
            # 绘制人脸框
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 如果有名字，显示名字
            if i < len(names) and names[i]:
                cv2.putText(
                    display_frame,
                    names[i],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # 添加状态信息
        cv2.putText(
            display_frame,
            f"Method: {current_method}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 显示识别状态
        if current_method == "face_recognition" and enable_recognition:
            cv2.putText(
                display_frame,
                "Mode: Specific Person Recognition",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Recognized {len(faces)} known faces",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                "Mode: All Faces Detection",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Detect {len(faces)} faces",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # 显示缓存状态
        if reused_result:
            cv2.putText(
                display_frame,
                f"Using cached result (frame {results_frame_count}/{max_reuse_frames})",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # 显示处理状态
        processing_status = "Processing..." if processing else "Ready"
        cv2.putText(
            display_frame,
            f"Status: {processing_status}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # 显示操作指引
        cv2.putText(
            display_frame,
            "use 'space' to change method",
            (10, display_frame.shape[0] - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "use 'r' to toggle recognition mode",
            (10, display_frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "use 'esc' to exit",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # 显示帧
        cv2.imshow("人脸检测测试", cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
        elif key == 32:  # 空格键
            # 切换到下一个可用的检测方法
            available_methods = ["opencv"]
            if has_mtcnn:
                available_methods.append("mtcnn")
            if has_face_recognition:
                available_methods.append("face_recognition")

            current_idx = (
                available_methods.index(current_method)
                if current_method in available_methods
                else 0
            )
            current_method = available_methods[
                (current_idx + 1) % len(available_methods)
            ]
            print(f"切换到检测方法: {current_method}")
        elif key == ord("r"):  # 'r'键
            # 切换识别模式
            if current_method == "face_recognition" and has_face_recognition:
                enable_recognition = not enable_recognition
                mode = "启用" if enable_recognition else "禁用"
                print(f"{mode}特定人员识别模式")
            else:
                print("特定人员识别仅在face_recognition模式下可用")

    cv2.destroyAllWindows()


def video_thread_function(tello):
    """处理视频流"""
    global frame, running, latest_frame

    frame_read = tello.get_frame_read()

    try:
        while running:
            current_frame = frame_read.frame
            if current_frame is not None:
                # 创建新的副本以避免引用问题
                frame = current_frame.copy()  # 用于兼容旧代码
                latest_frame = current_frame.copy()  # 用于人脸检测线程

            time.sleep(0.03)  # 限制帧率以减少CPU使用
    except Exception as e:
        print(f"视频流错误: {e}")
    finally:
        print("视频线程结束")


def main():
    global running, enable_recognition, latest_frame, processing

    # 初始化状态
    latest_frame = None
    processing = False

    print(f"测试人脸识别 - 使用{current_method}方法 - 不起飞")

    # 如果使用OpenCV DNN方法，预先初始化检测器
    if current_method == "opencv_dnn":
        init_opencv_dnn_detector()

    # 如果启用特定人员识别，加载已知人脸
    if enable_recognition and current_method == "face_recognition":
        load_known_faces(known_faces_dir)

        # 如果没有找到已知人脸，禁用识别模式
        if len(known_face_names) == 0:
            print("警告: 未找到有效的已知人脸，禁用识别模式")
            enable_recognition = False
    elif enable_recognition and current_method != "face_recognition":
        print("特定人员识别仅在face_recognition模式下可用，已忽略")
        enable_recognition = False

    try:
        # 连接到Tello
        tello = Tello()
        tello.connect()
        print(f"电池电量: {tello.get_battery()}%")

        # 启动视频流
        tello.streamoff()
        tello.streamon()
        print("视频流已启动")

        # 等待视频流初始化
        time.sleep(2)

        # 启动视频线程
        video_thread = threading.Thread(target=video_thread_function, args=(tello,))
        video_thread.daemon = True
        video_thread.start()

        # 确保视频线程已经获取到帧
        print("等待视频流初始化...")
        timeout = 10  # 10秒超时
        start_time = time.time()
        while latest_frame is None:
            if time.time() - start_time > timeout:
                print("等待视频流超时，可能无法正常工作")
                break
            time.sleep(0.1)

        if latest_frame is not None:
            print("视频流已准备就绪")

        # 启动人脸检测线程
        detection_thread = threading.Thread(target=face_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        print("人脸检测线程已启动")

        # 启动显示线程（主线程）
        process_frame()

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 清理资源
        running = False
        print("关闭视频流...")
        try:
            tello.streamoff()
        except:
            pass
        print("测试结束")


if __name__ == "__main__":
    main()
