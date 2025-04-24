#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多方法人脸检测与识别程序

支持四种检测方法:
1. OpenCV Haar级联分类器
2. OpenCV DNN (YuNet ONNX模型)
3. MTCNN多任务级联网络
4. face_recognition库

使用方法:
python face_detector.py --image path/to/image.jpg --method face_recognition --recognize

参数说明:
  --image: 要检测的图片路径
  --method: 检测方法 (opencv, opencv_dnn, mtcnn, face_recognition)
  --recognize: 启用人脸识别 (仅适用于face_recognition方法)
  --faces_dir: 已知人脸目录 (默认为"known_faces")
"""

import os
import sys
import cv2
import numpy as np
import argparse
import re
import time
from PIL import Image, ImageDraw, ImageFont


class Logger:
    """简单的日志类"""

    def __init__(self):
        self.DEBUG = 0
        self.INFO = 1
        self.WARNING = 2
        self.ERROR = 3
        self.level = self.INFO

    def debug(self, msg):
        if self.level <= self.DEBUG:
            print(f"[DEBUG] {msg}")

    def info(self, msg):
        if self.level <= self.INFO:
            print(f"[INFO] {msg}")

    def warning(self, msg):
        if self.level <= self.WARNING:
            print(f"[WARNING] {msg}")

    def error(self, msg):
        if self.level <= self.ERROR:
            print(f"[ERROR] {msg}")


# 初始化日志器
logger = Logger()


def put_chinese_text(img, text, position, font_path, font_size, color):
    """在图片上绘制中文文本"""
    if not text:
        return img

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        try:
            # 尝试系统字体
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size
            )
        except IOError:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
            except IOError:
                # 使用默认字体
                font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_result


class FaceDetector:
    """基础人脸检测器类"""

    def __init__(self):
        """初始化检测器"""
        pass

    def detect_faces(self, image):
        """检测人脸的抽象方法"""
        raise NotImplementedError("子类必须实现此方法")


class OpenCVFaceDetector(FaceDetector):
    """OpenCV Haar级联分类器人脸检测器"""

    def __init__(self):
        super().__init__()
        # 加载Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            logger.error("无法加载Haar级联分类器模型")
            sys.exit(1)

    def detect_faces(self, image):
        """使用OpenCV Haar级联分类器检测人脸"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # 较小的缩放步长提高精度
            minNeighbors=8,  # 较高的值减少误检
            minSize=(120, 120),  # 最小人脸尺寸
            flags=cv2.CASCADE_DO_CANNY_PRUNING
        )

        # 将结果转换为列表
        face_list = [(x, y, w, h) for (x, y, w, h) in faces]

        return face_list, [None] * len(face_list)


class OpenCVDNNFaceDetector(FaceDetector):
    """OpenCV DNN (YuNet) 人脸检测器"""

    def __init__(self, model_path="face_detection_yunet_2023mar.onnx"):
        super().__init__()
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.warning(f"找不到YuNet模型文件: {model_path}")
            logger.info(
                "您可以从以下地址下载: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet"
            )
            self.detector = None
            return

        # 初始化YuNet人脸检测器
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model_path,  # 模型路径
                "",  # 配置文件路径（空）
                (320, 320),  # 输入大小
                0.9,  # 得分阈值
                0.3,  # NMS阈值
                5000,  # 最大检测数量
            )
            logger.info("成功加载YuNet ONNX模型")
        except Exception as e:
            logger.error(f"加载YuNet模型失败: {e}")
            self.detector = None

    def detect_faces(self, image):
        """使用YuNet模型检测人脸"""
        # 如果检测器加载失败，回退到OpenCV Haar
        if self.detector is None:
            logger.warning("YuNet检测器不可用，回退到Haar级联分类器")
            fallback = OpenCVFaceDetector()
            return fallback.detect_faces(image)

        # 设置输入大小
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        # 执行检测
        _, faces = self.detector.detect(image)

        # 处理检测结果
        face_list = []
        if faces is not None:
            for face in faces:
                # YuNet返回的格式是[x, y, w, h, score, ...]
                x, y, w, h = map(int, face[:4])
                confidence = face[4]

                # 过滤低置信度的检测
                if confidence > 0.5:
                    face_list.append((x, y, w, h))
                    logger.debug(
                        f"检测到人脸: 位置=({x},{y}), 大小=({w},{h}), 置信度={confidence:.2f}"
                    )

        return face_list, [None] * len(face_list)


class MTCNNFaceDetector(FaceDetector):
    """MTCNN人脸检测器"""

    def __init__(self):
        super().__init__()
        # 检查MTCNN库是否可用
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
            self.has_mtcnn = True
            logger.info("MTCNN可用")
        except ImportError:
            logger.warning("MTCNN库不可用，请使用 'pip install mtcnn tensorflow' 安装")
            self.has_mtcnn = False
            self.detector = None

    def detect_faces(self, image):
        """使用MTCNN检测人脸"""
        # 如果MTCNN不可用，回退到OpenCV
        if not self.has_mtcnn or self.detector is None:
            logger.warning("MTCNN不可用，回退到OpenCV")
            fallback = OpenCVFaceDetector()
            return fallback.detect_faces(image)

        # 转换为RGB（MTCNN需要）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测人脸
        start_time = time.time()
        mtcnn_results = self.detector.detect_faces(rgb_image)
        detection_time = time.time() - start_time
        logger.debug(f"MTCNN检测耗时: {detection_time:.3f}秒")

        # 处理结果
        face_list = []
        for face in mtcnn_results:
            # 提取边界框坐标
            x, y, w, h = face["box"]
            confidence = face["confidence"]

            # 过滤低置信度的检测
            if confidence > 0.9:
                face_list.append((x, y, w, h))
                logger.debug(
                    f"MTCNN检测到人脸: 位置=({x},{y}), 大小=({w},{h}), 置信度={confidence:.2f}"
                )

                # 获取关键点
                keypoints = face["keypoints"]
                logger.debug(
                    f"关键点: 左眼={keypoints['left_eye']}, 右眼={keypoints['right_eye']}, 鼻子={keypoints['nose']}"
                )

        return face_list, [None] * len(face_list)


class FaceRecognitionDetector(FaceDetector):
    """face_recognition库人脸检测器"""

    def __init__(self, enable_recognition=False, known_faces_dir="known_faces"):
        super().__init__()
        # 检查face_recognition库是否可用
        try:
            import face_recognition

            self.face_recognition = face_recognition
            self.has_face_recognition = True
            logger.info("face_recognition可用")
        except ImportError:
            logger.warning(
                "face_recognition库不可用，请使用 'pip install face_recognition' 安装"
            )
            logger.info(
                "注意: 这可能需要先安装dlib，详见https://github.com/ageitgey/face_recognition"
            )
            self.has_face_recognition = False

        # 识别相关设置
        self.enable_recognition = enable_recognition
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings_by_person = {}
        self.known_face_names = []

        # 如果启用识别，加载已知人脸
        if self.enable_recognition and self.has_face_recognition:
            self.load_known_faces(known_faces_dir)

        # 设置识别参数
        self.detection_model = "hog"  # 可选: "hog"(CPU)或"cnn"(GPU)
        self.encoding_model = "small"  # 可选: "small"或"large"
        self.tolerance = 0.6  # 人脸比较的容差
        self.absolute_match_threshold = 0.45  # 绝对匹配阈值

    def load_known_faces(self, directory="known_faces"):
        """加载已知人脸图像并提取特征"""
        if not self.has_face_recognition:
            logger.error("无法加载已知人脸: face_recognition库不可用")
            return

        # 确保目录存在
        if not os.path.exists(directory):
            logger.warning(f"已知人脸目录'{directory}'不存在，正在创建")
            os.makedirs(directory)
            logger.info(f"请在'{directory}'目录中放置已知人员的图片")
            logger.info("图片命名格式: 'name.jpg'或'name1.jpg', 'name2.jpg'等")
            return

        # 获取所有图片文件
        face_images = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not face_images:
            logger.warning(f"目录'{directory}'中没有找到图片")
            return

        logger.info(f"正在加载{len(face_images)}张已知人脸图片...")

        # 临时存储每个人的所有编码
        person_encodings = {}

        # 处理每张图片
        for image_file in face_images:
            try:
                # 从文件名提取人名基础部分
                file_base = os.path.splitext(image_file)[0]
                # 使用正则表达式移除末尾的数字
                person_name = (
                    re.sub(r"\d+$", "", file_base).rstrip("_").rstrip("-").rstrip(".")
                )

                # 加载图像
                image_path = os.path.join(directory, image_file)
                image = self.face_recognition.load_image_file(image_path)

                # 尝试找到并编码人脸
                face_encodings = self.face_recognition.face_encodings(
                    image, model=self.encoding_model
                )

                if len(face_encodings) > 0:
                    # 使用第一个检测到的人脸
                    if person_name not in person_encodings:
                        person_encodings[person_name] = []

                    person_encodings[person_name].append(face_encodings[0])
                    logger.info(f"成功加载: {image_file} (识别为: {person_name})")
                else:
                    logger.warning(f"警告: 在图片{image_file}中未检测到人脸")

            except Exception as e:
                logger.error(f"处理图片{image_file}时出错: {e}")

        # 将收集的编码转换为最终格式
        self.known_face_encodings_by_person = person_encodings
        self.known_face_names = list(person_encodings.keys())

        # 打印统计信息
        total_encodings = sum(len(encodings) for encodings in person_encodings.values())
        logger.info(
            f"已加载{len(self.known_face_names)}个人的{total_encodings}个人脸编码:"
        )
        for name, encodings in person_encodings.items():
            logger.info(f"  - {name}: {len(encodings)}张照片")

    def detect_faces(self, image):
        """使用face_recognition库检测人脸"""
        # 检查face_recognition是否可用
        if not self.has_face_recognition:
            logger.warning("face_recognition不可用，回退到OpenCV")
            fallback = OpenCVFaceDetector()
            return fallback.detect_faces(image)

        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 测量性能
        start_time = time.time()

        # 使用face_recognition定位人脸
        face_locations = self.face_recognition.face_locations(
            rgb_image, model=self.detection_model
        )

        detection_time = time.time() - start_time
        logger.debug(f"人脸检测耗时: {detection_time:.3f}秒")

        # 如果不需要识别或没有已知人脸
        if not self.enable_recognition or len(self.known_face_names) == 0:
            # 转换为OpenCV格式 (x, y, w, h)
            faces = []
            for top, right, bottom, left in face_locations:
                faces.append((left, top, right - left, bottom - top))
            return faces, [None] * len(faces)

        # 提取人脸编码
        start_time = time.time()
        face_encodings = self.face_recognition.face_encodings(
            rgb_image, face_locations, model=self.encoding_model
        )
        encoding_time = time.time() - start_time
        logger.debug(f"人脸编码耗时: {encoding_time:.3f}秒")

        # 存储匹配的人脸和姓名
        matched_faces = []
        matched_names = []
        matched_distances = []

        # 存储未匹配的人脸
        unmatched_faces = []

        # 比较每个人脸与已知人脸
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            matched_person = None
            min_distance = 1.0  # 距离范围是0-1，越小越相似
            face_rect = (left, top, right - left, bottom - top)

            # 对每个已知人员进行检查
            for person_name in self.known_face_names:
                # 获取该人员的所有编码
                person_encodings = self.known_face_encodings_by_person[person_name]

                # 计算与所有编码的距离
                face_distances = self.face_recognition.face_distance(
                    person_encodings, face_encoding
                )

                # 如果有距离小于容差的编码
                if min(face_distances) < self.tolerance:
                    best_match_distance = min(face_distances)

                    # 如果这是目前最佳匹配
                    if best_match_distance < min_distance:
                        min_distance = best_match_distance
                        matched_person = person_name

            # 关键改进：只有当最佳匹配小于绝对阈值时，才认为找到了匹配
            if (
                matched_person is not None
                and min_distance < self.absolute_match_threshold
            ):
                matched_faces.append(face_rect)
                matched_names.append(matched_person)
                matched_distances.append(min_distance)
                logger.info(
                    f"匹配到人员: {matched_person}, 相似度: {(1-min_distance)*100:.1f}%"
                )
            else:
                # 记录未匹配的人脸
                unmatched_faces.append(face_rect)
                if matched_person is not None:
                    logger.debug(
                        f"拒绝低质量匹配: 与{matched_person}最佳距离为{min_distance:.3f}, 超过阈值{self.absolute_match_threshold}"
                    )

        # 如果想显示所有人脸（包括未匹配的）
        show_all_faces = True
        if show_all_faces:
            all_faces = matched_faces + unmatched_faces
            all_names = matched_names + [None] * len(unmatched_faces)
            return all_faces, all_names, matched_distances
        else:
            # 只返回匹配到的人脸
            return matched_faces, matched_names, matched_distances


def detect_and_draw(
    image_path,
    method="opencv",
    recognize=False,
    faces_dir="known_faces",
    output_path=None,
):
    """检测图片中的人脸并绘制结果"""
    # 加载图片
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法加载图片: {image_path}")
            return
    except Exception as e:
        logger.error(f"读取图片时出错: {e}")
        return

    # 创建检测器
    if method == "opencv":
        detector = OpenCVFaceDetector()
        method_name = "OpenCV Haar级联分类器"
    elif method == "opencv_dnn":
        detector = OpenCVDNNFaceDetector()
        method_name = "OpenCV DNN (YuNet)"
    elif method == "mtcnn":
        detector = MTCNNFaceDetector()
        method_name = "MTCNN多任务级联网络"
    elif method == "face_recognition":
        detector = FaceRecognitionDetector(recognize, faces_dir)
        method_name = "face_recognition库"
    else:
        logger.error(f"不支持的检测方法: {method}")
        return

    # 执行人脸检测
    start_time = time.time()

    if method == "face_recognition" and recognize:
        faces, names, distances = detector.detect_faces(image)
    else:
        faces, names = detector.detect_faces(image)
        distances = [None] * len(faces)

    detection_time = time.time() - start_time

    # 创建结果图像的副本
    result_image = image.copy()

    # 设置中文字体
    font_path = "simsun.ttc"  # 宋体
    if not os.path.exists(font_path):
        # 尝试常见的系统字体路径
        if os.path.exists("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
            font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        elif os.path.exists("C:/Windows/Fonts/simhei.ttf"):
            font_path = "C:/Windows/Fonts/simhei.ttf"

    # 绘制检测结果
    for i, (x, y, w, h) in enumerate(faces):
        # 绘制人脸框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 准备文本内容
        if i < len(names) and names[i]:
            name = names[i]
            if i < len(distances) and distances[i] is not None:
                similarity = (1 - distances[i]) * 100
                text = f"{name} ({similarity:.1f}%)"
            else:
                text = name
        else:
            text = "未知人员"

        # 绘制文本
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            # 包含中文字符
            result_image = put_chinese_text(
                result_image, text, (x, y - 30), font_path, 30, (0, 255, 0)
            )
        else:
            # 纯英文或数字
            cv2.putText(
                result_image,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    # 添加统计信息
    info_text = f"检测方法: {method_name} | 检测到 {len(faces)} 张人脸 | 耗时: {detection_time:.3f}秒"
    if any("\u4e00" <= char <= "\u9fff" for char in info_text):
        result_image = put_chinese_text(
            result_image, info_text, (10, 30), font_path, 30, (0, 255, 255)
        )
    else:
        cv2.putText(
            result_image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    # 如果指定了输出路径，保存图片
    if output_path:
        try:
            cv2.imwrite(output_path, result_image)
            logger.info(f"已保存结果图片到: {output_path}")
        except Exception as e:
            logger.error(f"保存图片时出错: {e}")

    # 显示结果
    cv2.imshow("人脸检测结果", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_image


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多方法人脸检测与识别程序")
    parser.add_argument("--image", required=True, help="要检测的图片路径")
    parser.add_argument(
        "--method",
        default="opencv",
        choices=["opencv", "opencv_dnn", "mtcnn", "face_recognition"],
        help="检测方法 (默认: opencv)",
    )
    parser.add_argument(
        "--recognize",
        action="store_true",
        help="启用人脸识别 (仅适用于face_recognition)",
    )
    parser.add_argument(
        "--faces_dir", default="known_faces", help="已知人脸目录 (默认: known_faces)"
    )
    parser.add_argument("--output", help="结果图片保存路径 (可选)")
    parser.add_argument("--debug", action="store_true", help="启用调试输出")

    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logger.level = logger.DEBUG

    # 检查图片路径
    if not os.path.exists(args.image):
        logger.error(f"找不到图片: {args.image}")
        return

    # 检测并绘制结果
    detect_and_draw(
        args.image, args.method, args.recognize, args.faces_dir, args.output
    )


if __name__ == "__main__":
    main()
