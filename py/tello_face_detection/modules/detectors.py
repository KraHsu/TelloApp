"""
检测器模块 (modules/detectors.py)
----------------------------

实现各种人脸检测算法，包括:
- OpenCV Haar级联分类器
- OpenCV DNN (YuNet ONNX)模型
- MTCNN深度学习检测器
- Face Recognition库

使用工厂模式创建不同的检测器实例。
"""

import cv2
import os
import re
import numpy as np
from abc import ABC, abstractmethod
from .utils import Logger

# 初始化日志器
logger = Logger()


class BaseFaceDetector(ABC):
    """人脸检测器基类"""

    @abstractmethod
    def detect_faces(self, image):
        """
        检测图像中的人脸

        参数:
            image: 输入图像

        返回:
            tuple: (faces, names)，其中faces是(x,y,w,h)坐标列表，names是对应的名称列表
        """
        pass


class OpenCVFaceDetector(BaseFaceDetector):
    """使用OpenCV Haar级联分类器的人脸检测器"""

    def __init__(self, config):
        """
        初始化OpenCV人脸检测器

        参数:
            config: 配置对象
        """
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_faces(self, image):
        """使用OpenCV Haar级联分类器检测人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用配置的参数
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.opencv_scale_factor,
            minNeighbors=self.config.opencv_min_neighbors,
            minSize=self.config.opencv_min_size,
        )

        # 将结果转换为列表
        face_list = [(x, y, w, h) for (x, y, w, h) in faces]

        # 返回人脸和空名称列表
        return face_list, [None] * len(face_list)


class OpenCVDNNFaceDetector(BaseFaceDetector):
    """使用OpenCV DNN (YuNet)的人脸检测器"""

    def __init__(self, config):
        """
        初始化OpenCV DNN人脸检测器

        参数:
            config: 配置对象
        """
        self.config = config
        self.detector = self._init_dnn_detector()

    def _init_dnn_detector(self):
        """初始化OpenCV DNN人脸检测器 (使用YuNet ONNX模型)"""
        try:
            # 使用YuNet ONNX模型
            model_file = "models/face_detection_yunet_2023mar.onnx"

            # 检查模型文件是否存在
            if not os.path.exists(model_file):
                logger.warning(f"警告: YuNet ONNX模型文件 '{model_file}' 不存在")
                logger.info(f"请下载模型文件到 {os.path.abspath('.')}/models 目录")
                logger.info(
                    "可从 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet 下载"
                )
                return None

            # 使用FaceDetectorYN创建检测器
            detector = cv2.FaceDetectorYN.create(
                model_file,
                "",
                (320, 320),  # 输入大小
                self.config.dnn_score_threshold,  # 得分阈值
                self.config.dnn_nms_threshold,  # NMS阈值
                self.config.dnn_max_detections,  # 最大检测数量
            )

            logger.info("YuNet ONNX人脸检测器加载成功")
            return detector
        except Exception as e:
            logger.error(f"加载YuNet ONNX模型失败: {e}")
            return None

    def detect_faces(self, image):
        """使用OpenCV YuNet ONNX模型检测人脸"""
        # 如果检测器未初始化，尝试初始化
        if self.detector is None:
            self.detector = self._init_dnn_detector()

        # 如果仍然无法初始化，回退到普通OpenCV方法
        if self.detector is None:
            logger.warning("警告: YuNet检测器不可用，回退到Haar级联分类器")
            fallback = OpenCVFaceDetector(self.config)
            return fallback.detect_faces(image)

        # 设置输入大小
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))

        # 执行检测 - YuNet返回tuple(retval, faces)
        detection_result = self.detector.detect(image)

        # 检查人脸信息是否存在
        if detection_result[1] is None:
            return [], [None]

        # 处理检测结果，转换为(x, y, w, h)格式
        face_boxes = []
        for face in detection_result[1]:
            # YuNet返回的格式是[x, y, w, h, score, landmarks...]
            x, y, w, h = map(int, face[:4])
            confidence = face[4]  # 第五个元素是置信度

            # 只有当置信度大于阈值时才添加
            if confidence > 0.3:  # 可调整置信度阈值
                face_boxes.append((x, y, w, h))

        return face_boxes, [None] * len(face_boxes)


class MTCNNFaceDetector(BaseFaceDetector):
    """使用MTCNN的人脸检测器"""

    def __init__(self, config):
        """
        初始化MTCNN人脸检测器

        参数:
            config: 配置对象
        """
        self.config = config
        self.detector = None
        self.has_mtcnn = self._init_mtcnn()

    def _init_mtcnn(self):
        """初始化MTCNN检测器"""
        try:
            from mtcnn import MTCNN

            self.detector = MTCNN()
            logger.info("MTCNN可用")
            return True
        except ImportError:
            logger.warning("MTCNN不可用，请使用 'pip install mtcnn tensorflow' 安装")
            return False

    def detect_faces(self, image):
        """使用MTCNN检测人脸"""
        if not self.has_mtcnn or self.detector is None:
            logger.warning("MTCNN不可用，回退到OpenCV检测器")
            fallback = OpenCVFaceDetector(self.config)
            return fallback.detect_faces(image)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mtcnn_results = self.detector.detect_faces(rgb_image)

        faces = []
        for face in mtcnn_results:
            # 提取边界框坐标
            x, y, w, h = face["box"]
            # 过滤低置信度检测
            if face["confidence"] > self.config.mtcnn_confidence_threshold:
                faces.append((x, y, w, h))

        return faces, [None] * len(faces)


class FaceRecognitionDetector(BaseFaceDetector):
    """使用face_recognition库的人脸检测与识别器"""

    def __init__(self, config, enable_recognition=False, known_faces_dir="known_faces"):
        """
        初始化face_recognition人脸检测器

        参数:
            config: 配置对象
            enable_recognition: 是否启用特定人员识别
            known_faces_dir: 已知人脸图像目录
        """
        self.config = config
        self.enable_recognition = enable_recognition
        self.known_faces_dir = known_faces_dir
        self.has_face_recognition = self._init_face_recognition()

        # 识别相关变量
        self.known_face_encodings_by_person = {}
        self.known_face_names = []

        # 如果启用识别，加载已知人脸
        if self.enable_recognition and self.has_face_recognition:
            self.load_known_faces(self.known_faces_dir)

    def _init_face_recognition(self):
        """初始化face_recognition库"""
        try:
            import face_recognition

            self.face_recognition = face_recognition
            logger.info("face_recognition可用")
            return True
        except ImportError:
            logger.warning(
                "face_recognition不可用，请使用 'pip install face_recognition' 安装"
            )
            logger.info(
                "注意: 这可能需要先安装dlib，详见https://github.com/ageitgey/face_recognition"
            )
            return False

    def load_known_faces(self, directory="known_faces"):
        """
        加载已知人脸图像并提取特征，支持每人多张图片

        参数:
            directory: 包含已知人脸图像的目录
        """
        if not self.has_face_recognition:
            logger.error("无法加载已知人脸: face_recognition库不可用")
            return

        if not os.path.exists(directory):
            logger.warning(f"已知人脸目录 '{directory}' 不存在，创建目录")
            os.makedirs(directory)
            logger.info(f"请在 '{directory}' 目录中放置已知人员的图片")
            logger.info("图片文件名格式: 'name.jpg' 或 'name1.jpg', 'name2.jpg' 等")
            return

        face_images = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not face_images:
            logger.warning(f"目录 '{directory}' 中没有找到图片")
            return

        logger.info(f"加载 {len(face_images)} 张已知人脸图片...")

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
                image = self.face_recognition.load_image_file(image_path)

                # 尝试找到并编码人脸
                face_encodings = self.face_recognition.face_encodings(
                    image, model=self.config.fr_encoding_model
                )

                if len(face_encodings) > 0:
                    # 使用第一个检测到的人脸
                    # 如果这个人已经有编码，则添加到列表中，否则创建新列表
                    if person_name not in person_encodings:
                        person_encodings[person_name] = []

                    person_encodings[person_name].append(face_encodings[0])
                    logger.info(f"成功加载: {image_file} (识别为: {person_name})")
                else:
                    logger.warning(f"警告: 在图片 {image_file} 中未检测到人脸")

            except Exception as e:
                logger.error(f"处理图片 {image_file} 时出错: {e}")

        # 将收集的编码转换为最终格式
        self.known_face_encodings_by_person = person_encodings
        self.known_face_names = list(person_encodings.keys())

        # 打印统计信息
        total_encodings = sum(len(encodings) for encodings in person_encodings.values())
        logger.info(
            f"已加载 {len(self.known_face_names)} 个人的 {total_encodings} 个人脸编码:"
        )
        for name, encodings in person_encodings.items():
            logger.info(f"  - {name}: {len(encodings)} 张照片")

    def detect_faces(self, image):
        """
        使用face_recognition库检测人脸，支持每人多张图片的识别

        参数:
            image: 输入图像

        返回:
            tuple: (faces, names)，faces是人脸坐标，names是对应的人名
        """
        if not self.has_face_recognition:
            logger.warning("face_recognition不可用，回退到OpenCV检测器")
            fallback = OpenCVFaceDetector(self.config)
            return fallback.detect_faces(image)

        # 转换为RGB格式（face_recognition需要）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 定位图像中的所有人脸
        face_locations = self.face_recognition.face_locations(
            rgb_image, model=self.config.fr_model
        )

        # 如果不需要识别特定人员或没有已知人脸，返回所有人脸
        if not self.enable_recognition or len(self.known_face_names) == 0:
            # 转换为OpenCV格式 (x, y, w, h)
            faces = []
            for top, right, bottom, left in face_locations:
                faces.append((left, top, right - left, bottom - top))
            return faces, [None] * len(faces)

        # 提取当前帧中所有人脸的编码
        face_encodings = self.face_recognition.face_encodings(
            rgb_image, face_locations, model=self.config.fr_encoding_model
        )

        # 存储匹配的人脸和姓名
        matched_faces = []
        matched_names = []

        # 添加额外的未匹配人脸
        unmatched_faces = []

        absolute_match_threshold = self.config.fr_absolute_match_threshold

        # 比较每个人脸与已知人脸
        for i, ((top, right, bottom, left), face_encoding) in enumerate(
            zip(face_locations, face_encodings)
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

                # 获取最小距离（最佳匹配）
                best_match_distance = min(face_distances)

                # 如果这是目前最佳匹配且小于容差
                if best_match_distance < min_distance:
                    min_distance = best_match_distance
                    matched_person = person_name

            # 关键改进：只有当最佳匹配小于绝对阈值时，才认为找到了匹配
            if matched_person is not None and min_distance < absolute_match_threshold:
                matched_faces.append(face_rect)
                matched_names.append(matched_person)
                # 可选：记录匹配距离，用于调试
                logger.info(f"匹配到人员: {matched_person}, 距离: {min_distance:.3f}")
            else:
                # 记录未匹配的人脸
                unmatched_faces.append(face_rect)
                # 可选：输出调试信息
                if matched_person is not None:
                    logger.info(
                        f"拒绝低质量匹配: 与 {matched_person} 最佳距离为 {min_distance:.3f}, 超过阈值 {absolute_match_threshold}"
                    )
                else:
                    logger.info(f"未找到匹配人员，人脸 #{i}")

        # 如果启用了"显示所有人脸"选项，则返回所有人脸（包括未匹配的）
        if self.config.show_all_faces:
            # 将未匹配人脸添加到结果中
            all_faces = matched_faces + unmatched_faces
            all_names = matched_names + [None] * len(unmatched_faces)
            return all_faces, all_names
        else:
            # 只返回匹配到的人脸
            return matched_faces, matched_names


class FaceDetectorFactory:
    """人脸检测器工厂类"""

    def __init__(self):
        """初始化工厂类，检查可用的检测方法"""
        # 检查MTCNN是否可用
        try:
            from mtcnn import MTCNN

            self.has_mtcnn = True
        except ImportError:
            self.has_mtcnn = False

        # 检查face_recognition是否可用
        try:
            import face_recognition

            self.has_face_recognition = True
        except ImportError:
            self.has_face_recognition = False

    def get_available_methods(self):
        """获取可用的检测方法列表"""
        available_methods = ["opencv", "opencv_dnn"]

        if self.has_mtcnn:
            available_methods.append("mtcnn")

        if self.has_face_recognition:
            available_methods.append("face_recognition")

        return available_methods

    def create_detector(
        self, method, enable_recognition=False, known_faces_dir="known_faces"
    ):
        """
        创建指定类型的检测器

        参数:
            method: 检测方法名称
            enable_recognition: 是否启用人脸识别
            known_faces_dir: 已知人脸目录

        返回:
            BaseFaceDetector: 检测器实例
        """
        # 创建一个配置对象
        from .config import Config

        config = Config(
            type(
                "Args",
                (),
                {
                    "method": method,
                    "recognize": enable_recognition,
                    "images": known_faces_dir,
                },
            )
        )

        # 根据方法创建对应的检测器
        if method == "opencv":
            return OpenCVFaceDetector(config)
        elif method == "opencv_dnn":
            return OpenCVDNNFaceDetector(config)
        elif method == "mtcnn" and self.has_mtcnn:
            return MTCNNFaceDetector(config)
        elif method == "face_recognition" and self.has_face_recognition:
            return FaceRecognitionDetector(config, enable_recognition, known_faces_dir)
        else:
            # 默认回退到OpenCV
            logger.warning(f"检测方法 '{method}' 不可用，回退到OpenCV")
            return OpenCVFaceDetector(config)
