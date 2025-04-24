"""
配置模块 (modules/config.py)
------------------------

负责系统的配置参数管理，包括检测方法、识别模式、性能参数等。
此模块集中管理所有可配置参数，便于系统维护和扩展。
"""


class Config:
    """系统配置类"""

    def __init__(self, args):
        """
        初始化配置

        参数:
            args: 解析后的命令行参数
        """
        # 检测方法配置
        self.current_method = args.method
        self.detection_methods = ["opencv", "opencv_dnn", "mtcnn", "face_recognition"]

        # 识别模式配置
        self.enable_recognition = args.recognize
        self.known_faces_dir = args.images

        # 性能配置
        self.max_reuse_frames = 60  # 最多重复使用结果的帧数

        # OpenCV检测器参数
        self.opencv_scale_factor = 1.05
        self.opencv_min_neighbors = 10
        self.opencv_min_size = (120, 120)

        # OpenCV DNN检测器参数
        self.dnn_score_threshold = 0.9
        self.dnn_nms_threshold = 0.3
        self.dnn_max_detections = 5000

        # MTCNN检测器参数
        self.mtcnn_confidence_threshold = 0.9

        # Face Recognition参数
        self.fr_model = "cnn"  # 可选: "hog"(CPU)或"cnn"(GPU)
        self.fr_tolerance = 0.6
        self.fr_encoding_model = "large"  # 可选: "small"或"large"

        # 匹配阈值和控制显示所有人脸的选项
        self.fr_absolute_match_threshold = 0.4  # 绝对匹配阈值
        self.show_all_faces = True  # 是否显示所有人脸（包括未匹配的）

        # UI配置
        self.font = "cv2.FONT_HERSHEY_SIMPLEX"
        self.font_scale = 0.7
        self.box_color = (0, 255, 0)  # BGR格式
        self.text_color = (0, 255, 0)  # BGR格式
        self.text_thickness = 2
