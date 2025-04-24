# Tello无人机人脸识别系统

## 项目概述

本项目是《2025智能无人系统综合实践II》课程的技术分享项目，实现了基于DJI Tello无人机的实时人脸检测与识别功能。系统支持多种人脸检测算法，包括传统的OpenCV Haar级联分类器、基于深度学习的OpenCV DNN (YuNet)、MTCNN和face_recognition库，并支持特定人员识别功能。

## 主要功能

- 实时视频流获取与处理
- 多种人脸检测算法支持与切换
- 特定人员识别功能（基于face_recognition）
- 多线程设计，提高系统响应速度
- 用户友好的交互界面

## 系统架构

本项目采用模块化设计，主要包含以下模块：

- **config**: 配置参数管理
- **detectors**: 实现不同的人脸检测算法
- **video_handler**: 视频流处理
- **utils**: 工具函数，包括日志、图像处理等
- **ui**: 用户界面组件

## 安装与依赖

### 环境要求

- Python 3.8 或更高版本
- DJI Tello 无人机

### 安装步骤

1. 克隆项目仓库

```bash
git clone https://github.com/yourusername/tello-face-recognition.git
cd tello-face-recognition
```

2. 安装依赖库

```bash
pip install -r requirements.txt
```

3. 安装可选依赖（如果需要高级功能）

**注意**：face_recognition 库需要先安装 dlib，这在某些平台可能较为复杂。详细安装步骤请参考 [face_recognition 安装指南](https://github.com/ageitgey/face_recognition#installation)。

## 使用方法

### 基本使用

```bash
python tello_face_recognition.py
```

### 命令行参数

```bash
python tello_face_recognition.py [-h] [-m METHOD] [-r] [-i IMAGES]
```

参数说明：
- `-m, --method`: 选择人脸检测方法: opencv, opencv_dnn, mtcnn, 或 face_recognition（默认: opencv）
- `-r, --recognize`: 启用特定人员识别模式（仅适用于face_recognition方法）
- `-i, --images`: 包含已知人脸图像的文件夹路径（默认: known_faces）

### 添加已知人脸

如果要使用特定人员识别功能，需要在`known_faces`目录中放置人脸图片：

1. 每个人的图片文件名应以该人的名字命名，可以包含多张图片
2. 图片命名格式：`name.jpg` 或 `name1.jpg`, `name2.jpg` 等
3. 确保图片中有清晰的人脸

### 键盘控制

在运行过程中，可以使用以下按键：

- `空格`: 切换人脸检测方法
- `r`: 切换识别模式（仅当使用face_recognition方法时有效）
- `ESC`: 退出程序

## 高级功能

### YuNet ONNX模型

本项目支持使用OpenCV DNN与YuNet ONNX模型进行人脸检测，此方法比传统Haar级联分类器更精确。您可以从[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)下载模型文件，并放置在`models`目录中。

### 多人脸识别

当使用face_recognition库时，系统支持每人多张图片的识别，可提高识别准确率。建议为每个人提供3-5张不同角度、表情和光照条件下的照片。

## 性能提示

- OpenCV Haar级联分类器速度最快但准确率较低
- MTCNN和face_recognition准确率高但需要更多计算资源
- 在低性能设备上，推荐使用opencv或opencv_dnn方法
- 检测结果会自动缓存以提高帧率

## 技术细节

详细的技术实现说明请参见[技术文档](docs/technical_details.md)。

## 贡献与开发

欢迎提交问题报告和改进建议。如需贡献代码，请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request