import cv2
import numpy as np

# 加载预训练的人脸检测器（Haar级联分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 打开摄像头（0表示默认摄像头）
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    
    # 如果无法读取帧，退出循环
    if not ret:
        print("无法获取摄像头画面")
        break
    
    # 将图像转换为灰度图，提高人脸检测效率
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # 在检测到的人脸周围绘制矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # 显示结果
    cv2.imshow('人脸检测', frame)
    
    # 按ESC键退出
    if cv2.waitKey(1) == 27:  # 27是ESC键的ASCII码
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
