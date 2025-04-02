# 关于人脸误识别的解决方案

## 仍然使用 Opencv Haar级联检测器

参数介绍
```py
faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,      
            minNeighbors=10,         
            minSize=(120, 120)      
        )
```

