from typing import *
from djitellopy import Tello
import time
import threading
import cv2
import os
import numpy as np
from datetime import datetime

# 添加face_recognition库导入
try:
    import face_recognition
    has_face_recognition = True
    print("face_recognition可用")
except ImportError:
    has_face_recognition = False
    print("face_recognition不可用，请使用 'pip install face_recognition' 安装")
    print("注意: 这可能需要先安装dlib，详见https://github.com/ageitgey/face_recognition")

# 飞行速度和控制参数
SPEED = 20
SEARCH_SPEED = 15
FACE_DISTANCE = 120  # 距离人脸多远停止 (TOF传感器读数，单位cm)
RECOGNITION_TIME = 3  # 识别时间（秒）

# 创建目录
if not os.path.exists('videos'):
    os.makedirs('videos')
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')
    print("请将组员照片放入 'known_faces' 文件夹中")

# 全局变量
frame_read = None
show = True
recording = False
video_writer = None
recording_start_time = None
face_detected = False
current_member = 0  # 当前识别的组员索引
members_recognized = [False, False]  # 两位组员的识别状态

# 人脸识别相关变量
known_face_encodings = []
known_face_names = []

# 加载备用的人脸检测器(如果face_recognition不可用)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_known_faces(directory='known_faces'):
    """加载已知人脸图像并提取特征"""
    global known_face_encodings, known_face_names
    
    if not has_face_recognition:
        print("无法加载已知人脸: face_recognition库不可用")
        return
    
    if not os.path.exists(directory):
        print(f"已知人脸目录 '{directory}' 不存在，创建目录")
        os.makedirs(directory)
        print(f"请在 '{directory}' 目录中放置已知人员的图片")
        print("图片文件名将被用作人员名称 (例如: 'member1.jpg')")
        return
    
    face_images = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not face_images:
        print(f"目录 '{directory}' 中没有找到图片")
        return
    
    print(f"加载 {len(face_images)} 张已知人脸图片...")
    
    for image_file in face_images:
        try:
            # 从文件名获取人名 (不含扩展名)
            person_name = os.path.splitext(image_file)[0]
            
            # 加载图像
            image_path = os.path.join(directory, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # 尝试找到并编码人脸
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                # 使用第一个检测到的人脸
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
                print(f"成功加载: {person_name}")
            else:
                print(f"警告: 在图片 {image_file} 中未检测到人脸")
                
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {e}")
    
    print(f"已加载 {len(known_face_encodings)} 个已知人脸编码")

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
        return int(tl.send_read_command('EXT tof?')[4:])
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    """使用face_recognition检测图像中的人脸并识别指定人员"""
    if not has_face_recognition or len(known_face_encodings) == 0:
        # 备用方案：使用OpenCV检测人脸
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        # 返回所有人脸位置和空的名称列表(因为无法识别)
        return faces, [None] * len(faces)
    
    # 使用face_recognition检测和识别人脸
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定位图像中的所有人脸
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    # 如果没有人脸，返回空列表
    if len(face_locations) == 0:
        return [], []
    
    # 提取人脸编码
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    faces = []
    names = []
    
    # 识别每个人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 将人脸位置转换为OpenCV格式 (x, y, w, h)
        x, y = left, top
        w, h = right - left, bottom - top
        faces.append((x, y, w, h))
        
        # 与已知人脸比较
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = None
        
        # 如果找到匹配，使用最佳匹配
        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        names.append(name)
    
    return faces, names

def draw_status(image, faces, names):
    """在图像上显示状态信息和人脸框，区分已知人员和未知人员"""
    # 绘制状态信息
    status_text = f"Rec progress: {sum(members_recognized)}/2"
    cv2.putText(image, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 如果正在录制，显示录制指示器
    if recording:
        elapsed = int((datetime.now() - recording_start_time).total_seconds())
        mins, secs = divmod(elapsed, 60)
        timer_text = f"REC {mins:02d}:{secs:02d}"
        cv2.circle(image, (image.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(image, timer_text, (image.shape[1] - 200, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 绘制人脸框，区分已知人员和未知人员
    for i, (x, y, w, h) in enumerate(faces):
        name = names[i] if i < len(names) else None
        
        if name is not None:
            # 已知人员，绘制绿色框
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 如果是已识别的组员
            if face_detected and current_member < len(members_recognized) and members_recognized[current_member]:
                label = f"mem {current_member+1} get!"
                cv2.putText(image, label, (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 未知人员，绘制红色框
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image

def show_img():
    """显示视频流并录制"""
    global recording, video_writer
    
    while show:
        if frame_read is not None:
            display_frame = frame_read.frame
            
            # 检测人脸并识别
            faces, names = detect_faces(display_frame)
            
            # 绘制状态和人脸框
            display_frame = draw_status(display_frame, faces, names)
            
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
    """搜索人脸的逻辑，只识别指定人员"""
    global face_detected, frame_read
    
    # 首先尝试按螺旋形搜索
    for _ in range(8):  # 控制搜索的圈数
        if face_detected:
            return
        
        # 旋转搜索
        for _ in range(8):  # 每圈旋转8次，每次45度
            if frame_read is not None:
                faces, names = detect_faces(frame_read.frame)
                
                # 只处理已知人脸
                known_faces = []
                known_indices = []
                
                for i, (face, name) in enumerate(zip(faces, names)):
                    if name is not None:  # 已知人脸
                        known_faces.append(face)
                        known_indices.append(i)
                
                if known_faces:
                    # 找到已知人脸，调整位置
                    face_detected = adjust_position(tl, known_faces[0], names[known_indices[0]])
                    if face_detected:
                        return
            
            # 没找到已知人脸，继续旋转
            tl.rotate_counter_clockwise(45)
            time.sleep(1)
        
        # 完成一圈旋转后，向前移动一些
        tl.move_forward(50)
        time.sleep(1)

def adjust_position(tl: Tello, face, name):
    """调整无人机位置到指定人脸前方合适距离"""
    global frame_read
    
    if name is None:  # 非指定人员
        return False
    
    x, y, w, h = face
    
    # 水平调整（左右）
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        if frame_read is not None:
            faces, names = detect_faces(frame_read.frame)
            
            # 检查是否仍能看到指定人员
            found = False
            for i, n in enumerate(names):
                if n == name:  # 找到同一个人
                    found = True
                    break
            
            if found:
                return True  # 成功找到指定人员
            
        attempts += 1
        time.sleep(0.5)
    
    return False

if __name__ == "__main__":
    tl = Tello()
    
    try:
        # 连接到Tello
        tl.connect()
        print("电池电量:", tl.get_battery(), "%")
        
        # 加载已知人脸
        load_known_faces()
        
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
        if 'video_thread' in locals():
            video_thread.join()
