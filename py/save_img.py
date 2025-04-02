from djitellopy import Tello
import cv2
import time
import os
import argparse

def create_save_dir(person_name):
    """创建保存照片的目录"""
    # 如果没有指定人名，使用时间戳
    if not person_name:
        person_name = f"person_{int(time.time())}"
    
    # 确保known_faces目录存在
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
    
    # 计算此人已有多少照片
    existing_files = [f for f in os.listdir("known_faces") 
                     if f.startswith(person_name) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Person name: {person_name}")
    print(f"Existing photos: {len(existing_files)}")
    return person_name

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Capture photos from Tello camera')
    parser.add_argument('-n', '--name', type=str, default='',
                        help='Person name (e.g., "john")')
    parser.add_argument('-w', '--webcam', action='store_true',
                        help='Use computer webcam instead of Tello')
    args = parser.parse_args()
    
    # 创建保存目录
    person_name = create_save_dir(args.name)
    
    # 照片计数
    photo_count = 0
    
    try:
        # 选择视频源
        if args.webcam:
            print("Using computer webcam...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open webcam")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 显示帧
                cv2.putText(frame, f"Photos: {photo_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE to capture, ESC to exit", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Photo Capture", frame)
                
                # 检测按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break
                elif key == 32:  # 空格键
                    # 保存照片
                    filename = f"known_faces/{person_name}{photo_count+1}.jpg"
                    cv2.imwrite(filename, frame)
                    photo_count += 1
                    print(f"Saved photo: {filename}")
                    
                    # 短暂显示保存提示
                    save_frame = frame.copy()
                    cv2.putText(save_frame, "Photo Saved!", (frame.shape[1]//2-100, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow("Photo Capture", save_frame)
                    cv2.waitKey(500)  # 显示500毫秒
                
                time.sleep(0.03)  # 限制帧率
                
            cap.release()
            
        else:
            # 使用Tello无人机
            print("Connecting to Tello...")
            tello = Tello()
            tello.connect()
            
            battery = tello.get_battery()
            print(f"Battery: {battery}%")
            
            tello.streamon()
            print("Video stream started")
            
            # 等待视频流初始化
            time.sleep(2)
            
            frame_read = tello.get_frame_read()
            
            while True:
                # 获取帧
                frame = frame_read.frame
                
                # 显示帧
                cv2.putText(frame, f"Photos: {photo_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE to capture, ESC to exit", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Photo Capture", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 检测按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break
                elif key == 32:  # 空格键
                    # 保存照片
                    filename = f"known_faces/{person_name}{photo_count+1}.jpg"
                    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    photo_count += 1
                    print(f"Saved photo: {filename}")
                    
                    # 短暂显示保存提示
                    save_frame = frame.copy()
                    cv2.putText(save_frame, "Photo Saved!", (frame.shape[1]//2-100, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow("Photo Capture", cv2.cvtColor(save_frame, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(500)  # 显示500毫秒
                
                time.sleep(0.03)  # 限制帧率
            
            # 关闭视频流
            tello.streamoff()
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print(f"Total photos saved: {photo_count}")
        print(f"Photos saved as known_faces/{person_name}[1-{photo_count}].jpg")

if __name__ == "__main__":
    main()
