import logging
from typing import *
from djitellopy import Tello
import time
import threading
import cv2
import os
import numpy as np
from datetime import datetime

# Initialize an empty list to store IMU data
imu_data = []


def get_imu(tl: Tello):
    """Gets yaw, pitch, and roll from the Tello drone."""
    # Return the values instead of just getting them
    return tl.get_yaw(), tl.get_pitch(), tl.get_roll()


if __name__ == "__main__":

    tl = Tello()

    Tello.LOGGER.setLevel(logging.ERROR)

    tl.connect()
    print("电池电量:", tl.get_battery(), "%")

    try:
        print("开始收集IMU数据... 按 Ctrl+C 停止。")
        while True:
            # Get the IMU data
            yaw, pitch, roll = get_imu(tl)
            # Append the data as a list or tuple to the main list
            imu_data.append([yaw, pitch, roll])
            # Optional: Add a small delay to control sampling rate
            time.sleep(0.1)  # Sample approx 10 times per second
    except KeyboardInterrupt:
        print("停止收集数据。")

    finally:
        print("任务结束，正在保存数据...")
        # Check if any data was collected
        if imu_data:
            # Convert the list to a NumPy array for easier saving
            imu_data_np = np.array(imu_data)
            # Generate a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"imu_data_{timestamp}.csv"
            # Define the header for the CSV file
            header = "Yaw,Pitch,Roll"
            # Save the data to a CSV file with the header
            np.savetxt(filename, imu_data_np, delimiter=",", header=header, comments="")
            print(f"数据已保存到 {filename}")
        else:
            print("没有收集到数据。")

        # Ensure the drone connection is closed if it was opened
        if tl.is_flying:  # Land if still flying
            print("正在降落...")
            tl.land()
        if tl.stream_on:  # Turn off stream if on
            tl.streamoff()
        # No explicit disconnect needed for Tello object usually,
        # but good practice if available or necessary for resource cleanup.
        print("程序退出。")
