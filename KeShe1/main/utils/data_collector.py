# -*- coding: utf-8 -*-
"""
数据收集器模块
"""
import os
import csv
import time
from datetime import datetime


class DataCollector:
    """
    数据收集器类，用于收集和保存飞行数据

    参数:
        file_path (str): 保存数据的CSV文件路径
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.headers = set()
        self.start_time = time.time()

    def collect_datas(self, data_points):
        """
        收集一组数据点

        参数:
            data_points (list): 包含 (key, value) 键值对的列表
        """
        timestamp = time.time() - self.start_time
        entry = {"timestamp": timestamp}

        for key, value in data_points:
            entry[key] = value
            self.headers.add(key)

        self.data.append(entry)

    def collect_data(self, key, value):
        """
        收集单个数据点

        参数:
            key (str): 数据点名称
            value: 数据点值
        """
        self.collect_datas([(key, value)])

    def save_to_csv(self):
        """
        将收集的数据保存为CSV文件
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)

        # 准备表头
        fieldnames = ["timestamp"] + sorted(list(self.headers))

        # 写入CSV
        with open(self.file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.data:
                # 确保所有字段都存在
                row = {field: entry.get(field, "") for field in fieldnames}
                writer.writerow(row)

        print(f"数据已保存至: {self.file_path}")

    def get_data_summary(self):
        """
        生成数据摘要

        返回:
            dict: 包含数据摘要的字典
        """
        if not self.data:
            return {"数据点数量": 0}

        summary = {
            "数据点数量": len(self.data),
            "记录时长(秒)": self.data[-1]["timestamp"] - self.data[0]["timestamp"],
            "记录开始时间": datetime.fromtimestamp(self.start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "数据类型": list(self.headers),
        }

        return summary
