import csv
import time
from datetime import datetime


class DataCollector:
    def __init__(self, filename="data.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.data = []
        self.column_names = set()  # 用于存储所有出现过的数据名称

    def collect(self, name, value):
        """采集一个带名称的浮点数据点"""
        timestamp = (time.time() - self.start_time) * 1000  # 转换为毫秒
        self.column_names.add(name)  # 记录数据名称
        self.data.append((timestamp, name, value))

    def collect_datas(self, datalist, timestamp=None):
        """采集一个带名称的浮点数据点"""
        if timestamp == None:
            timestamp = (time.time() - self.start_time) * 1000  # 转换为毫秒
        for name, value in datalist:
            self.column_names.add(name)  # 记录数据名称s
            self.data.append((timestamp, name, value))

    def save_to_csv(self):
        """保存数据到CSV文件"""
        # 将列名排序，确保顺序一致
        column_names = sorted(list(self.column_names))
        headers = ["Timestamp(ms)"] + column_names

        # 创建数据字典，用于存储每个时间点的所有数据
        data_dict = {}
        for timestamp, name, value in self.data:
            if timestamp not in data_dict:
                data_dict[timestamp] = {name: value}
            else:
                data_dict[timestamp][name] = value

        # 写入CSV文件
        with open(self.filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # 写入表头

            # 按时间戳排序写入数据
            for timestamp in sorted(data_dict.keys()):
                row = [timestamp]
                for name in column_names:
                    row.append(data_dict[timestamp].get(name, ""))
                writer.writerow(row)

        print(f"数据已保存到 {self.filename}")


# 使用示例
if __name__ == "__main__":
    collector = DataCollector("example_data.csv")

    # 模拟采集不同名称的数据
    import random

    for _ in range(10):
        collector.collect("temperature", random.uniform(20, 30))
        collector.collect("pressure", random.uniform(900, 1100))
        collector.collect("humidity", random.uniform(40, 60))
        time.sleep(0.1)  # 模拟数据采集间隔

    collector.save_to_csv()
