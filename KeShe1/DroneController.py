import time
import random
from enum import Enum, auto


# 定义无人机的状态
class DroneState(Enum):
    IDLE = auto()  # 静止在地面，待命
    TAKING_OFF = auto()  # 正在起飞
    FLYING_TO_CARD1 = auto()  # 正在飞往挑战卡1
    AT_CARD1_POSITIONING = auto()  # 到达卡1，正在调整位置准备拍摄
    AT_CARD1_SHOOTING = auto()  # 正在拍摄卡1
    FLYING_TO_CARD2 = auto()  # 正在飞往挑战卡2
    AT_CARD2_POSITIONING = auto()  # 到达卡2，识别目标并调整位置
    AT_CARD2_SHOOTING = auto()  # 正在拍摄卡2的目标
    FLYING_TO_CARD3 = auto()  # 正在飞往挑战卡3
    AT_CARD3_POSITIONING = auto()  # 到达卡3，识别目标并调整位置
    AT_CARD3_SHOOTING = auto()  # 正在拍摄卡3的目标
    RETURNING_TO_HOME = auto()  # 正在返回起飞区
    LANDING = auto()  # 正在降落
    LANDED = auto()  # 已降落，任务完成
    ERROR = auto()  # 发生错误


# 状态控制器类
class DroneController:
    def __init__(self):
        self.current_state = DroneState.IDLE
        print(f"无人机初始化完毕，当前状态: {self.current_state.name}")

    def _transition_to(self, new_state):
        """处理状态转换"""
        print(f"状态转换: {self.current_state.name} -> {new_state.name}")
        self.current_state = new_state

    def start_mission(self):
        """开始任务"""
        if self.current_state == DroneState.IDLE:
            print("任务开始指令收到！")
            self._transition_to(DroneState.TAKING_OFF)
        else:
            print(f"无法在 {self.current_state.name} 状态下开始任务。")

    def update(self):
        """
        状态机核心更新逻辑。
        在实际应用中，这里会检查传感器数据、GPS、任务完成信号等。
        为了演示，我们这里使用简单的逻辑和模拟延迟/成功。
        """
        if self.current_state == DroneState.IDLE:
            # 等待开始指令，由 start_mission() 触发
            pass  # 在 IDLE 状态下无自动行为

        elif self.current_state == DroneState.TAKING_OFF:
            print("  正在执行起飞程序...")
            # 模拟起飞过程
            time.sleep(1)  # 模拟耗时
            print("  起飞完成，达到预定高度。")
            self._transition_to(DroneState.FLYING_TO_CARD1)

        elif self.current_state == DroneState.FLYING_TO_CARD1:
            print("  正在飞往挑战卡1...")
            # 模拟飞行过程
            time.sleep(2)  # 模拟耗时
            print("  已到达挑战卡1位置。")
            self._transition_to(DroneState.AT_CARD1_POSITIONING)

        elif self.current_state == DroneState.AT_CARD1_POSITIONING:
            print("  正在调整姿态和位置，准备拍摄挑战卡1...")
            # 模拟定位和调整过程
            time.sleep(0.5)  # 模拟耗时
            print("  位置调整完毕。")
            self._transition_to(DroneState.AT_CARD1_SHOOTING)

        elif self.current_state == DroneState.AT_CARD1_SHOOTING:
            print("  正在拍摄挑战卡1...")
            # 模拟拍摄过程
            time.sleep(0.5)  # 模拟耗时
            print("  拍摄挑战卡1完成。")
            self._transition_to(DroneState.FLYING_TO_CARD2)

        elif self.current_state == DroneState.FLYING_TO_CARD2:
            print("  正在飞往挑战卡2...")
            time.sleep(2)
            print("  已到达挑战卡2位置。")
            self._transition_to(DroneState.AT_CARD2_POSITIONING)

        elif self.current_state == DroneState.AT_CARD2_POSITIONING:
            print("  正在识别目标并调整位置，准备拍摄挑战卡2旁目标...")
            # 模拟识别和定位过程 (可能成功或失败)
            time.sleep(1)
            if random.random() > 0.1:  # 90% 成功率
                print("  目标识别成功，位置调整完毕。")
                self._transition_to(DroneState.AT_CARD2_SHOOTING)
            else:
                print("  错误：目标识别或定位失败！")
                self._transition_to(DroneState.ERROR)

        elif self.current_state == DroneState.AT_CARD2_SHOOTING:
            print("  正在拍摄挑战卡2旁目标...")
            time.sleep(0.5)
            print("  拍摄挑战卡2旁目标完成。")
            self._transition_to(DroneState.FLYING_TO_CARD3)

        elif self.current_state == DroneState.FLYING_TO_CARD3:
            print("  正在飞往挑战卡3...")
            time.sleep(2)
            print("  已到达挑战卡3位置。")
            self._transition_to(DroneState.AT_CARD3_POSITIONING)

        elif self.current_state == DroneState.AT_CARD3_POSITIONING:
            print("  正在识别目标并调整位置，准备拍摄挑战卡3旁目标...")
            time.sleep(1)
            if random.random() > 0.1:  # 90% 成功率
                print("  目标识别成功，位置调整完毕。")
                self._transition_to(DroneState.AT_CARD3_SHOOTING)
            else:
                print("  错误：目标识别或定位失败！")
                self._transition_to(DroneState.ERROR)

        elif self.current_state == DroneState.AT_CARD3_SHOOTING:
            print("  正在拍摄挑战卡3旁目标...")
            time.sleep(0.5)
            print("  拍摄挑战卡3旁目标完成。")
            self._transition_to(DroneState.RETURNING_TO_HOME)

        elif self.current_state == DroneState.RETURNING_TO_HOME:
            print("  正在返回起飞区域...")
            time.sleep(3)  # 模拟返航耗时
            print("  已到达起飞区域上方。")
            self._transition_to(DroneState.LANDING)

        elif self.current_state == DroneState.LANDING:
            print("  正在执行降落程序...")
            time.sleep(1.5)  # 模拟降落耗时
            print("  降落成功。")
            self._transition_to(DroneState.LANDED)

        elif self.current_state == DroneState.LANDED:
            print("任务完成，无人机已安全降落。")
            # 保持在此状态或可以重置回 IDLE
            pass  # 最终状态

        elif self.current_state == DroneState.ERROR:
            print("错误状态：任务中断。需要人工干预。")
            # 在此状态下停止进一步操作
            pass  # 错误状态

    def run_mission_simulation(self):
        """运行完整的任务模拟"""
        if self.current_state != DroneState.IDLE:
            print("请先确保无人机处于 IDLE 状态。")
            return

        self.start_mission()

        # 模拟状态机的持续更新，直到任务完成或出错
        while self.current_state not in [DroneState.LANDED, DroneState.ERROR]:
            self.update()
            # 在实际应用中，这个 update 会被一个更高频率的循环调用
            # 这里我们为了模拟，每次 update 后稍微暂停一下
            time.sleep(0.1)

        # 处理最终状态
        if self.current_state == DroneState.LANDED:
            print("\n--- 任务成功结束 ---")
        elif self.current_state == DroneState.ERROR:
            print("\n--- 任务因错误中止 ---")


# --- 主程序入口 ---
if __name__ == "__main__":
    controller = DroneController()
    # 运行模拟
    controller.run_mission_simulation()

    # 可以在这里添加代码尝试在错误状态或结束后再次启动，以测试状态保护
    # print("\n尝试在非IDLE状态下启动任务:")
    # controller.start_mission()
