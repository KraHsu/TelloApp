import numpy as np
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_smooth_clamp(l: float, r: float, delta: float):
    """
    创建一个函数f，该函数f实现一个平滑的钳位（clamp）功能。

    Args:
        l: 钳位范围的下界。
        r: 钳位范围的上界。
        delta: 在 l 和 r 附近的过渡区域的大小。
               要求 l < r, delta > 0, 并且 2 * delta <= r - l
               以保证中间的线性区域 [l + delta, r - delta] 存在或至少为一个点。

    Returns:
        一个函数 f(x: float) -> float，该函数实现：
        - 当 x <= l 时, f(x) = l
        - 当 x >= r 时, f(x) = r
        - 当 l + delta <= x <= r - delta 时, f(x) = x
        - 当 l < x < l + delta 时, f(x) 从 l 平滑过渡到 x
        - 当 r - delta < x < r 时, f(x) 从 x 平滑过渡到 r
        该返回的函数处处连续且一阶可导。

    Raises:
        ValueError: 如果参数不满足 l < r, delta > 0 或 2 * delta > r - l 的条件。
        TypeError: 如果 l, r, delta 不是数值类型。
    """
    if not isinstance(l, (int, float)) or not isinstance(r, (int, float)) or not isinstance(delta, (int, float)):
         raise TypeError("l, r, and delta must be numeric (int or float)")

    # 参数检查
    if l >= r:
        raise ValueError("下界 l 必须严格小于上界 r。")
    if delta <= 0:
        raise ValueError("过渡区域大小 delta 必须大于 0。")
    # 检查过渡区是否重叠或超出范围
    if 2 * delta > r - l:
        raise ValueError("delta 过大：2 * delta 必须小于或等于 r - l。")

    l_plus_delta = l + delta
    r_minus_delta = r - delta

    def f(x: float) -> float:
        """
        实际执行平滑钳位操作的函数。

        Args:
            x: 输入值。

        Returns:
            经过平滑钳位处理后的值。

        Raises:
            TypeError: 如果 x 不是数值类型。
        """
        if not isinstance(x, (int, float)):
             raise TypeError("Input x must be numeric (int or float)")

        # 转换x为float进行计算
        x = float(x)

        # 1. 低于下界区域
        if x <= l:
            return float(l)

        # 2. 左侧过渡区域 [l, l + delta)
        elif x < l_plus_delta:
            # 使用三次 Hermite 插值
            # t 从 0 变化到 1
            t = (x - l) / delta
            # G(t) = delta * (2*t^2 - t^3)
            # f(x) = l + G(t)
            g_t = delta * (2 * t**2 - t**3)
            return float(l + g_t)

        # 3. 中间线性区域 [l + delta, r - delta]
        elif x <= r_minus_delta:
            # y = x
            return float(x)

        # 4. 右侧过渡区域 (r - delta, r)
        elif x < r:
            # 使用三次 Hermite 插值
            # t 从 0 变化到 1
            t = (x - r_minus_delta) / delta
            # H(t) = delta * (-t^3 + t^2 + t)
            # f(x) = r_minus_delta + H(t)
            h_t = delta * (-t**3 + t**2 + t)
            return float(r_minus_delta + h_t)

        # 5. 高于上界区域 [r, +inf)
        else: # x >= r
            return float(r)

    return f