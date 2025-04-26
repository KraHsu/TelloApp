import numpy as np
from Utils import *


def smooth_clamp(x, a, b, delta):
    if a >= b:
        raise ValueError("Lower bound 'a' must be less than upper bound 'b'.")
    if delta <= 0:
        return np.clip(x, a, b)
    # Ensure delta does not exceed half the interval to avoid overlap
    delta = min(delta, (b - a) / 2)

    # Process lower boundary
    if x <= a + delta:
        t = x - a
        y = a + (2 / delta) * t**2 - (1 / delta**2) * t**3
    # Process upper boundary
    elif x >= b - delta:
        t = x - (b - delta)
        y = (b - delta) + t + (t**2) / delta - (t**3) / (delta**2)
    # Middle region remains linear
    else:
        y = x
    # Clip the result to handle cases where delta is too large
    return np.clip(y, a, b)


# 示例使用
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    l = 0
    r = 5

    a = 20
    b = 40
    delta = 5
    g = lambda t: 4 * sigmoid(t) * (1 - sigmoid(t))
    h = lambda t: 50 * g(t / 1.5) + 10
    clamp = create_smooth_clamp(a, b, delta)
    x = np.linspace(l, r, 500)
    y = [clamp(h(xi)) for xi in x]
    
    hy = [h(xi) for xi in x]

    plt.plot(x, y, label="Smooth Clamp")
    plt.plot([l - 1, l - 1], [0, l - 1], "k--", alpha=0.3)
    plt.plot([r + 1, r + 1], [0, r + 1], "k--", alpha=0.3)
    plt.plot(x, np.clip(hy, a, b), "r--", alpha=0.5, label="Hard Clamp")
    plt.title(f"Smooth Clamping between {a} and {b} with delta {delta}")
    plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()
