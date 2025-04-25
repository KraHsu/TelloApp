import time


class PIDController:
    """一个简单的PID控制器"""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        setpoint: float,
        output_limits=(-100, 100),
        int_limits=(-50, 50),
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.int_limits = int_limits

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_error = 0

        self.reset()

    def update(self, current_value, dt=None):
        current_time = time.monotonic()
        if dt == None:
            try:
                dt = (
                    current_time - self._last_time
                    if self._last_time is not None
                    else 1.0 / 30.0
                )
            except TypeError:
                dt = 1.0 / 30.0

        if dt <= 0:
            return self.clamp(self._proportional + self._integral + self._derivative)

        error = self.setpoint - current_value

        self._proportional = self.Kp * error

        self._integral += self.Ki * error * dt
        self._integral = self.clamp(self._integral, self.int_limits)

        if self._last_time is not None:
            delta_error = error - self._last_error
            self._derivative = self.Kd * (delta_error / dt)
        else:
            self._derivative = 0

        output = self._proportional + self._integral + self._derivative

        output = self.clamp(output, self.output_limits)

        self._last_error = error
        self._last_time = current_time

        return output

    def reset(self):
        self._proportional = 0
        self._integral = 0
        self._derivative = 0
        self._last_error = 0
        self._last_time = None

    def set_setpoint(self, new_setpoint):
        self.setpoint = new_setpoint
        self.reset()

    def set_tunings(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def clamp(self, value, limits=None):
        if limits is None:
            limits = self.output_limits
        lower, upper = limits
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value
