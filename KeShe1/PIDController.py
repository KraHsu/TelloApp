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
        """
        Initialize the PID controller.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            setpoint (float): The target value the controller tries to achieve.
            output_limits (tuple): A tuple (min_output, max_output) to clamp the controller output.
                                   Defaults to Tello's typical RC range (-100, 100).
            int_limits (tuple):
        """
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

        self.reset()  # Initialize properly

    def update(self, current_value, dt=None):
        """
        Calculate the PID output based on the current value.

        Args:
            current_value (float): The current measured value from the system.

        Returns:
            float: The calculated control output, clamped within the output limits.
        """
        current_time = time.monotonic()  # Use monotonic clock for time differences
        if dt == None:
            try:
                dt = (
                    current_time - self._last_time
                    if self._last_time is not None
                    else 1.0 / 30.0
                )  # Estimate dt for first run
            except (
                TypeError
            ):  # Handle potential None subtraction if reset wasn't perfect
                dt = 1.0 / 30.0  # Default to ~30fps timeframe if last_time is invalid

        if dt <= 0:  # Avoid division by zero or negative time travel
            # Reuse last output or return 0 if no output yet
            return self.clamp(self._proportional + self._integral + self._derivative)

        error = self.setpoint - current_value

        # Proportional term
        self._proportional = self.Kp * error

        # Integral term (with anti-windup via clamping)
        self._integral += self.Ki * error * dt
        self._integral = self.clamp(
            self._integral, self.int_limits
        )  # Basic anti-windup

        # Derivative term
        if self._last_time is not None:
            delta_error = error - self._last_error
            self._derivative = self.Kd * (delta_error / dt)
        else:
            self._derivative = 0  # No derivative on the first run

        # Calculate total output
        output = self._proportional + self._integral + self._derivative

        # Clamp output to limits
        output = self.clamp(output, self.output_limits)

        # Update state for next iteration
        self._last_error = error
        self._last_time = current_time

        return output

    def reset(self):
        """Reset the PID controller's integral sum and timing."""
        self._proportional = 0
        self._integral = 0
        self._derivative = 0
        self._last_error = 0
        self._last_time = None  # Reset time to force dt estimation on next update

    def set_setpoint(self, new_setpoint):
        """Update the target setpoint."""
        self.setpoint = new_setpoint
        self.reset()  # Reset controller state when setpoint changes significantly

    def set_tunings(self, Kp, Ki, Kd):
        """Update the PID gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def clamp(self, value, limits=None):
        """Clamp the value within the specified limits."""
        if limits is None:
            limits = self.output_limits
        lower, upper = limits
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value
