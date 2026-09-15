#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import TYPE_CHECKING

from lerobot.processor import RobotAction
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_smart_servo_available, require_package

from ..teleoperator import Teleoperator
from .config_rebot_102_leader import RebotArm102LeaderTeleopConfig

if TYPE_CHECKING or _motorbridge_smart_servo_available:
    from motorbridge_smart_servo import FashionStarServo, ServoMonitor
else:
    FashionStarServo = None
    ServoMonitor = None

logger = logging.getLogger(__name__)

_SETTLE_SEC = 0.01
_STOP_MODE_UNLOADED = 0x10
_STOP_MODE_HOLD = 0x11
_DEFAULT_HOLD_POWER = 500


class RebotArm102Leader(Teleoperator):
    """Seeed Studio StarArm102 / reBot Arm 102 leader arm.

    A 7-joint (incl. gripper) leader built on FashionStar UART smart servos. Servo
    communication is handled by the ``motorbridge-smart-servo`` package; this class
    reads joint angles and can explicitly position/hold the leader for a safe
    policy-to-teleoperation handoff.
    """

    config_class = RebotArm102LeaderTeleopConfig
    name = "rebot_102_leader"

    def __init__(self, config: RebotArm102LeaderTeleopConfig):
        require_package("motorbridge-smart-servo", extra="rebot", import_name="motorbridge_smart_servo")
        super().__init__(config)
        self.config = config
        self.bus: FashionStarServo | None = None
        self.motor_names = list(config.joint_ids.keys())
        self._last_raw_positions: dict[str, float] = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port}...")
        bus = FashionStarServo(self.config.port, baudrate=self.config.baudrate)
        try:
            for motor_name, motor_id in self.config.joint_ids.items():
                if not bus.ping(motor_id):
                    raise RuntimeError(f"Servo not found for {motor_name} (id={motor_id}).")
                self._last_raw_positions[motor_name] = 0.0
            self.bus = bus

            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()
        except Exception:
            bus.close()
            self.bus = None
            raise

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and set(self.calibration) == set(self.motor_names)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        input(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot Arm 102 to its zero pose and close the gripper.\n"
            "Press ENTER when ready..."
        )

        self.calibration = {}
        for motor_name, motor_id in self.config.joint_ids.items():
            self.bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
            self.bus.set_origin_point(motor_id)
            range_min, range_max = self.config.joint_ranges[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=motor_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        for motor_id in self.config.joint_ids.values():
            self.bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)
        # Reset the multi-turn counter of each servo individually.
        for motor_id in self.config.joint_ids.values():
            self.bus.reset_multi_turn(motor_id)

    def _read_raw_positions(self) -> dict[str, float]:
        result: dict[int, ServoMonitor | None] = self.bus.sync_monitor(list(self.config.joint_ids.values()))
        id_to_name = {v: k for k, v in self.config.joint_ids.items()}
        raw_positions: dict[str, float] = {}
        for motor_id, monitor in result.items():
            motor_name = id_to_name[motor_id]
            if monitor is None:
                raise RuntimeError(f"Servo {motor_name} (id={motor_id}) has never responded.")
            raw_positions[motor_name] = monitor.angle_deg
        return raw_positions

    @staticmethod
    def _round_to_valid_range(value: float, min_value: float, max_value: float) -> tuple[float, int]:
        """Unwrap a multi-turn angle into the ±180° window centred on (min+max)/2.

        The servo may report an angle that has accumulated extra full rotations
        (value = true_angle + N*360). Subtract the nearest whole number of turns
        to bring it back into [center-180, center+180]. Returns the unwrapped
        angle and the number of turns removed.
        """
        center = (min_value + max_value) / 2.0
        turns = round((value - center) / 360.0)
        return value - turns * 360.0, abs(turns)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        try:
            raw_positions = self._read_raw_positions()
            self._last_raw_positions = raw_positions
        except Exception as e:
            logger.error(f"Failed to read raw positions: {e}")
            logger.warning("[EMERGENCY STOP] Hold the follower arm and cut off the main power to the arms.")
            logger.warning(
                "[EMERGENCY STOP] Break the teleoperation session and check the leader USB connection or power."
            )
            raw_positions = self._last_raw_positions

        action_dict: dict[str, float] = {}
        for motor_name in self.motor_names:
            range_min, range_max = self.config.joint_ranges[motor_name]
            direction = self.config.joint_directions[motor_name]
            sign = 1.0 if direction >= 0 else -1.0
            unwrapped, k = self._round_to_valid_range(
                raw_positions[motor_name], range_min * sign, range_max * sign
            )
            position = unwrapped * direction
            if k > 0:
                logger.debug(
                    f"Servo {motor_name} (id={self.config.joint_ids[motor_name]}) wrapped {k} * 360°. "
                    f"Unwrapped pos: {unwrapped:.1f}° (raw: {raw_positions[motor_name]:.1f}°)"
                )
            action_dict[f"{motor_name}.pos"] = max(float(range_min), min(float(range_max), position))

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action_dict

    def _raw_target_from_action(self, motor_name: str, action_value: float) -> float:
        """Map a calibrated follower-space action back to an Arm102 servo angle."""
        range_min, range_max = self.config.joint_ranges[motor_name]
        if not range_min <= action_value <= range_max:
            raise ValueError(
                f"Target for {motor_name} ({action_value:.2f} degrees) is outside "
                f"the configured range [{range_min}, {range_max}]."
            )
        return action_value / self.config.joint_directions[motor_name]

    @check_if_not_connected
    def enable_torque(self, power: int = _DEFAULT_HOLD_POWER) -> None:
        """Enable bounded position hold on all leader servos."""
        if power < 0:
            raise ValueError("Leader hold power must be non-negative.")
        for motor_id in self.config.joint_ids.values():
            self.bus.set_stop_mode(motor_id, mode=_STOP_MODE_HOLD, power=power)
            time.sleep(_SETTLE_SEC)

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Release all leader servos so the operator can move the Arm102 manually."""
        for motor_id in self.config.joint_ids.values():
            self.bus.unlock(motor_id)
            time.sleep(_SETTLE_SEC)

    @check_if_not_connected
    def move_to(
        self,
        target: RobotAction,
        *,
        duration_s: float = 1.0,
        hold_power: int = _DEFAULT_HOLD_POWER,
    ) -> None:
        """Smoothly move the leader to a follower-space target and keep it there.

        ``FashionStarServo.set_angle(..., interval_ms=...)`` performs the timed
        interpolation in the servo firmware. The same calibrated action convention
        as :meth:`get_action` is used, including the Arm102 gripper scale.
        """
        if duration_s <= 0:
            raise ValueError("Leader handoff duration must be positive.")
        missing = [f"{name}.pos" for name in self.motor_names if f"{name}.pos" not in target]
        if missing:
            raise ValueError(f"Leader handoff target is missing joints: {missing}.")

        interval_ms = max(1, round(duration_s * 1_000))
        self.enable_torque(hold_power)
        for motor_name, motor_id in self.config.joint_ids.items():
            action_value = float(target[f"{motor_name}.pos"])
            raw_target = self._raw_target_from_action(motor_name, action_value)
            self.bus.set_angle(motor_id, raw_target, multi_turn=False, interval_ms=interval_ms)
            time.sleep(_SETTLE_SEC)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError(
            "Continuous policy feedback is not enabled for the reBot Arm 102 leader. "
            "Use move_to() for the guarded teleoperation handoff."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.close()
        self.bus = None
        logger.info(f"{self} disconnected.")
