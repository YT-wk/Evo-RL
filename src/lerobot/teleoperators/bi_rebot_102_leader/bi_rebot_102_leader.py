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
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_rebot_102_leader import BiRebot102LeaderConfig

logger = logging.getLogger(__name__)


class BiRebot102Leader(Teleoperator):
    """Bimanual Seeed Studio StarArm102 / reBot Arm 102 leader.

    Composes two single-arm :class:`RebotArm102Leader` instances. Action keys of
    each arm are namespaced with a ``left_`` / ``right_`` prefix, so a bimanual
    leader can teleoperate a bimanual reBot B601 follower.
    """

    config_class = BiRebot102LeaderConfig
    name = "bi_rebot_102_leader"

    def __init__(self, config: BiRebot102LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = RebotArm102LeaderTeleopConfig(
            id=f"{config.id or self.name}_left",
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            baudrate=config.left_arm_config.baudrate,
            joint_ids=config.left_arm_config.joint_ids,
            joint_directions=config.left_arm_config.joint_directions,
            joint_ranges=config.left_arm_config.joint_ranges,
        )

        right_arm_config = RebotArm102LeaderTeleopConfig(
            id=f"{config.id or self.name}_right",
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            baudrate=config.right_arm_config.baudrate,
            joint_ids=config.right_arm_config.joint_ids,
            joint_directions=config.right_arm_config.joint_directions,
            joint_ranges=config.right_arm_config.joint_ranges,
        )

        self.left_arm = RebotArm102Leader(left_arm_config)
        self.right_arm = RebotArm102Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect both leaders and release either side if the other fails."""
        try:
            self._run_both(
                "connection",
                lambda: self.left_arm.connect(calibrate),
                lambda: self.right_arm.connect(calibrate),
            )
        except Exception:
            logger.exception("Bimanual Arm102 connection failed; releasing initialized leader arms.")
            self._release_connected_arms()
            raise

    def calibrate(self) -> None:
        # Each leader calibration is interactive, so do not interleave prompts.
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self._run_both("configuration", self.left_arm.configure, self.right_arm.configure)

    def _release_connected_arms(self) -> None:
        for arm in (self.right_arm, self.left_arm):
            if not arm.is_connected:
                continue
            try:
                arm.disable_torque()
            except Exception:
                logger.exception("Failed to release Arm102 leader during cleanup.")
            try:
                arm.disconnect()
            except Exception:
                logger.exception("Failed to disconnect Arm102 leader during cleanup.")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}
        action_dict.update({f"left_{k}": v for k, v in self.left_arm.get_action().items()})
        action_dict.update({f"right_{k}": v for k, v in self.right_arm.get_action().items()})
        return action_dict

    def _run_both(self, operation: str, left_fn, right_fn) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(left_fn)
            right_future = executor.submit(right_fn)
            errors = []
            for side, future in (("left", left_future), ("right", right_future)):
                try:
                    future.result()
                except Exception as error:
                    errors.append((side, error))
        if errors:
            details = "; ".join(f"{side}: {error}" for side, error in errors)
            raise RuntimeError(f"Bimanual Arm102 {operation} failed ({details})") from errors[0][1]

    @check_if_not_connected
    def enable_torque(self, power: int = 500) -> None:
        try:
            self._run_both(
                "torque enable",
                lambda: self.left_arm.enable_torque(power),
                lambda: self.right_arm.enable_torque(power),
            )
        except Exception:
            self.disable_torque()
            raise

    @check_if_not_connected
    def disable_torque(self) -> None:
        self._run_both("torque disable", self.left_arm.disable_torque, self.right_arm.disable_torque)

    @check_if_not_connected
    def move_to(
        self,
        target: RobotAction,
        *,
        duration_s: float = 1.0,
        hold_power: int = 500,
    ) -> None:
        """Align both Arm102 leaders concurrently to a prefixed 14D target."""
        left_target = {
            key.removeprefix("left_"): value for key, value in target.items() if key.startswith("left_")
        }
        right_target = {
            key.removeprefix("right_"): value for key, value in target.items() if key.startswith("right_")
        }
        try:
            self._run_both(
                "handoff alignment",
                lambda: self.left_arm.move_to(
                    left_target, duration_s=duration_s, hold_power=hold_power
                ),
                lambda: self.right_arm.move_to(
                    right_target, duration_s=duration_s, hold_power=hold_power
                ),
            )
        except Exception:
            self.disable_torque()
            raise

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the reBot Arm 102 leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.disable_torque()
        finally:
            self._release_connected_arms()
