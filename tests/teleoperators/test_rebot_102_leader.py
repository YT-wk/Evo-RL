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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102Leader, BiRebot102LeaderConfig
from lerobot.teleoperators.rebot_102_leader import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
    RebotArm102LeaderTeleopConfig,
)

_MODULE = "lerobot.teleoperators.rebot_102_leader.rebot_102_leader"


def _make_bus_mock(joint_ids: dict[str, int]) -> MagicMock:
    bus = MagicMock(name="FashionStarServoMock")
    bus.ping.return_value = True

    def _sync_monitor(ids):
        # Report each servo at 5 degrees raw.
        monitors = {}
        for servo_id in ids:
            monitor = MagicMock()
            monitor.angle_deg = 5.0
            monitors[servo_id] = monitor
        return monitors

    bus.sync_monitor.side_effect = _sync_monitor
    return bus


@pytest.fixture
def leader():
    cfg = RebotArm102LeaderTeleopConfig(port="/dev/null")
    bus_mock = _make_bus_mock(cfg.joint_ids)
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.FashionStarServo", return_value=bus_mock),
    ):
        teleop = RebotArm102Leader(cfg)
        teleop.connect(calibrate=False)
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_action_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        teleop = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port="/dev/null"))
    assert set(teleop.action_features) == {f"{m}.pos" for m in teleop.motor_names}
    assert teleop.feedback_features == {}


def test_connect_disconnect(leader):
    assert leader.is_connected
    leader.disconnect()
    assert not leader.is_connected


def test_get_action_applies_direction_and_clamp(leader):
    action = leader.get_action()
    assert set(action) == {f"{m}.pos" for m in leader.motor_names}
    # shoulder_pan has direction -1, so a +5deg raw reading flips to -5deg.
    assert action["shoulder_pan.pos"] == pytest.approx(-5.0)
    # Every joint stays within its configured range.
    for motor, value in action.items():
        lo, hi = leader.config.joint_ranges[motor.removesuffix(".pos")]
        assert lo <= value <= hi


def test_send_feedback_not_implemented(leader):
    with pytest.raises(NotImplementedError):
        leader.send_feedback({})


def test_move_to_holds_leader_at_calibrated_follower_target(leader):
    target = {
        "shoulder_pan.pos": -30.0,
        "shoulder_lift.pos": -50.0,
        "elbow_flex.pos": -70.0,
        "wrist_flex.pos": 20.0,
        "wrist_yaw.pos": -30.0,
        "wrist_roll.pos": 40.0,
        "gripper.pos": -120.0,
    }
    leader.bus.reset_mock()

    leader.move_to(target, duration_s=1.25, hold_power=400)

    assert leader.bus.set_stop_mode.call_count == 7
    leader.bus.set_stop_mode.assert_any_call(0, mode=0x11, power=400)
    assert leader.bus.set_angle.call_args_list == [
        ((0, 30.0), {"multi_turn": False, "interval_ms": 1250}),
        ((1, 50.0), {"multi_turn": False, "interval_ms": 1250}),
        ((2, -70.0), {"multi_turn": False, "interval_ms": 1250}),
        ((3, 20.0), {"multi_turn": False, "interval_ms": 1250}),
        ((4, -30.0), {"multi_turn": False, "interval_ms": 1250}),
        ((5, -40.0), {"multi_turn": False, "interval_ms": 1250}),
        ((6, 20.0), {"multi_turn": False, "interval_ms": 1250}),
    ]


def test_move_to_rejects_missing_or_out_of_range_target(leader):
    with pytest.raises(ValueError, match="missing joints"):
        leader.move_to({})

    target = {f"{name}.pos": 0.0 for name in leader.motor_names}
    target["shoulder_pan.pos"] = 151.0
    with pytest.raises(ValueError, match="outside the configured range"):
        leader.move_to(target)


def test_disable_torque_unlocks_every_leader_servo(leader):
    leader.bus.reset_mock()

    leader.disable_torque()

    assert leader.bus.unlock.call_count == 7
    leader.bus.unlock.assert_any_call(0)
    leader.bus.unlock.assert_any_call(6)


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebot102LeaderConfig(
            left_arm_config=RebotArm102LeaderConfig(port="/dev/null0"),
            right_arm_config=RebotArm102LeaderConfig(port="/dev/null1"),
        )
        teleop = BiRebot102Leader(cfg)
    assert any(k.startswith("left_") for k in teleop.action_features)
    assert any(k.startswith("right_") for k in teleop.action_features)
    assert "left_gripper.pos" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features
    assert teleop.left_arm.calibration_fpath != teleop.right_arm.calibration_fpath


def test_bimanual_move_to_splits_prefixed_targets_and_releases_on_error():
    class FakeArm:
        is_connected = True

        def __init__(self, fail=False):
            self.fail = fail
            self.move_calls = []
            self.disable_calls = 0

        def move_to(self, target, *, duration_s, hold_power):
            self.move_calls.append((target, duration_s, hold_power))
            if self.fail:
                raise RuntimeError("serial failure")

        def disable_torque(self):
            self.disable_calls += 1

        def disconnect(self):
            pass

    teleop = object.__new__(BiRebot102Leader)
    teleop.left_arm = FakeArm()
    teleop.right_arm = FakeArm()
    target = {
        **{f"left_{name}.pos": float(index) for index, name in enumerate(teleop.left_arm.__class__.__dict__.get("motor_names", []))},
        "left_shoulder_pan.pos": 1.0,
        "right_shoulder_pan.pos": 2.0,
    }
    # Full joint coverage is validated by each Arm102 instance; this fake only
    # checks prefix routing.
    teleop.move_to(target, duration_s=1.2, hold_power=400)
    assert teleop.left_arm.move_calls == [({"shoulder_pan.pos": 1.0}, 1.2, 400)]
    assert teleop.right_arm.move_calls == [({"shoulder_pan.pos": 2.0}, 1.2, 400)]

    teleop.right_arm.fail = True
    with pytest.raises(RuntimeError, match="handoff alignment failed"):
        teleop.move_to(target)
    assert teleop.left_arm.disable_calls == 1
    assert teleop.right_arm.disable_calls == 1
