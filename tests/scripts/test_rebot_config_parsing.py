import draccus

from lerobot.scripts.lerobot_calibrate import CalibrateConfig
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.scripts.lerobot_replay import ReplayConfig
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig


def _single_robot_args() -> list[str]:
    return [
        "--robot.type=rebot_b601_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=b601",
    ]


def _single_teleop_args() -> list[str]:
    return [
        "--teleop.type=rebot_102_leader",
        "--teleop.port=/dev/ttyUSB0",
        "--teleop.id=arm102",
    ]


def _bimanual_robot_args() -> list[str]:
    return [
        "--robot.type=bi_rebot_b601_follower",
        "--robot.left_arm_config.port=/dev/ttyACM0",
        "--robot.right_arm_config.port=/dev/ttyACM1",
        "--robot.id=bi_b601",
    ]


def _bimanual_teleop_args() -> list[str]:
    return [
        "--teleop.type=bi_rebot_102_leader",
        "--teleop.left_arm_config.port=/dev/ttyUSB0",
        "--teleop.right_arm_config.port=/dev/ttyUSB1",
        "--teleop.id=bi_arm102",
    ]


def test_calibrate_parses_single_rebot_robot_and_leader():
    robot_cfg = draccus.parse(CalibrateConfig, config_path=None, args=_single_robot_args())
    teleop_cfg = draccus.parse(CalibrateConfig, config_path=None, args=_single_teleop_args())

    assert robot_cfg.robot.type == "rebot_b601_follower"
    assert teleop_cfg.teleop.type == "rebot_102_leader"


def test_teleoperate_parses_bimanual_rebot_pair():
    cfg = draccus.parse(
        TeleoperateConfig,
        config_path=None,
        args=[*_bimanual_robot_args(), *_bimanual_teleop_args()],
    )

    assert cfg.robot.type == "bi_rebot_b601_follower"
    assert cfg.teleop.type == "bi_rebot_102_leader"
    assert cfg.robot.left_arm_config.port == "/dev/ttyACM0"
    assert cfg.robot.right_arm_config.port == "/dev/ttyACM1"
    assert cfg.teleop.left_arm_config.port == "/dev/ttyUSB0"
    assert cfg.teleop.right_arm_config.port == "/dev/ttyUSB1"


def test_record_parses_single_rebot_pair():
    cfg = draccus.parse(
        RecordConfig,
        config_path=None,
        args=[
            *_single_robot_args(),
            *_single_teleop_args(),
            "--dataset.repo_id=dummy/rebot",
            "--dataset.single_task=test",
            "--dataset.num_episodes=1",
            "--dataset.episode_time_s=1",
            "--dataset.reset_time_s=1",
            "--dataset.push_to_hub=false",
        ],
    )

    assert cfg.robot.type == "rebot_b601_follower"
    assert cfg.teleop.type == "rebot_102_leader"


def test_replay_parses_bimanual_rebot_robot():
    cfg = draccus.parse(
        ReplayConfig,
        config_path=None,
        args=[
            *_bimanual_robot_args(),
            "--dataset.repo_id=dummy/rebot",
            "--dataset.episode=0",
        ],
    )

    assert cfg.robot.type == "bi_rebot_b601_follower"
