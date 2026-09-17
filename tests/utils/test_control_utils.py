from types import SimpleNamespace

import pytest

from lerobot.utils.control_utils import TTYKeyboardListener, sanity_check_bimanual_piper_pair


def test_tty_listener_routes_a_shared_guarded_handoff_key_to_the_state_machine():
    listener = object.__new__(TTYKeyboardListener)
    listener.events = {
        "toggle_intervention": False,
        "prepare_intervention": False,
        "confirm_intervention": False,
    }
    listener.intervention_toggle_key = "i"
    listener.intervention_prepare_key = "i"
    listener.intervention_confirm_key = "i"
    listener._shared_handoff_key = True
    listener.episode_success_key = None
    listener.episode_failure_key = None
    listener._last_intervention_time = -1e9

    listener._handle_key("i")

    assert listener.events["toggle_intervention"]
    assert not listener.events["prepare_intervention"]
    assert not listener.events["confirm_intervention"]


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piper_leader"),
        ("bi_piperx_follower", "bi_piperx_leader"),
        ("so101_follower", "so101_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_accepts_valid_pairs(robot_type, teleop_type):
    sanity_check_bimanual_piper_pair(
        SimpleNamespace(type=robot_type),
        SimpleNamespace(type=teleop_type),
    )


def test_sanity_check_bimanual_piper_pair_accepts_missing_teleop():
    sanity_check_bimanual_piper_pair(SimpleNamespace(type="bi_piperx_follower"), None)


@pytest.mark.parametrize(
    ("robot_type", "teleop_type"),
    [
        ("bi_piper_follower", "bi_piperx_leader"),
        ("bi_piperx_follower", "bi_piper_leader"),
        ("so101_follower", "bi_piperx_leader"),
        ("so101_follower", "bi_piper_leader"),
    ],
)
def test_sanity_check_bimanual_piper_pair_rejects_mixed_pairs(robot_type, teleop_type):
    with pytest.raises(ValueError, match="must be paired"):
        sanity_check_bimanual_piper_pair(
            SimpleNamespace(type=robot_type),
            SimpleNamespace(type=teleop_type),
        )
