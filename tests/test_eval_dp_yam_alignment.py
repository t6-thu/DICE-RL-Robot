import numpy as np

from scripts.eval_dp_yam import (
    _camera_ages,
    _pack_policy_images,
    _select_policy_state_dict,
    _split_physical_images,
)


def test_hanoi_wrist_base_maps_wrist_to_rgb0_and_base_to_rgb1():
    base = np.full((3, 2, 2), 10, dtype=np.float32)
    wrist = np.full((3, 2, 2), 20, dtype=np.float32)

    packed = _pack_policy_images(base, wrist, "wrist_base")

    np.testing.assert_array_equal(packed[:3], wrist)
    np.testing.assert_array_equal(packed[3:], base)
    physical_base, physical_wrist = _split_physical_images(packed, "wrist_base")
    np.testing.assert_array_equal(physical_base, base)
    np.testing.assert_array_equal(physical_wrist, wrist)


def test_ema_is_selected_explicitly_without_silent_model_fallback():
    ema = {"weight": "ema"}
    model = {"weight": "model"}
    key, state = _select_policy_state_dict(
        {"state_dicts": {"model": model, "ema_model": ema}},
        "ema",
    )

    assert key == "ema_model"
    assert state is ema

    try:
        _select_policy_state_dict({"state_dicts": {"model": model}}, "ema")
    except KeyError as exc:
        assert "ema_model" in str(exc)
    else:
        raise AssertionError("missing EMA weights must fail instead of using model weights")


def test_camera_age_marks_missing_timestamp_as_stale():
    base_age, wrist_age = _camera_ages(9.8, 0.0, now=10.0)

    assert abs(base_age - 0.2) < 1e-9
    assert wrist_age == float("inf")
