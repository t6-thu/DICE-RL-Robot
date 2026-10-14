"""Hardware-free tests for real-time chunking (RTC) deployment pieces."""
import numpy as np
import torch

from dice_rl.env_runner.rtc import RTCScheduler
from dice_rl.replay_buffer.yam_replay_buffer import YAMReplayBuffer, chunk_valid_mask
from diffusion_policy.policy.diffusion_unet_timm_mod1_policy import build_prefix_condition


def _chunk(start: int, H: int = 16) -> np.ndarray:
    """Chunk whose action at frame f is [f] so continuity is checkable."""
    return np.arange(start, start + H, dtype=np.float64)[:, None]


def _simulate(H: int, delay: int, infer_frames, n_frames: int = 120):
    """Run the scheduler with an inference that completes `infer_frames` frames
    after launch (list consumed per launch, last value repeats).

    The fake policy returns, for obs frame k, the chunk [k, k+1, ..., k+H-1],
    so a seamless execution emits exactly the frame index at every frame.
    """
    sched = RTCScheduler(H, delay)
    sched.reset(_chunk(0, H))
    pending = None
    sent, valid = [], []
    last_cmd = np.array([-1.0])
    budget = list(infer_frames)
    for frame in range(n_frames):
        a, v = sched.action_for_frame(frame, last_cmd)
        sent.append(float(a[0])); valid.append(v)
        last_cmd = np.asarray(a)
        if pending is not None and frame >= pending[1]:
            req, _ = pending
            pending = None
            new = _chunk(req.obs_frame, H)
            assert np.array_equal(new[: len(req.prefix)], req.prefix), "prefix not honoured"
            sched.on_result(frame, req, new)
        if pending is None:
            req = sched.maybe_launch(frame, last_cmd)
            if req is not None:
                d = budget.pop(0) if len(budget) > 1 else budget[0]
                pending = (req, frame + d)
    return np.array(sent), np.array(valid), sched.stats


def test_rtc_seamless_when_inference_within_budget():
    for d_infer in (1, 2, 3):
        sent, valid, st = _simulate(16, 3, [d_infer])
        assert valid.all()
        np.testing.assert_array_equal(sent, np.arange(len(sent)))
        assert st.hold_frames == 0 and st.late_results == 0
        assert st.launches >= 6
        assert all(n == 4 for n in st.prefix_lengths), st.prefix_lengths


def test_rtc_zero_delay_inference_switches_early():
    sent, valid, st = _simulate(16, 3, [0])
    assert valid.all()
    np.testing.assert_array_equal(sent, np.arange(len(sent)))
    assert st.early_switches > 0


def test_rtc_overrun_holds_and_flags_frames():
    sent, valid, st = _simulate(16, 3, [5])  # 2 frames over budget each chunk
    assert not valid.all()
    assert st.hold_frames > 0
    # Held frames repeat the previous command and are flagged invalid.
    for i in np.flatnonzero(~valid):
        assert sent[i] == sent[i - 1]
    # Once a chunk is adopted the stream is continuous again.
    assert valid[-1] or valid[-2]


def test_rtc_prefix_is_the_remaining_committed_actions():
    sched = RTCScheduler(16, 3)
    sched.reset(_chunk(0))
    for f in range(12):
        sched.action_for_frame(f, np.zeros(1))
        assert sched.maybe_launch(f, np.zeros(1)) is None or f == 12
    sched.action_for_frame(12, np.zeros(1))
    req = sched.maybe_launch(12, np.array([12.0]))
    assert req is not None and req.obs_frame == 12
    np.testing.assert_array_equal(req.prefix[:, 0], [12, 13, 14, 15])
    # Second call while in flight does nothing.
    assert sched.maybe_launch(13, np.zeros(1)) is None


def test_rtc_hold_prefix_is_the_issued_command():
    sched = RTCScheduler(16, 3)
    sched.reset(_chunk(0))
    for f in range(16):
        sched.action_for_frame(f, np.zeros(1))
    sched.cancel_in_flight()
    a, v = sched.action_for_frame(16, np.array([7.0]))
    assert not v and a[0] == 7.0
    req = sched.maybe_launch(16, np.array([7.0]))
    np.testing.assert_array_equal(req.prefix, [[7.0]])


def test_rtc_late_result_is_discarded():
    sched = RTCScheduler(16, 3)
    sched.reset(_chunk(0))
    for f in range(13):
        sched.action_for_frame(f, np.zeros(1))
    req = sched.maybe_launch(12, np.zeros(1))
    for f in range(13, 40):
        sched.action_for_frame(f, np.zeros(1))
    assert sched.on_result(40, req, _chunk(12)) is False
    assert sched.stats.late_results == 1
    assert sched.in_flight is None


def test_rtc_delay_bounds():
    try:
        RTCScheduler(16, 15)
    except ValueError:
        pass
    else:
        raise AssertionError("delay_steps >= H-1 must be rejected")


def test_prefix_condition_masks_first_n_steps():
    cond = torch.zeros(1, 16, 7)
    mask = torch.zeros(1, 16, 7, dtype=torch.bool)
    prefix = torch.rand(1, 4, 7)
    c2, m2 = build_prefix_condition(cond, mask, prefix)
    assert torch.equal(c2[:, :4], prefix) and m2[:, :4].all()
    assert not m2[:, 4:].any() and (c2[:, 4:] == 0).all()
    assert not mask.any(), "input must not be mutated"
    for bad in (torch.rand(1, 16, 7), torch.rand(1, 0, 7), torch.rand(1, 4, 6)):
        try:
            build_prefix_condition(cond, mask, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid prefix shape {tuple(bad.shape)}")


def test_chunk_valid_mask():
    valid = np.ones(10, dtype=bool)
    valid[5] = False
    m = chunk_valid_mask(valid, 3)
    # windows t..t+2 for t in 0..6; those touching 5 are t=3,4,5
    np.testing.assert_array_equal(m, [1, 1, 1, 0, 0, 0, 1])
    assert chunk_valid_mask(np.ones(3, dtype=bool), 3).shape == (0,)
    assert chunk_valid_mask(np.ones(4, dtype=bool), 3).tolist() == [True]


def _bare_buffer(H: int) -> YAMReplayBuffer:
    rb = YAMReplayBuffer.__new__(YAMReplayBuffer)
    rb.expected_policy_camera_order = None
    rb.action_horizon = H
    rb.obs_horizon = 2
    rb.hire_shaper = None
    rb.robometer_shaper = None
    rb.hire_pbrs_decay_start_episode = 0
    rb._num_online_episodes = 0
    rb._max_online = 10_000
    from collections import deque
    rb._online = deque(maxlen=10_000)
    rb._online_episodes = {}
    rb._online_episode_refcounts = {}
    rb._next_online_episode_id = 0
    rb.use_sparse_for_online_success = False
    return rb


def test_replay_buffer_skips_chunks_over_held_frames():
    H, T = 4, 12
    ep = {
        "images": np.zeros((T, 6, 2, 2), dtype=np.uint8),
        "states": np.zeros((T, 7), dtype=np.float32),
        "actions": np.zeros((T, 7), dtype=np.float32),
        "rewards": np.zeros(T, dtype=np.float32),
        "dones": np.zeros(T, dtype=bool),
        "action_valid": np.ones(T, dtype=bool),
    }
    ep["action_valid"][6] = False
    rb = _bare_buffer(H)
    rb.add_episode(ep)
    ts = sorted(t for _, t in rb._online)
    expected = [t for t in range(T - H) if not (t <= 6 < t + H)]
    assert ts == expected, (ts, expected)

    # Episodes without the field keep every transition (backwards compatible).
    rb2 = _bare_buffer(H)
    ep2 = {k: v for k, v in ep.items() if k != "action_valid"}
    rb2.add_episode(ep2)
    assert len(rb2._online) == T - H
