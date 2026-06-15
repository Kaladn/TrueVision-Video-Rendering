from __future__ import annotations

import math

from trueframegen.avatar_motion_address import forward_state, loop_state, nearest_clock, smoothstep


def test_smoothstep_edges_and_midpoint() -> None:
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(0.5) == 0.5


def test_forward_state_samples_clock_path() -> None:
    start = forward_state(0.0)
    middle = forward_state(0.5)
    end = forward_state(1.0)

    assert start.clock == "10"
    assert start.yaw_deg == -40.0
    assert math.isclose(start.pitch_deg, 0.0, abs_tol=1e-6)

    assert middle.clock == "12"
    assert math.isclose(middle.yaw_deg, 0.0, abs_tol=1e-6)
    assert math.isclose(middle.pitch_deg, 12.0, abs_tol=1e-6)

    assert end.clock == "2"
    assert end.yaw_deg == 40.0
    assert math.isclose(end.pitch_deg, 0.0, abs_tol=1e-6)


def test_eyes_lead_head_during_forward_turn() -> None:
    state = forward_state(0.25)
    assert state.eye_yaw_deg > state.yaw_deg


def test_loop_returns_cleanly_to_left_address() -> None:
    duration = 10.0
    assert loop_state(0.0, duration_seconds=duration).clock == "10"
    assert loop_state(5.0, duration_seconds=duration).clock == "2"

    near_end = loop_state(duration - (1.0 / 60.0), duration_seconds=duration)
    assert near_end.clock == "10"
    assert near_end.yaw_deg < -39.0


def test_nearest_clock_addresses() -> None:
    assert nearest_clock(-40.0) == "10"
    assert nearest_clock(-19.0) == "11"
    assert nearest_clock(0.0) == "12"
    assert nearest_clock(19.0) == "1"
    assert nearest_clock(40.0) == "2"
