from __future__ import annotations

import numpy as np

from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark


def test_robotwin_action_telemetry_records_action_and_qpos_stats() -> None:
    action = np.arange(14, dtype=np.float64)
    previous = np.zeros(14, dtype=np.float64)
    qpos = np.ones(14, dtype=np.float64)

    telemetry = RoboTwinBenchmark._make_action_telemetry(action, qpos, previous)

    assert telemetry["action"] == action.tolist()
    assert telemetry["qpos"] == qpos.tolist()
    assert telemetry["action_min"] == 0.0
    assert telemetry["action_max"] == 13.0
    assert telemetry["qpos_min"] == 1.0
    assert telemetry["qpos_max"] == 1.0
    assert telemetry["action_delta_from_prev_qpos_max_abs"] == 13.0
    assert telemetry["qpos_step_delta_max_abs"] == 1.0
    assert set(telemetry).issubset(RoboTwinBenchmark._ALL_RECORD_FIELDS)


def test_robotwin_extract_qpos_pads_and_trims_to_14d() -> None:
    short = {"joint_action": {"vector": np.ones(8, dtype=np.float64)}}
    long = {"joint_action": {"vector": np.arange(16, dtype=np.float64)}}

    assert RoboTwinBenchmark._extract_qpos(short).shape == (14,)
    np.testing.assert_array_equal(RoboTwinBenchmark._extract_qpos(short)[8:], np.zeros(6))
    np.testing.assert_array_equal(RoboTwinBenchmark._extract_qpos(long), np.arange(14, dtype=np.float64))


def test_robotwin_object_proxy_detects_largest_red_orange_component() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:65, 10:30] = [180, 70, 40]
    frame[0:12, 70:110] = [190, 80, 30]

    telemetry = RoboTwinBenchmark._make_object_proxy_telemetry(frame)

    assert telemetry["object_proxy_count"] == 2
    assert telemetry["object_proxy_area"] == 900
    assert telemetry["object_proxy_bbox"] == [10, 20, 29, 64]
    assert telemetry["object_proxy_extent_wh"] == [20, 45]
    assert telemetry["object_proxy_top_edge"] is False
    assert len(telemetry["object_proxy_components"]) == 2
    assert telemetry["object_proxy_non_edge_bbox"] == [10, 20, 29, 64]
    assert telemetry["object_proxy_non_edge_aspect"] < 1.0
    assert set(telemetry).issubset(RoboTwinBenchmark._ALL_RECORD_FIELDS)


def test_robotwin_object_proxy_marks_top_edge_horizontal_component() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[0:12, 70:110] = [190, 80, 30]

    telemetry = RoboTwinBenchmark._make_object_proxy_telemetry(frame)

    assert telemetry["object_proxy_bbox"] == [70, 0, 109, 11]
    assert telemetry["object_proxy_extent_wh"] == [40, 12]
    assert telemetry["object_proxy_aspect"] > 3.0
    assert telemetry["object_proxy_top_edge"] is True
    assert telemetry["object_proxy_non_edge_area"] == 0
    assert telemetry["object_proxy_non_edge_bbox"] == []


def test_robotwin_oracle_suffix_gate_runs_once_at_configured_step() -> None:
    benchmark = RoboTwinBenchmark(
        task_name="pick_diverse_bottles",
        oracle_suffix_at_step=128,
    )
    benchmark._env = type("FakeEnv", (), {"take_action_cnt": 127})()

    assert benchmark._should_run_oracle_suffix(done=False) == (False, "")

    benchmark._env.take_action_cnt = 128
    assert benchmark._should_run_oracle_suffix(done=False) == (True, "step>=128")

    benchmark._oracle_suffix_ran = True
    assert benchmark._should_run_oracle_suffix(done=False) == (False, "")
    assert benchmark._should_run_oracle_suffix(done=True) == (False, "")


def test_robotwin_oracle_suffix_object_proxy_guard_uses_area_and_aspect() -> None:
    benchmark = RoboTwinBenchmark(
        task_name="pick_diverse_bottles",
        oracle_suffix_trigger="object_proxy_guard",
        oracle_suffix_min_step=70,
        oracle_suffix_non_edge_area_lte=360,
        oracle_suffix_non_edge_aspect_gte=0.55,
    )
    benchmark._env = type("FakeEnv", (), {"take_action_cnt": 69})()
    telemetry = {
        "object_proxy_non_edge_area": 355,
        "object_proxy_non_edge_aspect": 0.56,
    }

    assert benchmark._should_run_oracle_suffix(done=False, telemetry=telemetry) == (False, "")

    benchmark._env.take_action_cnt = 85
    assert benchmark._should_run_oracle_suffix(done=False, telemetry=telemetry) == (
        True,
        "non_edge_area<=360,non_edge_aspect>=0.550",
    )

    telemetry["object_proxy_non_edge_aspect"] = 0.50
    assert benchmark._should_run_oracle_suffix(done=False, telemetry=telemetry) == (False, "")
