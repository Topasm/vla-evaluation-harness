"""LIBERO benchmark implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast
import uuid

import math

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.libero.utils import preprocess_libero_image
from vla_eval.rotation import matrix_to_quat, quat_to_axisangle
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

# EGL for headless rendering
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _quat_to_axisangle_robosuite(quat: np.ndarray) -> np.ndarray:
    """Robosuite-style quat [x,y,z,w] → axis-angle. No antipodal normalization."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

MAX_STEP_MAPPING = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
    "libero_90": 400,
}


class LIBEROBenchmark(StepBenchmark):
    """LIBERO tabletop manipulation benchmark (MuJoCo/robosuite).

    Non-obvious behaviors:
        - **PyTorch compat**: Patches ``torch.load`` to use
          ``weights_only=False`` for PyTorch ≥2.6 compatibility with LIBERO's
          initial-state files (numpy arrays stored via ``torch.save``).
        - **Headless rendering**: Sets ``EGL_PLATFORM=device`` and
          ``PYOPENGL_PLATFORM=egl`` on import for GPU-accelerated headless
          rendering.
        - **Dummy wait steps**: At episode start, ``num_steps_wait`` steps
          (default 10) are executed with a fixed open-gripper action to let
          objects settle in the physics simulation.
        - **Suite-specific max_steps**: libero_spatial=220, libero_object=280,
          libero_goal=300, libero_10=520, libero_90=400.
        - **Image preprocessing**: robosuite renders images with inverted axes.
          Both agentview and wrist images are flipped ``[::-1, ::-1]`` to
          correct orientation, then resized to 256×256 with padding.

    Args:
        suite: LIBERO suite name (e.g. "libero_spatial", "libero_10").
        seed: Random seed for environment initialization.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive 8-D state
            ``[pos3, axisangle3, gripper2]`` in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
            When True, sets ``robot.controller.use_delta = False`` after the
            initial dummy-wait steps.
        max_steps: Override the default suite-specific max step count.
            When None, uses ``MAX_STEP_MAPPING[suite]``.
        env_seed: Seed for ``env.seed()``.  When None, defaults to ``seed``.
            OpenVLA reference uses ``env_seed=0`` separately from ``seed=7``.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        suite: str = "libero_spatial",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        quat_no_antipodal: bool = False,
        send_physics_state_hash: bool = False,
        f8x_counterfactual_enabled: bool = False,
        f8x_max_horizon: int = 6,
        f8x_min_horizon: int = 6,
        f8x_cadence: int = 1,
        f8x_log_path: str = "",
    ) -> None:
        super().__init__()
        self.suite = suite
        self.seed = seed
        self._quat_to_aa = _quat_to_axisangle_robosuite if quat_no_antipodal else quat_to_axisangle
        self.env_seed = env_seed if env_seed is not None else seed
        self.num_steps_wait = num_steps_wait
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self.send_physics_state_hash = send_physics_state_hash
        self.f8x_counterfactual_enabled = bool(f8x_counterfactual_enabled)
        self.f8x_max_horizon = max(int(f8x_max_horizon), 1)
        self.f8x_min_horizon = max(int(f8x_min_horizon), 1)
        self.f8x_cadence = max(int(f8x_cadence), 1)
        if self.f8x_min_horizon > self.f8x_max_horizon:
            raise ValueError("f8x_min_horizon must be <= f8x_max_horizon")
        default_f8x_path = Path("/workspace/results") / f"f8x_counterfactual_{uuid.uuid4().hex}.jsonl"
        self.f8x_log_path = Path(f8x_log_path) if f8x_log_path else default_f8x_path
        self._max_steps = max_steps
        self._env = None
        self._task_suite = None
        self._current_task_id: int | None = None

    @staticmethod
    def _processed_action(raw_action: Any) -> list[float]:
        if isinstance(raw_action, np.ndarray):
            raw_action = raw_action.tolist()
        if len(raw_action) != 7:
            raise ValueError(f"Action dimension mismatch: got {len(raw_action)}, expected 7")
        gripper = -1.0 if raw_action[-1] < 0 else 1.0
        return [float(value) for value in raw_action[:-1]] + [gripper]

    @staticmethod
    def _copy_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, (bool, int, float, str, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            copied_items = [LIBEROBenchmark._copy_value(item) for item in value]
            if all(item is not None for item in copied_items):
                return type(value)(copied_items)
            return None
        if isinstance(value, dict):
            copied_dict = {}
            for key, item in value.items():
                if not isinstance(key, (bool, int, float, str)):
                    return None
                copied_item = LIBEROBenchmark._copy_value(item)
                if copied_item is None:
                    return None
                copied_dict[key] = copied_item
            return copied_dict
        return None

    @staticmethod
    def _restore_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, list):
            return [LIBEROBenchmark._restore_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(LIBEROBenchmark._restore_value(item) for item in value)
        if isinstance(value, dict):
            return {key: LIBEROBenchmark._restore_value(item) for key, item in value.items()}
        return value

    def _capture_f8x_sim_data(self) -> dict[str, np.ndarray]:
        assert self._env is not None
        data = self._env.sim.data
        state: dict[str, np.ndarray] = {}
        for name in (
            "ctrl",
            "qacc_warmstart",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "userdata",
        ):
            value = getattr(data, name, None)
            if isinstance(value, np.ndarray):
                state[name] = value.copy()
        return state

    def _restore_f8x_sim_data(self, state: dict[str, np.ndarray]) -> None:
        assert self._env is not None
        data = self._env.sim.data
        for name, value in state.items():
            target = getattr(data, name, None)
            if isinstance(target, np.ndarray) and target.shape == value.shape:
                target[...] = value

    def _capture_f8x_snapshot(self) -> dict[str, Any]:
        assert self._env is not None
        env = self._env.env
        robot = env.robots[0]
        controller = robot.controller
        controller_state = {
            name: copied
            for name, value in controller.__dict__.items()
            if (copied := self._copy_value(value)) is not None and name != "sim"
        }
        robot_state: dict[str, Any] = {}
        for name in (
            "recent_qpos",
            "recent_actions",
            "recent_torques",
            "recent_ee_acc",
            "recent_ee_forcetorques",
            "recent_ee_pose",
            "recent_ee_vel",
            "recent_ee_vel_buffer",
        ):
            buffer = getattr(robot, name, None)
            if buffer is not None:
                robot_state[name] = {
                    key: copied
                    for key, value in buffer.__dict__.items()
                    if (copied := self._copy_value(value)) is not None
                }
        robot_state["torques"] = np.asarray(robot.torques).copy()
        gripper_state = {
            name: copied
            for name, value in robot.gripper.__dict__.items()
            if (copied := self._copy_value(value)) is not None
        }
        observable_state = {
            name: {
                key: copied
                for key, value in observable.__dict__.items()
                if (copied := self._copy_value(value)) is not None
            }
            for name, observable in env._observables.items()
        }
        return {
            "sim_state": self._env.sim.get_state().flatten().copy(),
            "sim_data": self._capture_f8x_sim_data(),
            "timestep": int(env.timestep),
            "cur_time": float(env.cur_time),
            "done": bool(env.done),
            "controller": controller_state,
            "robot": robot_state,
            "gripper": gripper_state,
            "observables": observable_state,
        }

    def _restore_f8x_snapshot(self, snapshot: dict[str, Any]) -> None:
        assert self._env is not None
        env = self._env.env
        robot = env.robots[0]
        self._env.set_state(snapshot["sim_state"])
        self._env.sim.forward()
        self._restore_f8x_sim_data(snapshot.get("sim_data", {}))
        env.timestep = snapshot["timestep"]
        env.cur_time = snapshot["cur_time"]
        env.done = snapshot["done"]
        for name, value in snapshot["controller"].items():
            setattr(robot.controller, name, self._restore_value(value))
        for name, state in snapshot["robot"].items():
            if name == "torques":
                robot.torques = state.copy()
                continue
            buffer = getattr(robot, name)
            for key, value in state.items():
                setattr(buffer, key, self._restore_value(value))
        for name, value in snapshot.get("gripper", {}).items():
            setattr(robot.gripper, name, self._restore_value(value))
        for name, state in snapshot["observables"].items():
            observable = env._observables[name]
            for key, value in state.items():
                setattr(observable, key, self._restore_value(value))

    @staticmethod
    def _state_hash(state: np.ndarray) -> str:
        array = np.ascontiguousarray(np.asarray(state))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _goal_fraction(self) -> float:
        assert self._env is not None
        env = self._env.env
        goals = list(env.parsed_problem.get("goal_state", []))
        if not goals:
            return float(bool(env._check_success()))
        return float(np.mean([bool(env._eval_predicate(goal)) for goal in goals]))

    def _run_f8x_plan(self, plan: list[list[float]]) -> dict[str, Any]:
        assert self._env is not None
        cumulative_reward = 0.0
        success = False
        success_step: int | None = None
        first_state: np.ndarray | None = None
        executed = 0
        for index, action in enumerate(plan, start=1):
            _obs, reward, done, _info = self._env.step(self._processed_action(action))
            executed = index
            cumulative_reward += float(reward)
            state = self._env.sim.get_state().flatten().copy()
            if first_state is None:
                first_state = state
            success = bool(done or self._env.check_success())
            if success:
                success_step = index
                break
        final_state = self._env.sim.get_state().flatten().copy()
        goal_fraction = self._goal_fraction()
        speed_bonus = 0.0 if success_step is None else 0.01 * (len(plan) - success_step) / max(len(plan), 1)
        return {
            "success": success,
            "success_step": success_step,
            "executed_steps": executed,
            "cumulative_reward": cumulative_reward,
            "goal_fraction": goal_fraction,
            "task_score": goal_fraction + speed_bonus,
            "first_state": first_state,
            "final_state": final_state,
            "final_state_hash": self._state_hash(final_state),
        }

    def _prepare_f8x_record(self, payload: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray] | None:
        action_step = int(payload.get("action_step", -1))
        if action_step < 0 or action_step % self.f8x_cadence != 0:
            return None
        if int(payload.get("action_horizon", -1)) != 1 or int(payload.get("query_step", -2)) + 1 != action_step:
            raise ValueError("F8X candidate/action horizon alignment failed")
        base_plan = [list(row) for row in payload.get("base_plan", [])[: self.f8x_max_horizon]]
        if len(base_plan) < self.f8x_min_horizon:
            return None
        snapshot = self._capture_f8x_snapshot()
        outcomes: dict[str, dict[str, Any]] = {}

        self._restore_f8x_snapshot(snapshot)
        outcomes["base"] = self._run_f8x_plan(base_plan)
        for name, first_action in payload.get("branches", {}).items():
            self._restore_f8x_snapshot(snapshot)
            outcomes[str(name)] = self._run_f8x_plan([list(first_action)] + base_plan[1:])
        self._restore_f8x_snapshot(snapshot)
        outcomes["base_repeat"] = self._run_f8x_plan(base_plan)
        self._restore_f8x_snapshot(snapshot)

        base_final = outcomes["base"]["final_state"]
        repeat_final = outcomes["base_repeat"]["final_state"]
        repeat_max_abs = float(np.max(np.abs(base_final - repeat_final)))
        base_score = float(outcomes["base"]["task_score"])
        labels: dict[str, dict[str, Any]] = {}
        for name, outcome in outcomes.items():
            if name in {"base", "base_repeat"}:
                continue
            delta = float(outcome["task_score"] - base_score)
            labels[name] = {
                "task_score_delta": delta,
                "counterfactual_benefit": bool(delta > 1e-12),
                "counterfactual_harm": bool(delta < -1e-12),
            }
        serializable_outcomes = {
            name: {key: value for key, value in outcome.items() if key not in {"first_state", "final_state"}}
            for name, outcome in outcomes.items()
        }
        record = {
            "kind": "f8x_counterfactual_fork",
            "suite": self.suite,
            "task_id": self._current_task_id,
            "task_description": self._task.get("name", ""),
            "episode_id": str(payload.get("episode_id", "")),
            "action_step": action_step,
            "query_step": int(payload["query_step"]),
            "horizon": len(base_plan),
            "support": payload.get("support", {}),
            "support_thresholds": payload.get("support_thresholds", {}),
            "candidate_base_l2": float(payload.get("candidate_base_l2", float("nan"))),
            "gripper_sign_flip": bool(payload.get("gripper_sign_flip", False)),
            "outcomes": serializable_outcomes,
            "labels": labels,
            "restore_parity": {"base_repeat_final_max_abs": repeat_max_abs},
        }
        return record, np.asarray(outcomes["base"]["first_state"])

    def _write_f8x_record(self, record: dict[str, Any]) -> None:
        self.f8x_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.f8x_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _init_libero(self) -> None:
        """Lazily initialize LIBERO (heavy imports)."""
        if self._task_suite is not None:
            return
        # LIBERO init states use torch.save with numpy arrays.
        # PyTorch ≥2.6 defaults weights_only=True which blocks numpy globals.
        # Patch torch.load to default weights_only=False for LIBERO compatibility.
        import functools

        import torch

        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = cast(Any, _patched_load)

        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        self._task_suite = benchmark_dict[self.suite]()

    def get_tasks(self) -> list[Task]:
        self._init_libero()
        assert self._task_suite is not None
        tasks = []
        for task_id in range(self._task_suite.n_tasks):
            task = self._task_suite.get_task(task_id)
            tasks.append(
                {
                    "name": task.language,
                    "suite": self.suite,
                    "task_id": task_id,
                    "task_obj": task,
                }
            )
        return tasks

    def reset(self, task: Task) -> Any:
        from pathlib import Path

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_obj = task["task_obj"]
        task_id = task["task_id"]
        episode_idx = task.get("episode_idx", 0)

        # Only create a new env when the task changes (reuse across episodes)
        if self._env is None or self._current_task_id != task_id:
            if self._env is not None:
                self._env.close()

            bddl_file = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
            env_args = {
                "bddl_file_name": str(bddl_file),
                "camera_heights": LIBERO_ENV_RESOLUTION,
                "camera_widths": LIBERO_ENV_RESOLUTION,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(self.env_seed)
            self._env = env
            self._current_task_id = task_id

        # Reset env before setting init state (matches reference)
        self._env.reset()

        # Set initial state
        assert self._task_suite is not None
        initial_states = self._task_suite.get_task_init_states(task_id)
        obs = self._env.set_init_state(initial_states[episode_idx])

        # Run dummy action wait steps (always in delta mode to avoid slamming to origin)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)

        # Switch to absolute action mode after settling (e.g. for X-VLA)
        if self.absolute_action:
            for robot in self._env.robots:
                robot.controller.use_delta = False

        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        processed_action = self._processed_action(raw_action)

        assert self._env is not None
        prepared = None
        if self.f8x_counterfactual_enabled and "f8x_counterfactual" in action:
            prepared = self._prepare_f8x_record(action["f8x_counterfactual"])
        obs, reward, done, info = self._env.step(processed_action)
        if prepared is not None:
            record, expected_first_state = prepared
            actual_state = self._env.sim.get_state().flatten().copy()
            record["restore_parity"]["actual_base_first_max_abs"] = float(
                np.max(np.abs(actual_state - expected_first_state))
            )
            record["restore_parity"]["pass_atol_1e_10"] = bool(
                record["restore_parity"]["base_repeat_final_max_abs"] <= 1e-10
                and record["restore_parity"]["actual_base_first_max_abs"] <= 1e-10
            )
            self._write_f8x_record(record)
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(reward), done=bool(done), success=bool(done))
        return StepResult(obs=obs, reward=reward, done=done, info=info)

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        frame = raw_obs.get("agentview_image")
        if frame is None:
            return None
        # Robosuite renders agentview/wrist inverted; flip to upright.
        return np.ascontiguousarray(frame[::-1, ::-1])

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        img = preprocess_libero_image(raw_obs["agentview_image"], LIBERO_ENV_RESOLUTION)

        obs_dict: dict[str, Any] = {
            "images": {"agentview": img},
            "task_description": task["name"],
        }

        if self.send_wrist_image:
            wrist = preprocess_libero_image(raw_obs["robot0_eye_in_hand_image"], LIBERO_ENV_RESOLUTION)
            obs_dict["images"]["wrist"] = wrist

        if self.send_state:
            # Both sources: observation (default) and controller.
            # Most models (Pi0, OFT, GR00T) use obs; X-VLA uses controller.
            obs_dict["states"] = np.concatenate(
                [
                    raw_obs["robot0_eef_pos"],
                    self._quat_to_aa(raw_obs["robot0_eef_quat"]),
                    raw_obs["robot0_gripper_qpos"],
                ]
            )
            assert self._env is not None
            robot = self._env.robots[0]
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_ori_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
            ee_aa = quat_to_axisangle(matrix_to_quat(ee_ori_mat))
            obs_dict["controller_states"] = np.concatenate(
                [ee_pos, ee_aa, np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)]
            )

        if self.send_physics_state_hash:
            assert self._env is not None
            state = np.ascontiguousarray(np.asarray(self._env.sim.get_state().flatten()))
            digest = hashlib.sha256()
            digest.update(str(state.dtype).encode("ascii"))
            digest.update(str(tuple(state.shape)).encode("ascii"))
            digest.update(state.tobytes())
            obs_dict["physics_state_hash"] = digest.hexdigest()

        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_steps or MAX_STEP_MAPPING.get(self.suite, 300),
            "max_episodes_per_task": 50,  # bounded by initial_states per task
            "suite": self.suite,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "agentview": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_AA_GRIP
        return spec

    def render(self) -> np.ndarray | None:
        try:
            assert self._env is not None
            return self._env.render()
        except Exception:
            return None
