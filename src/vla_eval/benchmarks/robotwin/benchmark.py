"""RoboTwin 2.0 benchmark — dual-arm manipulation on SAPIEN/CuRobo.

Ported from the existing ``vla_evaluation_harness`` implementation
shipped in the ``robotwin`` Docker image.

Non-obvious behaviors:
    - **Expert check**: ``get_tasks()`` optionally runs the oracle planner
      per seed to verify solvability (``skip_expert_check=False``).
    - **Lazy init**: Heavy imports happen on first use, not at construction.
    - **14D action**: dual-arm qpos; 16D inputs are trimmed to 14D.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from contextlib import contextmanager
from typing import Any, cast

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

ROBOTWIN_ROOT = os.environ.get("ROBOTWIN_ROOT", "/app/RoboTwin")


class _EvalGripperPlanner:
    """Minimal planner shim for eval-only RoboTwin startup.

    RoboTwin's qpos evaluation path still calls ``plan_grippers()`` during
    env setup, but it never uses CuRobo path planning afterwards.  This shim
    keeps gripper interpolation working while avoiding the expensive CuRobo
    warmup in ``Robot.set_planner()``.
    """

    def plan_grippers(self, now_val: float, target_val: float) -> dict[str, Any]:
        num_step = 200
        per_step = (target_val - now_val) / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": per_step, "result": vals}

    def update_point_cloud(self, world_pcd: Any, resolution: float = 0.02) -> None:
        return None

    def plan_path(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo path planning during episode execution.")

    def plan_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("RoboTwin eval fast-path disables CuRobo batch planning during episode execution.")


class _LazyOpen3D(types.ModuleType):
    """Import open3d only when one of its attributes is first accessed."""

    def __init__(self) -> None:
        super().__init__("open3d")
        self._real_module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._real_module is not None:
            return self._real_module

        if sys.modules.get("open3d") is self:
            sys.modules.pop("open3d", None)
        try:
            module = importlib.import_module("open3d")
        except Exception:
            sys.modules["open3d"] = self
            raise
        self.__dict__.update(module.__dict__)
        self._real_module = module
        sys.modules["open3d"] = module
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


@contextmanager
def _defer_open3d_import(enabled: bool):
    """Defer open3d import during RoboTwin module import when pointclouds are unused."""
    if not enabled:
        yield
        return

    previous = sys.modules.get("open3d")
    proxy = _LazyOpen3D()
    sys.modules["open3d"] = proxy
    try:
        yield
    finally:
        if sys.modules.get("open3d") is proxy:
            if previous is None:
                sys.modules.pop("open3d", None)
            else:
                sys.modules["open3d"] = previous


def _make_fast_set_planner(robot_mod: Any):
    def _set_planner_fast(self: Any, scene: Any = None) -> None:
        self.communication_flag = False
        self.left_planner = _EvalGripperPlanner()
        self.right_planner = _EvalGripperPlanner()

        if self.need_topp:
            self.left_mplib_planner = robot_mod.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = robot_mod.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    return _set_planner_fast


@contextmanager
def _patched_robot_set_planner(enabled: bool):
    """Temporarily skip CuRobo planner warmup during env setup."""
    if not enabled:
        yield
        return

    import envs.robot.robot as robot_mod

    original = robot_mod.Robot.set_planner
    robot_mod.Robot.set_planner = _make_fast_set_planner(robot_mod)
    try:
        yield
    finally:
        robot_mod.Robot.set_planner = original


@contextmanager
def _patched_render_setup(enabled: bool):
    """Temporarily use SAPIEN's default shader during env setup."""
    if not enabled:
        yield
        return

    import sapien.render as sapien_render

    originals = {
        "set_camera_shader_dir": sapien_render.set_camera_shader_dir,
        "set_ray_tracing_samples_per_pixel": sapien_render.set_ray_tracing_samples_per_pixel,
        "set_ray_tracing_path_depth": sapien_render.set_ray_tracing_path_depth,
        "set_ray_tracing_denoiser": sapien_render.set_ray_tracing_denoiser,
    }

    def _set_camera_shader_dir_fast(shader_dir: str) -> None:
        originals["set_camera_shader_dir"]("default")

    sapien_render.set_camera_shader_dir = _set_camera_shader_dir_fast
    sapien_render.set_ray_tracing_samples_per_pixel = lambda spp: None
    sapien_render.set_ray_tracing_path_depth = lambda depth: None
    sapien_render.set_ray_tracing_denoiser = lambda name: None
    try:
        yield
    finally:
        for name, func in originals.items():
            setattr(sapien_render, name, func)


class RoboTwinBenchmark(StepBenchmark):
    """RoboTwin dual-arm manipulation benchmark (SAPIEN/CuRobo).

    Args:
        task_name: RoboTwin task (e.g. ``"grab_roller"``).
        task_config: Config name under ``task_config/`` (default ``"demo_clean"``).
        seed: Base seed index.  Starting seed = ``100000 * (1 + seed)``.
        instruction_type: Instruction variant (``"seen"`` or ``"unseen"``).
        test_num: Number of valid episodes to evaluate.
        skip_expert_check: If ``True``, skip oracle planner verification in
            ``get_tasks()`` (useful for quick smoke tests).
        fast_init: If ``True``, skip CuRobo planner warmup for qpos evaluation
            episodes after task discovery. This preserves the eval path used by
            the harness while substantially reducing cold-start time.
        fast_render: If ``True``, use SAPIEN's default camera shader instead of
            RoboTwin's ray-traced renderer. Faster, but observation fidelity may
            differ from the reference benchmark.
    """

    _ALL_RECORD_FIELDS = frozenset(
        {
            "reward",
            "done",
            "success",
            "action",
            "action_min",
            "action_max",
            "action_l2",
            "action_delta_from_prev_qpos_l2",
            "action_delta_from_prev_qpos_max_abs",
            "qpos",
            "qpos_min",
            "qpos_max",
            "qpos_l2",
            "qpos_step_delta_l2",
            "qpos_step_delta_max_abs",
            "object_proxy_count",
            "object_proxy_area",
            "object_proxy_bbox",
            "object_proxy_centroid",
            "object_proxy_extent_wh",
            "object_proxy_aspect",
            "object_proxy_top_edge",
            "object_proxy_components",
            "object_proxy_non_edge_area",
            "object_proxy_non_edge_bbox",
            "object_proxy_non_edge_centroid",
            "object_proxy_non_edge_extent_wh",
            "object_proxy_non_edge_aspect",
        }
    )

    def __init__(
        self,
        task_name: str,
        task_config: str = "demo_clean",
        seed: int = 0,
        instruction_type: str = "seen",
        test_num: int = 100,
        skip_expert_check: bool = False,
        fast_init: bool = True,
        fast_render: bool = False,
        oracle_suffix_at_step: int | None = None,
        oracle_suffix_trigger: str = "step",
        oracle_suffix_min_step: int = 0,
        oracle_suffix_max_step: int | None = None,
        oracle_suffix_non_edge_area_lte: int | None = None,
        oracle_suffix_non_edge_aspect_gte: float | None = None,
        oracle_suffix_mode: str = "task_default",
        oracle_suffix_end_episode: bool = True,
    ) -> None:
        import re

        super().__init__()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task_name):
            raise ValueError(f"Invalid task_name: {task_name!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_config):
            raise ValueError(f"Invalid task_config: {task_config!r}")
        self.task_name = task_name
        self.task_config = task_config
        self.seed = seed
        self.instruction_type = instruction_type
        self.test_num = test_num
        self.skip_expert_check = skip_expert_check
        self.fast_init = fast_init
        self.fast_render = fast_render
        self.oracle_suffix_at_step = oracle_suffix_at_step
        self.oracle_suffix_trigger = oracle_suffix_trigger
        self.oracle_suffix_min_step = oracle_suffix_min_step
        self.oracle_suffix_max_step = oracle_suffix_max_step
        self.oracle_suffix_non_edge_area_lte = oracle_suffix_non_edge_area_lte
        self.oracle_suffix_non_edge_aspect_gte = oracle_suffix_non_edge_aspect_gte
        self.oracle_suffix_mode = oracle_suffix_mode
        self.oracle_suffix_end_episode = oracle_suffix_end_episode
        self._env: Any = None
        self._env_class: Any = None
        self._args: dict[str, Any] | None = None
        self._last_qpos: np.ndarray | None = None
        self._oracle_suffix_ran = False
        self._oracle_suffix_result: dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Lazy init
    # -----------------------------------------------------------------

    def _init_robotwin(self) -> None:
        """Add RoboTwin paths, load YAML configs, resolve embodiment."""
        if self._args is not None:
            return

        for p in [ROBOTWIN_ROOT, f"{ROBOTWIN_ROOT}/policy", f"{ROBOTWIN_ROOT}/description/utils"]:
            if p not in sys.path:
                sys.path.insert(0, p)

        os.chdir(ROBOTWIN_ROOT)
        import yaml

        config_path = os.path.join(
            ROBOTWIN_ROOT,
            "task_config",
            f"{self.task_config}.yml",
        )
        with open(config_path) as f:
            args: dict[str, Any] = yaml.safe_load(f)

        args["task_name"] = self.task_name
        args["task_config"] = self.task_config

        from envs import CONFIGS_PATH

        embodiment_type = args.get("embodiment")
        if not isinstance(embodiment_type, list) or not all(isinstance(item, str) for item in embodiment_type):
            raise ValueError(f"RoboTwin config {config_path!r} must define embodiment as a list of strings")
        embodiment_type = cast(list[str], embodiment_type)
        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml")) as f:
            _embodiment_types = yaml.safe_load(f)

        def _get_file(etype: str) -> str:
            return _embodiment_types[etype]["file_path"]

        if len(embodiment_type) == 1:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = _get_file(embodiment_type[0])
            args["right_robot_file"] = _get_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False

        def _get_config(robot_file: str) -> dict:
            with open(os.path.join(robot_file, "config.yml")) as f:
                return yaml.safe_load(f)

        args["left_embodiment_config"] = _get_config(args["left_robot_file"])
        args["right_embodiment_config"] = _get_config(args["right_robot_file"])

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml")) as f:
            _camera_config = yaml.safe_load(f)

        hcam = args["camera"]["head_camera_type"]
        args["head_camera_h"] = _camera_config[hcam]["h"]
        args["head_camera_w"] = _camera_config[hcam]["w"]
        args["eval_mode"] = True

        self._args = args
        with _defer_open3d_import(enabled=not args.get("data_type", {}).get("pointcloud", False)):
            envs_module = importlib.import_module(f"envs.{self.task_name}")
        self._env_class = getattr(envs_module, self.task_name)
        logger.info("RoboTwin initialised: task=%s", self.task_name)

    def _create_env(self) -> Any:
        assert self._env_class is not None
        return self._env_class()

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception:
                pass
            self._env = None

    # -----------------------------------------------------------------
    # StepBenchmark interface
    # -----------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        self._init_robotwin()
        assert self._args is not None
        st_seed = 100000 * (1 + self.seed)

        if self.skip_expert_check:
            return [
                {
                    "name": self.task_name,
                    "suite": "robotwin",
                    "seed": st_seed + i,
                    "episode_idx": i,
                    "instruction": f"Perform the {self.task_name} task.",
                }
                for i in range(self.test_num)
            ]

        # Full expert check — run oracle planner per seed
        from generate_episode_instructions import generate_episode_descriptions

        env = self._create_env()
        tasks: list[Task] = []
        now_seed = st_seed
        episode_idx = 0
        logger.info("Running expert checks from seed %d ...", st_seed)

        while len(tasks) < self.test_num:
            try:
                env.setup_demo(
                    now_ep_num=episode_idx,
                    seed=now_seed,
                    is_test=True,
                    **self._args,
                )
                episode_info = env.play_once()
                env.close_env()
                if env.plan_success and env.check_success():
                    results = generate_episode_descriptions(
                        self.task_name,
                        [episode_info["info"]],
                        self.test_num,
                    )
                    instruction = np.random.choice(
                        results[0][self.instruction_type],
                    )
                    tasks.append(
                        {
                            "name": self.task_name,
                            "suite": "robotwin",
                            "seed": now_seed,
                            "episode_idx": episode_idx,
                            "instruction": instruction,
                        }
                    )
                    episode_idx += 1
            except Exception as e:
                logger.warning("Expert check failed for seed %d: %s", now_seed, e)
                try:
                    env.close_env()
                except Exception:
                    pass
            now_seed += 1
        return tasks

    def reset(self, task: Task) -> Any:
        self._init_robotwin()
        assert self._args is not None

        if self._env is not None:
            try:
                self._env.close_env(clear_cache=True)
            except Exception as e:
                logger.warning("Failed to close previous RoboTwin env: %s", e)
            self._env = None

        self._env = self._create_env()
        with _patched_robot_set_planner(self.fast_init), _patched_render_setup(self.fast_render):
            self._env.setup_demo(
                now_ep_num=task.get("episode_idx", 0),
                seed=task["seed"],
                is_test=True,
                **self._args,
            )
        self._env.set_instruction(instruction=task["instruction"])
        raw_obs = self._env.get_obs()
        self._last_qpos = self._extract_qpos(raw_obs)
        self._oracle_suffix_ran = False
        self._oracle_suffix_result = {}
        self._recorder.record_video(self._extract_frame(raw_obs))
        return raw_obs

    def step(self, action: Action) -> StepResult:
        raw = action.get("actions", action.get("action"))
        act = np.asarray(raw, dtype=np.float64).flatten()
        if len(act) > 14:
            act = act[:14]
        elif len(act) < 14:
            act = np.pad(act, (0, 14 - len(act)))
        assert act.shape[-1] == 14, f"Action dimension mismatch: got {act.shape[-1]}, expected 14"

        self._env.take_action(act, action_type="qpos")
        raw_obs = self._env.get_obs()
        qpos = self._extract_qpos(raw_obs)
        telemetry = self._make_action_telemetry(act, qpos, self._last_qpos)
        frame = self._extract_frame(raw_obs)
        telemetry.update(self._make_object_proxy_telemetry(frame))
        self._last_qpos = qpos
        success = bool(self._env.eval_success)
        done = success or (self._env.take_action_cnt >= self._env.step_lim)
        self._recorder.record_video(frame)
        self._recorder.record_step(
            reward=1.0 if success else 0.0,
            done=done,
            success=success,
            **telemetry,
        )
        should_run, trigger_reason = self._should_run_oracle_suffix(done, telemetry)
        if should_run:
            self._oracle_suffix_result = self._run_oracle_suffix(trigger_reason=trigger_reason, telemetry=telemetry)
            raw_obs = self._env.get_obs()
            success = bool(self._oracle_suffix_result.get("post_success", False) or self._env.eval_success)
            if success:
                self._env.eval_success = True
            done = bool(self.oracle_suffix_end_episode or success or (self._env.take_action_cnt >= self._env.step_lim))
        return StepResult(obs=raw_obs, reward=1.0 if success else 0.0, done=done, info={"success": success})

    def _should_run_oracle_suffix(self, done: bool, telemetry: dict[str, Any] | None = None) -> tuple[bool, str]:
        if done or self._oracle_suffix_ran:
            return False, ""

        step = int(self._env.take_action_cnt)
        if step < int(self.oracle_suffix_min_step):
            return False, ""
        if self.oracle_suffix_max_step is not None and step > int(self.oracle_suffix_max_step):
            return False, ""

        if self.oracle_suffix_trigger == "step":
            if self.oracle_suffix_at_step is None:
                return False, ""
            if step < int(self.oracle_suffix_at_step):
                return False, ""
            return True, f"step>={int(self.oracle_suffix_at_step)}"

        if self.oracle_suffix_trigger != "object_proxy_guard":
            raise ValueError(f"unsupported oracle_suffix_trigger={self.oracle_suffix_trigger!r}")

        telemetry = telemetry or {}
        reasons: list[str] = []
        if self.oracle_suffix_non_edge_area_lte is not None:
            area = int(telemetry.get("object_proxy_non_edge_area", 0))
            if area > int(self.oracle_suffix_non_edge_area_lte):
                return False, ""
            reasons.append(f"non_edge_area<={int(self.oracle_suffix_non_edge_area_lte)}")
        if self.oracle_suffix_non_edge_aspect_gte is not None:
            aspect = float(telemetry.get("object_proxy_non_edge_aspect", 0.0))
            if aspect < float(self.oracle_suffix_non_edge_aspect_gte):
                return False, ""
            reasons.append(f"non_edge_aspect>={float(self.oracle_suffix_non_edge_aspect_gte):.3f}")
        if not reasons:
            return False, ""
        return True, ",".join(reasons)

    def _run_oracle_suffix(self, *, trigger_reason: str, telemetry: dict[str, Any]) -> dict[str, Any]:
        self._oracle_suffix_ran = True
        result: dict[str, Any] = {
            "oracle_suffix_ran": True,
            "oracle_suffix_step": int(self._env.take_action_cnt),
            "oracle_suffix_trigger": self.oracle_suffix_trigger,
            "oracle_suffix_trigger_reason": trigger_reason,
            "oracle_suffix_mode": self.oracle_suffix_mode,
            "trigger_non_edge_area": int(telemetry.get("object_proxy_non_edge_area", 0)),
            "trigger_non_edge_aspect": float(telemetry.get("object_proxy_non_edge_aspect", 0.0)),
            "trigger_non_edge_bbox": telemetry.get("object_proxy_non_edge_bbox", []),
            "pre_success": bool(self._env.check_success()),
            "pre_plan_success": bool(getattr(self._env, "plan_success", False)),
            "post_success": False,
            "plan_success": bool(getattr(self._env, "plan_success", False)),
            "grasp_move_success": False,
            "lift_move_success": False,
            "place_move_success": False,
            "error": "",
        }
        try:
            if self.task_name != "pick_diverse_bottles":
                raise NotImplementedError(f"oracle suffix is not implemented for task {self.task_name!r}")
            if self.oracle_suffix_mode not in {"task_default", "pick_diverse_bottles"}:
                raise ValueError(f"unsupported oracle_suffix_mode={self.oracle_suffix_mode!r}")

            from envs.utils import ArmTag

            left = ArmTag("left")
            right = ArmTag("right")
            result["grasp_move_success"] = bool(
                self._env.move(
                    self._env.grasp_actor(self._env.bottle1, arm_tag=left, pre_grasp_dis=0.08),
                    self._env.grasp_actor(self._env.bottle2, arm_tag=right, pre_grasp_dis=0.08),
                )
            )
            result["lift_move_success"] = bool(
                self._env.move(
                    self._env.move_by_displacement(arm_tag=left, z=0.1),
                    self._env.move_by_displacement(arm_tag=right, z=0.1),
                )
            )
            result["place_move_success"] = bool(
                self._env.move(
                    self._env.place_actor(
                        self._env.bottle1,
                        target_pose=self._env.left_target_pose,
                        arm_tag=left,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                    ),
                    self._env.place_actor(
                        self._env.bottle2,
                        target_pose=self._env.right_target_pose,
                        arm_tag=right,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                    ),
                )
            )
            result["plan_success"] = bool(getattr(self._env, "plan_success", False))
            result["post_success"] = bool(self._env.check_success())
        except Exception as exc:
            logger.exception("RoboTwin oracle suffix failed")
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["plan_success"] = bool(getattr(self._env, "plan_success", False))
            try:
                result["post_success"] = bool(self._env.check_success())
            except Exception:
                result["post_success"] = False
        return result

    @staticmethod
    def _extract_frame(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        try:
            return np.asarray(raw_obs["observation"]["head_camera"]["rgb"])
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _extract_qpos(raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        try:
            qpos = np.asarray(raw_obs["joint_action"]["vector"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return None
        if qpos.size < 14:
            return np.pad(qpos, (0, 14 - qpos.size))
        if qpos.size > 14:
            return qpos[:14]
        return qpos

    @staticmethod
    def _make_action_telemetry(
        action: np.ndarray,
        qpos: np.ndarray | None,
        previous_qpos: np.ndarray | None,
    ) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)[:14]
        out: dict[str, Any] = {
            "action": action.tolist(),
            "action_min": float(np.min(action)),
            "action_max": float(np.max(action)),
            "action_l2": float(np.linalg.norm(action)),
        }
        if previous_qpos is not None:
            previous = np.asarray(previous_qpos, dtype=np.float64).reshape(-1)[:14]
            command_delta = action - previous
            out.update(
                {
                    "action_delta_from_prev_qpos_l2": float(np.linalg.norm(command_delta)),
                    "action_delta_from_prev_qpos_max_abs": float(np.max(np.abs(command_delta))),
                }
            )
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)[:14]
            out.update(
                {
                    "qpos": qpos.tolist(),
                    "qpos_min": float(np.min(qpos)),
                    "qpos_max": float(np.max(qpos)),
                    "qpos_l2": float(np.linalg.norm(qpos)),
                }
            )
            if previous_qpos is not None:
                previous = np.asarray(previous_qpos, dtype=np.float64).reshape(-1)[:14]
                qpos_delta = qpos - previous
                out.update(
                    {
                        "qpos_step_delta_l2": float(np.linalg.norm(qpos_delta)),
                        "qpos_step_delta_max_abs": float(np.max(np.abs(qpos_delta))),
                    }
                )
        return out

    @staticmethod
    def _make_object_proxy_telemetry(frame: np.ndarray | None) -> dict[str, Any]:
        """Track the largest red/orange object-like component in the head camera.

        This is a lightweight diagnostic for RoboTwin bottle tasks, not a
        general detector. It lets closed-loop recordings expose whether a
        bottle-like region stays upright, gets pushed to the image edge, or
        collapses into a horizontal/top-edge component.
        """
        if frame is None:
            return {
                "object_proxy_count": 0,
                "object_proxy_area": 0,
                "object_proxy_bbox": [],
                "object_proxy_centroid": [],
                "object_proxy_extent_wh": [],
                "object_proxy_aspect": 0.0,
                "object_proxy_top_edge": False,
                "object_proxy_components": [],
                "object_proxy_non_edge_area": 0,
                "object_proxy_non_edge_bbox": [],
                "object_proxy_non_edge_centroid": [],
                "object_proxy_non_edge_extent_wh": [],
                "object_proxy_non_edge_aspect": 0.0,
            }
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return {
                "object_proxy_count": 0,
                "object_proxy_area": 0,
                "object_proxy_bbox": [],
                "object_proxy_centroid": [],
                "object_proxy_extent_wh": [],
                "object_proxy_aspect": 0.0,
                "object_proxy_top_edge": False,
                "object_proxy_components": [],
                "object_proxy_non_edge_area": 0,
                "object_proxy_non_edge_bbox": [],
                "object_proxy_non_edge_centroid": [],
                "object_proxy_non_edge_extent_wh": [],
                "object_proxy_non_edge_aspect": 0.0,
            }
        rgb = arr[..., :3].astype(np.int16)
        red = rgb[..., 0]
        green = rgb[..., 1]
        blue = rgb[..., 2]
        mask = (red > 115) & (green < 135) & (blue < 115) & ((red - blue) > 45)
        components = RoboTwinBenchmark._connected_components(mask, min_area=50)
        if not components:
            return {
                "object_proxy_count": 0,
                "object_proxy_area": 0,
                "object_proxy_bbox": [],
                "object_proxy_centroid": [],
                "object_proxy_extent_wh": [],
                "object_proxy_aspect": 0.0,
                "object_proxy_top_edge": False,
                "object_proxy_components": [],
                "object_proxy_non_edge_area": 0,
                "object_proxy_non_edge_bbox": [],
                "object_proxy_non_edge_centroid": [],
                "object_proxy_non_edge_extent_wh": [],
                "object_proxy_non_edge_aspect": 0.0,
            }
        comp = components[0]
        width, height = comp["extent_wh"]
        aspect = float(width / max(height, 1))
        non_edge = next((candidate for candidate in components if candidate["bbox"][1] > 2), None)
        if non_edge is not None:
            non_edge_width, non_edge_height = non_edge["extent_wh"]
            non_edge_fields: dict[str, Any] = {
                "object_proxy_non_edge_area": non_edge["area"],
                "object_proxy_non_edge_bbox": non_edge["bbox"],
                "object_proxy_non_edge_centroid": non_edge["centroid"],
                "object_proxy_non_edge_extent_wh": non_edge["extent_wh"],
                "object_proxy_non_edge_aspect": float(non_edge_width / max(non_edge_height, 1)),
            }
        else:
            non_edge_fields = {
                "object_proxy_non_edge_area": 0,
                "object_proxy_non_edge_bbox": [],
                "object_proxy_non_edge_centroid": [],
                "object_proxy_non_edge_extent_wh": [],
                "object_proxy_non_edge_aspect": 0.0,
            }
        return {
            "object_proxy_count": len(components),
            "object_proxy_area": comp["area"],
            "object_proxy_bbox": comp["bbox"],
            "object_proxy_centroid": comp["centroid"],
            "object_proxy_extent_wh": comp["extent_wh"],
            "object_proxy_aspect": aspect,
            "object_proxy_top_edge": comp["bbox"][1] <= 2,
            "object_proxy_components": components[:5],
            **non_edge_fields,
        }

    @staticmethod
    def _connected_components(mask: np.ndarray, *, min_area: int) -> list[dict[str, Any]]:
        if mask.ndim != 2:
            return []
        height, width = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[dict[str, Any]] = []
        ys, xs = np.where(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if visited[y0, x0] or not mask[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            if len(pixels) < min_area:
                continue
            py = np.asarray([p[0] for p in pixels], dtype=np.float64)
            px = np.asarray([p[1] for p in pixels], dtype=np.float64)
            min_x = int(px.min())
            min_y = int(py.min())
            max_x = int(px.max())
            max_y = int(py.max())
            components.append(
                {
                    "area": int(len(pixels)),
                    "bbox": [min_x, min_y, max_x, max_y],
                    "centroid": [float(px.mean()), float(py.mean())],
                    "extent_wh": [max_x - min_x + 1, max_y - min_y + 1],
                }
            )
        components.sort(key=lambda comp: int(comp["area"]), reverse=True)
        return components

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": {
                "head_camera": raw_obs["observation"]["head_camera"]["rgb"],
                "left_camera": raw_obs["observation"]["left_camera"]["rgb"],
                "right_camera": raw_obs["observation"]["right_camera"]["rgb"],
            },
            "task_description": raw_obs.get(
                "language",
                task.get("instruction", ""),
            ),
            "joint_state": np.array(raw_obs["joint_action"]["vector"]),
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False), **self._oracle_suffix_result}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": 400,
            "task_name": self.task_name,
            "action_dim": 14,
            "max_episodes_per_task": self.test_num,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 14D dual-arm joint positions
        return {
            "joints": DimSpec("joints", 14, "joint_positions"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }
