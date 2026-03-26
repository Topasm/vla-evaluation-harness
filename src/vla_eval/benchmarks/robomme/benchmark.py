"""RoboMME benchmark implementation using ManiSkill3 fork + SAPIEN.

Creates a fresh environment per episode via BenchmarkEnvBuilder.
Each episode produces a conditioning video (via motion planning) that
is sent to the model server as ``video_history`` on the first observation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# Prevent display issues in headless environments
os.environ.setdefault("DISPLAY", "")

_DEFAULT_TASK_LIST = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


class RoboMMEBenchmark(StepBenchmark):
    """RoboMME (Memory-Augmented Manipulation Evaluation) benchmark.

    16 tasks across 4 cognitive suites (Counting, Permanence, Reference,
    Imitation).  Built on a ManiSkill3 fork with SAPIEN rendering.

    Non-obvious behaviors:
        - **Conditioning video**: On ``reset``, the environment runs motion
          planning to produce a demonstration trajectory.  These frames are
          sent as ``video_history`` in the first observation only.
        - **Fresh env per episode**: ``BenchmarkEnvBuilder.make_env_for_episode``
          creates a full wrapper chain for each episode.
        - **Error obs**: ``FailAwareWrapper`` returns ``obs=None`` on exception;
          ``EndeffectorDemonstrationWrapper`` returns ``obs={}`` on IK failure.
          Both are handled gracefully in ``make_obs``.
        - **Torch scalars**: ``reward``, ``terminated``, ``truncated`` may be
          torch tensors — always cast with ``float()`` / ``bool()``.

    Args:
        tasks: Subset of task names to evaluate.  ``None`` runs all 16.
        action_space: ``"joint_angle"`` (8D) or ``"ee_pose"`` (7D).
        dataset: Dataset split — ``"test"``, ``"val"``, or ``"train"``.
        max_steps: Maximum steps per episode (paper default: 1300).
        image_size: Camera resolution as ``[height, width]``.
        send_wrist_image: Include wrist camera in observations.
        send_state: Include proprioceptive state in observations.
        send_video_history: Send conditioning video on the first observation.
    """

    _rendering_configured: ClassVar[bool] = False

    def __init__(
        self,
        tasks: list[str] | None = None,
        action_space: str = "joint_angle",
        dataset: str = "test",
        max_steps: int = 1300,
        image_size: list[int] | None = None,
        send_wrist_image: bool = True,
        send_state: bool = True,
        send_video_history: bool = True,
    ) -> None:
        super().__init__()
        self.tasks = tasks or list(_DEFAULT_TASK_LIST)
        self.action_space = action_space
        self.dataset = dataset
        self.max_steps = max_steps
        self.image_size = tuple(image_size or [256, 256])
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.send_video_history = send_video_history

        self._env: Any = None
        self._task_description: str = ""
        self._video_frames: list[np.ndarray] = []
        self._wrist_video_frames: list[np.ndarray] = []
        self._video_states: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup_rendering(self) -> None:
        """Configure SAPIEN rendering for headless environments.

        SAPIEN 3.x uses Vulkan for all rendering.  The NVIDIA open kernel
        module (common on H100/A100 clusters) has incomplete Vulkan support
        that causes ``take_picture()`` to hang indefinitely when a scene
        contains geometry.

        When the lavapipe software Vulkan ICD is available the following
        patches are applied (once, idempotent):

        1. ``VK_ICD_FILENAMES`` → lavapipe ICD so SAPIEN uses CPU Vulkan.
        2. ``sapien.render.RenderSystem`` → ignore device args so the
           auto-detected lavapipe device is used instead of ``cuda:N``.
        3. ManiSkill3 ``render_backend`` → ``"sapien_cpu"`` so
           ``get_picture()`` returns CPU tensors (no CUDA↔Vulkan mapping).
        """
        if RoboMMEBenchmark._rendering_configured:
            return

        lavapipe_icd = os.environ.get("ROBOMME_LAVAPIPE_ICD", "/opt/lavapipe/lvp_icd.json")
        if not os.path.isfile(lavapipe_icd):
            # Fallback to default mesa location (container without Dockerfile patch)
            lavapipe_icd = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

        if not os.path.isfile(lavapipe_icd):
            logger.debug("Lavapipe ICD not found; using default NVIDIA Vulkan renderer")
            RoboMMEBenchmark._rendering_configured = True
            return

        # 1. Point Vulkan at the lavapipe software renderer
        os.environ["VK_ICD_FILENAMES"] = lavapipe_icd
        logger.info("SAPIEN rendering: using lavapipe software Vulkan (%s)", lavapipe_icd)

        import sapien.render as sr

        # 2. Make RenderSystem() ignore device args (lavapipe auto-detection)
        _OrigRenderSystem = sr.RenderSystem

        def _lavapipe_render_system(*args, **kwargs):
            return _OrigRenderSystem()

        sr.RenderSystem = _lavapipe_render_system

        # 3. Force ManiSkill3 to use the CPU get_picture() path
        from mani_skill.envs.utils.system.backend import (
            parse_sim_and_render_backend as _orig_parse,
        )

        def _patched_parse(sim_backend, render_backend):
            result = _orig_parse(sim_backend, render_backend)
            if result.render_backend == "sapien_cuda":
                result.render_backend = "sapien_cpu"
            return result

        import mani_skill.envs.sapien_env

        mani_skill.envs.sapien_env.parse_sim_and_render_backend = _patched_parse

        RoboMMEBenchmark._rendering_configured = True

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to ``self.image_size`` if dimensions differ."""
        if img.shape[:2] == self.image_size:
            return img
        import cv2

        return cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "env_id": t} for t in self.tasks]

    def reset(self, task: Task) -> Any:
        self._setup_rendering()
        import robomme.robomme_env  # noqa: F401 — registers gym environments
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        # Close previous env — fresh env per episode
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        builder = BenchmarkEnvBuilder(
            env_id=task["env_id"],
            dataset=self.dataset,
            action_space=self.action_space,
            gui_render=False,
            max_steps=self.max_steps,
        )
        self._env = builder.make_env_for_episode(task.get("episode_idx", 0))
        obs_batch, info_flat = self._env.reset()

        # Store conditioning video frames (demo trajectory, excluding final init frame)
        self._video_frames = list(obs_batch["front_rgb_list"][:-1])
        if self.send_wrist_image:
            self._wrist_video_frames = list(obs_batch.get("wrist_rgb_list", [])[:-1])

        # Store video states for memory-augmented models
        self._video_states = []
        for js, gs in zip(
            obs_batch.get("joint_state_list", [])[:-1],
            obs_batch.get("gripper_state_list", [])[:-1],
        ):
            joint = np.asarray(js, dtype=np.float64)
            gripper = np.asarray(gs, dtype=np.float64)[:1]
            self._video_states.append(np.concatenate([joint, gripper]).astype(np.float32))

        # Extract task description
        task_goal = info_flat["task_goal"]
        self._task_description = task_goal[0] if isinstance(task_goal, list) else str(task_goal)

        return obs_batch

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raise ValueError("Action dict must contain 'actions' or 'action' key")
        if hasattr(raw_action, "flatten"):
            raw_action = raw_action.flatten().tolist()
        elif not isinstance(raw_action, list):
            raw_action = list(raw_action)

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(raw_action)

        # Cast potential torch scalars
        terminated = bool(terminated)
        truncated = bool(truncated)
        reward = float(reward)
        done = terminated or truncated or info.get("status") == "error"

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Handle error cases (FailAwareWrapper → None, IK failure → {})
        if not raw_obs:
            return {"images": {}, "task_description": self._task_description}

        front_list = raw_obs.get("front_rgb_list", [])
        if not front_list:
            return {"images": {}, "task_description": self._task_description}

        front = front_list[-1]
        front = self._resize(front)

        obs: dict[str, Any] = {
            "images": {"agentview": front},
            "task_description": self._task_description,
        }

        if self.send_wrist_image:
            wrist_list = raw_obs.get("wrist_rgb_list")
            if wrist_list:
                obs["images"]["wrist"] = self._resize(wrist_list[-1])

        if self.send_state:
            joint = np.asarray(raw_obs["joint_state_list"][-1], dtype=np.float64)
            gripper = np.asarray(raw_obs["gripper_state_list"][-1], dtype=np.float64)[:1]
            obs["states"] = np.concatenate([joint, gripper]).astype(np.float32)

        if self.send_video_history and self._video_frames:
            obs["video_history"] = [self._resize(f) for f in self._video_frames]
            if self._video_states:
                obs["video_history_states"] = list(self._video_states)
            if self.send_wrist_image and self._wrist_video_frames:
                obs["wrist_video_history"] = [self._resize(f) for f in self._wrist_video_frames]
            obs["episode_restart"] = True
            # Clear — sent only once per episode
            self._video_frames = []
            self._wrist_video_frames = []
            self._video_states = []

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("status") == "success"}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps, "action_space": self.action_space}

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
