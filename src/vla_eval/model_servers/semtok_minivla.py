# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "numpy>=1.24",
#     "pyyaml>=6.0",
#     "pillow>=9.0",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-07-07"
# ///
"""SemTok/OpenVLA-Mini model server bridge for RoboTwin.

This adapter lets ``vla-eval`` talk to the SemTok RoboTwin deploy policy
without duplicating the P0 runtime code.  Heavy OpenVLA-Mini weights are still
loaded lazily by SemTok's executor on the first action request.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, STATE_JOINT, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class SemTokMiniVLAModelServer(PredictModelServer):
    """Bridge SemTok's RoboTwin ``deploy_policy.py`` into vla-eval.

    Args:
        semtok_repo: Local SemTok repository path.
        robotwin_root: Local RoboTwin repository path.
        deploy_config: SemTok RoboTwin policy YAML, e.g.
            ``deploy_policy_p0_stabilized.yml``.
        image_key: Primary image key passed to SemTok; RoboTwin benchmark
            provides ``head_camera`` and this adapter aliases it to ``cam_high``.
        chunk_size: Action chunk size exposed to vla-eval. P0 is normally 1.
    """

    def __init__(
        self,
        semtok_repo: str = "/home/shkim_rllab/Desktop/semtok_vla",
        robotwin_root: str = "/home/shkim_rllab/Desktop/RoboTwin",
        deploy_config: str = (
            "/home/shkim_rllab/Desktop/semtok_vla/"
            "integrations/robotwin/policy/SemTokHarness/deploy_policy_p0_stabilized.yml"
        ),
        image_key: str = "cam_high",
        chunk_size: int = 1,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.semtok_repo = Path(semtok_repo).expanduser().resolve()
        self.robotwin_root = Path(robotwin_root).expanduser().resolve()
        self.deploy_config = Path(deploy_config).expanduser().resolve()
        self.image_key = image_key
        self._model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.semtok_repo.exists():
            raise FileNotFoundError(f"SemTok repo not found: {self.semtok_repo}")
        if not self.robotwin_root.exists():
            raise FileNotFoundError(f"RoboTwin root not found: {self.robotwin_root}")
        if not self.deploy_config.exists():
            raise FileNotFoundError(f"SemTok deploy config not found: {self.deploy_config}")

        for path in (
            str(self.semtok_repo),
            str(self.robotwin_root),
            str(self.robotwin_root / "policy"),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)

        from integrations.robotwin.policy.SemTokHarness.deploy_policy import get_model

        config = yaml.safe_load(self.deploy_config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"SemTok deploy config is not a mapping: {self.deploy_config}")
        config.setdefault("semtok_repo", str(self.semtok_repo))
        config.setdefault("image_key", self.image_key)
        self._model = get_model(config)
        logger.info("Loaded SemTok policy backend from %s", self.deploy_config)

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"joints": DimSpec("joints", 14, "joint_positions")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head_camera": IMAGE_RGB,
            "left_camera": IMAGE_RGB,
            "right_camera": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        del config, ctx
        assert self._model is not None
        self._model.reset_model()

    def _to_semtok_observation(self, obs: Observation) -> dict[str, Any]:
        images = obs.get("images", {})
        if not isinstance(images, dict):
            images = {}
        head = images.get("head_camera")
        if head is None:
            head = images.get("cam_high")
        if head is None:
            head = next(iter(images.values()), None)
        left = images.get("left_camera")
        right = images.get("right_camera")
        state = obs.get("state", obs.get("joint_state"))
        if state is None:
            raise KeyError("SemTok RoboTwin adapter requires state or joint_state")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size < 14:
            state_array = np.pad(state_array, (0, 14 - state_array.size))
        elif state_array.size > 14:
            state_array = state_array[:14]

        image_map = {str(key): value for key, value in images.items()}
        if head is not None:
            image_map.setdefault("cam_high", head)
            image_map.setdefault("head_camera", head)
        if left is not None:
            image_map.setdefault("left_camera", left)
        if right is not None:
            image_map.setdefault("right_camera", right)

        observation_tree: dict[str, Any] = {}
        for camera_name, value in (
            ("head_camera", head),
            ("left_camera", left),
            ("right_camera", right),
        ):
            if value is not None:
                observation_tree[camera_name] = {"rgb": value}

        return {
            "images": image_map,
            "observation": observation_tree,
            "state": state_array,
            "qpos": state_array,
            "joint_action": {"vector": state_array},
            "language": str(obs.get("task_description", "")),
        }

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        del ctx
        assert self._model is not None
        semtok_obs = self._to_semtok_observation(obs)
        instruction = str(obs.get("task_description") or semtok_obs.get("language") or "")
        self._model.update_obs(semtok_obs)
        self._model.bind_instruction(instruction)
        actions = np.asarray(self._model.get_action(), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"SemTok policy returned invalid action shape: {actions.shape}")
        return {"actions": actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(SemTokMiniVLAModelServer)
