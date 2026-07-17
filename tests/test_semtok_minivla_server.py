from __future__ import annotations

import numpy as np

from vla_eval.model_servers.semtok_minivla import SemTokMiniVLAModelServer


def _server_without_loading_model() -> SemTokMiniVLAModelServer:
    server = SemTokMiniVLAModelServer.__new__(SemTokMiniVLAModelServer)
    server.image_key = "cam_high"
    return server


def test_semtok_observation_aliases_robotwin_state_and_images() -> None:
    server = _server_without_loading_model()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    state = np.arange(14, dtype=np.float32)

    converted = server._to_semtok_observation(
        {
            "images": {"head_camera": image, "left_camera": image},
            "joint_state": state,
            "task_description": "pick the bottle",
        }
    )

    assert converted["images"]["cam_high"] is image
    assert converted["images"]["head_camera"] is image
    assert converted["observation"]["head_camera"]["rgb"] is image
    np.testing.assert_array_equal(converted["qpos"], state)
    np.testing.assert_array_equal(converted["joint_action"]["vector"], state)
    assert converted["language"] == "pick the bottle"


def test_semtok_observation_pads_smoke_state_to_robotwin_width() -> None:
    server = _server_without_loading_model()

    converted = server._to_semtok_observation(
        {
            "images": {"agentview": np.zeros((8, 8, 3), dtype=np.uint8)},
            "state": np.ones(8, dtype=np.float32),
            "task_description": "smoke test",
        }
    )

    assert converted["qpos"].shape == (14,)
    np.testing.assert_array_equal(converted["qpos"][:8], np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(converted["qpos"][8:], np.zeros(6, dtype=np.float32))
