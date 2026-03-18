# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "tensorflow",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../.." }
#
# [tool.uv]
# exclude-newer = "2026-03-18T00:00:00Z"
# ///
"""SemTok-VLA model server for vla-evaluation-harness.

Supports two inference modes:
  1. Standard (continuous): uses predict_action() like vanilla OpenVLA
  2. Latent (semantic tokens): uses predict_latent_action() + ActionDecoder
     with temporal aggregation and history action tracking

Usage:
  # Standard mode (continuous actions)
  vla-eval serve --config configs/model_servers/semtok_vla.yaml

  # Latent action mode (semantic tokens + action decoder)
  vla-eval serve --config configs/model_servers/semtok_vla_latent.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action Decoder (mirrors semtok_vla/experiments/robot/libero/run_libero_eval.py)
# ---------------------------------------------------------------------------

def _lazy_import_semtok():
    """Import semtok_vla modules. Requires semtok_vla on sys.path."""
    from prismatic.models.policy.transformer_utils import MAPBlock
    return MAPBlock


class ActionDecoderHead(nn.Module):
    def __init__(self, window_size: int = 5):
        MAPBlock = _lazy_import_semtok()
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
        self.proj = nn.Sequential(
            nn.Linear(512, 7 * window_size),
            nn.Tanh(),
        )

    def forward(self, latent_action_tokens, visual_embed):
        latent_action_tokens = latent_action_tokens[:, -4:]
        visual_embed = self.visual_pool(visual_embed)
        action = self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))
        return action


class ActionDecoder(nn.Module):
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.net = ActionDecoderHead(window_size=window_size)
        self.temporal_size = window_size
        self.temporal_mask = torch.flip(
            torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)),
            dims=[1],
        ).numpy()

        balancing_factor = 0.1
        self.temporal_weights = np.array(
            [np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)]
        )[:, None]

        self.reset()

    def reset(self):
        self.action_buffer = np.zeros((self.temporal_size, self.temporal_size, 7))
        self.action_buffer_mask = np.zeros((self.temporal_size, self.temporal_size), dtype=np.bool_)

    def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
        pred_action = self.net(
            latent_actions.to(torch.float), visual_embed.to(torch.float)
        ).reshape(-1, self.temporal_size, 7)
        pred_action = np.array(pred_action.tolist())

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = pred_action
        self.action_buffer_mask[0] = np.array([True] * self.temporal_size, dtype=np.bool_)

        # Temporal ensemble
        action_prediction = (
            np.sum(
                self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights,
                axis=0,
            )
            / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
        )

        # Unnormalize
        action_prediction = np.where(
            mask,
            0.5 * (action_prediction + 1) * (action_high - action_low) + action_low,
            action_prediction,
        )
        return action_prediction


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def _center_crop_image(pil_image, crop_scale: float = 0.9):
    """Center crop then resize back to 224x224 (matching training augmentation)."""
    import tensorflow as tf
    img_array = np.array(pil_image)
    image = tf.convert_to_tensor(img_array)
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    new_h = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
    new_w = tf.clip_by_value(tf.sqrt(crop_scale), 0, 1)
    h_off = (1 - new_h) / 2
    w_off = (1 - new_w) / 2
    bounding_boxes = tf.reshape(
        tf.stack([h_off, w_off, h_off + new_h, w_off + new_w]),
        (1, 4),
    )
    image = tf.image.crop_and_resize(image, bounding_boxes, [0], (224, 224))
    image = tf.clip_by_value(image[0], 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    from PIL import Image as PILImage
    return PILImage.fromarray(image.numpy()).convert("RGB")


def _normalize_gripper_action(action, binarize: bool = True):
    """Normalize gripper [0,1] -> [-1,+1] and optionally binarize."""
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action):
    """Flip gripper sign: RLDS uses 0=close/1=open, env expects -1=open/+1=close."""
    action[..., -1] *= -1.0
    return action


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------

class SemTokVLAModelServer(PredictModelServer):
    """SemTok-VLA model server supporting both standard and latent action modes.

    Args:
        model_path: Path to the finetuned checkpoint (HF format).
        unnorm_key: Dataset key for action unnormalization stats.
        use_latent: If True, use predict_latent_action + ActionDecoder.
        action_decoder_path: Path to action_decoder.pt (required if use_latent).
        window_size: Temporal aggregation window for ActionDecoder.
        center_crop: Apply center crop (for models trained with image aug).
        crop_scale: Center crop scale factor.
        load_in_8bit: Load model with 8-bit quantization.
        load_in_4bit: Load model with 4-bit quantization.
        semtok_vla_path: Path to semtok_vla repo root (added to sys.path).
    """

    def __init__(
        self,
        model_path: str,
        unnorm_key: str | None = None,
        *,
        use_latent: bool = False,
        action_decoder_path: str | None = None,
        window_size: int = 12,
        center_crop: bool = True,
        crop_scale: float = 0.9,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        semtok_vla_path: str = "/scratch/e1816a02/semtok_vla",
        chunk_size: int = 1,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.use_latent = use_latent
        self.action_decoder_path = action_decoder_path
        self.window_size = window_size
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.semtok_vla_path = semtok_vla_path

        self._model = None
        self._processor = None
        self._action_decoder = None
        self._device = None

        # Per-session state for latent action mode
        self._hist_actions: dict[str, list[str]] = {}
        self._latent_detokenize = [f"<ACT_{i}>" for i in range(32)]

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        # Add semtok_vla to path so its modules can be imported
        if self.semtok_vla_path not in sys.path:
            sys.path.insert(0, self.semtok_vla_path)

        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        # Register custom model classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SemTok-VLA from %s on %s", self.model_path, self._device)

        # Attention backend
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if not self.load_in_8bit and not self.load_in_4bit:
            self._model = self._model.to(self._device)

        # Load dataset statistics for unnormalization
        stats_path = os.path.join(self.model_path, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                self._model.norm_stats = json.load(f)

        # Auto-detect unnorm_key with _no_noops suffix
        if self.unnorm_key and self.unnorm_key not in self._model.norm_stats:
            alt_key = f"{self.unnorm_key}_no_noops"
            if alt_key in self._model.norm_stats:
                logger.info("Using unnorm_key '%s' (auto-detected _no_noops suffix)", alt_key)
                self.unnorm_key = alt_key

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # Load action decoder for latent mode
        if self.use_latent:
            if not self.action_decoder_path:
                raise ValueError("action_decoder_path is required when use_latent=True")
            logger.info("Loading ActionDecoder from %s (window_size=%d)", self.action_decoder_path, self.window_size)
            self._action_decoder = ActionDecoder(window_size=self.window_size)
            self._action_decoder.net.load_state_dict(torch.load(self.action_decoder_path, map_location="cpu"))
            self._action_decoder.eval().to(self._device)

        logger.info("SemTok-VLA model loaded (use_latent=%s).", self.use_latent)

    # ------------------------------------------------------------------
    # Observation → PIL
    # ------------------------------------------------------------------

    def _obs_to_pil(self, obs: Observation) -> Any:
        from PIL import Image as PILImage

        # Eval harness sends obs["images"] as dict of camera_name -> ndarray
        images_dict = obs.get("images", {})
        if isinstance(images_dict, dict) and images_dict:
            img_array = next(iter(images_dict.values()))
        else:
            # Fallback: direct ndarray or "full_image" key
            img_array = obs.get("full_image", images_dict)

        if isinstance(img_array, np.ndarray):
            pil_image = PILImage.fromarray(img_array).convert("RGB")
        else:
            pil_image = img_array

        if self.center_crop:
            pil_image = _center_crop_image(pil_image, self.crop_scale)

        return pil_image

    # ------------------------------------------------------------------
    # Episode lifecycle (reset stateful buffers)
    # ------------------------------------------------------------------

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        sid = ctx.session_id
        self._hist_actions[sid] = [""]
        if self._action_decoder is not None:
            self._action_decoder.reset()

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_end(result, ctx)
        self._hist_actions.pop(ctx.session_id, None)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._model is not None and self._processor is not None

        pil_image = self._obs_to_pil(obs)
        task_description = obs.get("task_description", "")

        if self.use_latent:
            return self._predict_latent(pil_image, task_description, ctx)
        else:
            return self._predict_continuous(pil_image, task_description)

    def _predict_continuous(self, pil_image, task_description: str) -> Action:
        """Standard OpenVLA-style continuous action prediction."""
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch.bfloat16)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key

        action = self._model.predict_action(**inputs, **kwargs)

        # Normalize + invert gripper for LIBERO-style envs
        action = _normalize_gripper_action(action, binarize=True)
        action = _invert_gripper_action(action)

        return {"actions": action}

    def _predict_latent(self, pil_image, task_description: str, ctx: SessionContext) -> Action:
        """Latent action prediction with ActionDecoder and temporal aggregation."""
        assert self._action_decoder is not None

        sid = ctx.session_id
        hist = self._hist_actions.get(sid, [""])

        # Build prompt with history
        last_hist = hist[-1] if hist else ""
        if len(last_hist) > 0:
            prompt = f"In: What action should the robot take to {task_description.lower()}? History action {last_hist}\nOut:"
        else:
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch.bfloat16)

        # Get latent action tokens + visual embeddings
        latent_action, visual_embed, generated_ids = self._model.predict_latent_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
        )

        # Record history
        hist_str = ""
        for token_id in generated_ids[0]:
            hist_str += self._latent_detokenize[token_id.item() - 32001]
        hist.append(hist_str)

        # Get action norm stats
        action_norm_stats = self._model.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])

        # Decode latent → continuous action with temporal aggregation
        action = self._action_decoder(latent_action, visual_embed, mask, action_low, action_high)

        # Normalize + invert gripper
        action = _normalize_gripper_action(action, binarize=True)
        action = _invert_gripper_action(action)

        return {"actions": action}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemTok-VLA model server")
    parser.add_argument("--model_path", required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--unnorm_key", default=None, help="Unnormalization key (e.g. 'libero_spatial_no_noops')")
    parser.add_argument("--use_latent", action="store_true", help="Use latent action mode (semantic tokens)")
    parser.add_argument("--action_decoder_path", default=None, help="Path to action_decoder.pt")
    parser.add_argument("--window_size", type=int, default=12, help="Temporal aggregation window")
    parser.add_argument("--center_crop", action="store_true", help="Apply center crop (for aug-trained models)")
    parser.add_argument("--crop_scale", type=float, default=0.9, help="Center crop scale")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--semtok_vla_path", default="/scratch/e1816a02/semtok_vla")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = SemTokVLAModelServer(
        model_path=args.model_path,
        unnorm_key=args.unnorm_key,
        use_latent=args.use_latent,
        action_decoder_path=args.action_decoder_path,
        window_size=args.window_size,
        center_crop=args.center_crop,
        crop_scale=args.crop_scale,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        semtok_vla_path=args.semtok_vla_path,
        chunk_size=args.chunk_size,
        action_ensemble=args.action_ensemble,
    )

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
