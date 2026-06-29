#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
MAX_TEXT_SEQUENCE_LENGTH = 226
QUALITY_INDEX = 0
CAMERA_KEY_MAP = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}
CAMERA_RESIZE = {
    "observation.images.cam_high": (256, 320),
    "observation.images.cam_left_wrist": (128, 160),
    "observation.images.cam_right_wrist": (128, 160),
}
DATASET_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": [16],
        "names": ["left_eef", "left_gripper", "right_eef", "right_gripper"],
    },
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "coarse_task_index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
    "coarse_quality_index": {"dtype": "int64", "shape": [1], "names": None},
    "quality_index": {"dtype": "int64", "shape": [1], "names": None},
    "task_name": {"dtype": "string", "shape": [1], "names": ["task_name"]},
    "language_instruction": {
        "dtype": "string",
        "shape": [1],
        "names": ["language_instruction"],
    },
}


class ConversionError(RuntimeError):
    pass


class MissingDependencyError(ConversionError):
    pass


@dataclass(frozen=True)
class ActionSegment:
    start_frame: int
    end_frame: int
    action_text: str
    source: str


@dataclass(frozen=True)
class EpisodeCandidate:
    task_name: str
    task_root: Path
    episode_id: int
    hdf5_path: Path
    instruction_path: Path
    video_path: Path | None
    language_annotation_path: Path | None


@dataclass(frozen=True)
class ValidatedEpisode:
    candidate: EpisodeCandidate
    episode_prompt: str
    frame_count: int
    source_fps: float
    segments: tuple[ActionSegment, ...]


def _import_or_raise(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            f"Missing required dependency '{module_name}'. Install it first, e.g. `{install_hint}`."
        ) from exc


def _load_h5py():
    return _import_or_raise("h5py", "pip install h5py")


def _load_pyarrow():
    return _import_or_raise("pyarrow", "pip install pyarrow")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, feature_stats in stats.items():
        out[key] = {}
        for stat_name, value in feature_stats.items():
            out[key][stat_name] = np.asarray(value).tolist()
    return out


def compute_feature_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    keepdims = array.ndim == 1
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.array([len(array)], dtype=np.int64),
    }


def compute_episode_stats(
    episode_rows: dict[str, list[Any]],
    features: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, spec in features.items():
        if spec["dtype"] == "string":
            continue
        array = np.asarray(episode_rows[key])
        if spec["dtype"] == "float32":
            array = array.astype(np.float32, copy=False)
        elif spec["dtype"] in {"int64", "int32"}:
            array = array.astype(np.int64, copy=False)
        elif spec["dtype"] == "bool":
            array = array.astype(bool, copy=False)
        stats[key] = compute_feature_stats(array)
    return stats


def aggregate_feature_stats(stats_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)

    counts_for_weight = counts
    while counts_for_weight.ndim < means.ndim:
        counts_for_weight = np.expand_dims(counts_for_weight, axis=-1)

    total_mean = (means * counts_for_weight).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = (
        ((variances + delta_means**2) * counts_for_weight).sum(axis=0) / total_count
    )
    return {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    keys = {key for episode_stats in stats_list for key in episode_stats}
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in sorted(keys):
        out[key] = aggregate_feature_stats([episode_stats[key] for episode_stats in stats_list if key in episode_stats])
    return out


def parse_task_names(tasks_arg: str) -> list[str] | None:
    if tasks_arg.strip().lower() == "all":
        return None
    task_names = [item.strip() for item in tasks_arg.split(",") if item.strip()]
    if not task_names:
        raise ConversionError("`--tasks` is empty. Use a comma-separated list or `all`.")
    return task_names


def find_task_roots(rmbench_root: Path, setting: str, task_names: list[str] | None) -> list[Path]:
    if task_names is None:
        candidates = sorted(path for path in rmbench_root.iterdir() if path.is_dir())
    else:
        candidates = [rmbench_root / task_name for task_name in task_names]

    task_roots: list[Path] = []
    missing: list[str] = []
    for task_dir in candidates:
        setting_dir = task_dir / setting
        if setting_dir.is_dir():
            task_roots.append(setting_dir)
        else:
            missing.append(str(setting_dir))
    if missing:
        raise ConversionError(
            "Missing RMBench task directories:\n" + "\n".join(f"- {path}" for path in missing)
        )
    if not task_roots:
        raise ConversionError(f"No RMBench task directories found under {rmbench_root}.")
    return sorted(task_roots)


def parse_episode_id(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if not match:
        raise ConversionError(f"Unexpected RMBench episode filename: {path.name}")
    return int(match.group(1))


def scan_episode_candidates(task_root: Path) -> list[EpisodeCandidate]:
    data_dir = task_root / "data"
    instruction_dir = task_root / "instructions"
    if not data_dir.is_dir():
        raise ConversionError(f"Missing data directory: {data_dir}")
    if not instruction_dir.is_dir():
        raise ConversionError(f"Missing instructions directory: {instruction_dir}")

    candidates: list[EpisodeCandidate] = []
    for hdf5_path in sorted(data_dir.glob("episode*.hdf5")):
        episode_id = parse_episode_id(hdf5_path)
        instruction_path = instruction_dir / f"episode{episode_id}.json"
        if not instruction_path.is_file():
            raise ConversionError(f"Missing instruction file: {instruction_path}")
        video_path = task_root / "video" / f"episode{episode_id}.mp4"
        language_annotation_path = task_root / "language_annotation.json"
        candidates.append(
            EpisodeCandidate(
                task_name=task_root.parent.name,
                task_root=task_root,
                episode_id=episode_id,
                hdf5_path=hdf5_path,
                instruction_path=instruction_path,
                video_path=video_path if video_path.is_file() else None,
                language_annotation_path=(
                    language_annotation_path if language_annotation_path.is_file() else None
                ),
            )
        )
    if not candidates:
        raise ConversionError(f"No episode*.hdf5 files found in {data_dir}")
    return candidates


def list_hdf5_keys(node, prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key in node.keys():
        child = node[key]
        child_path = f"{prefix}/{key}"
        if hasattr(child, "keys"):
            keys.extend(list_hdf5_keys(child, child_path))
        else:
            keys.append(child_path)
    return keys


def load_instruction_prompt(path: Path, desc_type: str) -> str:
    payload = load_json(path)
    if desc_type not in payload:
        raise ConversionError(
            f"{path} does not contain desc_type='{desc_type}'. Available keys: {sorted(payload.keys())}"
        )
    prompts = payload[desc_type]
    if not isinstance(prompts, list) or not prompts:
        raise ConversionError(f"{path} contains no prompt under desc_type='{desc_type}'.")
    prompt = str(prompts[0]).strip()
    if not prompt:
        raise ConversionError(f"{path} contains an empty prompt under desc_type='{desc_type}'.")
    return prompt


def infer_source_fps(video_path: Path | None, fallback_fps: float) -> float:
    if video_path is None:
        return fallback_fps
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return fallback_fps
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps is None or fps <= 0:
        return fallback_fps
    return float(fps)


def load_language_annotation_segments(
    candidate: EpisodeCandidate,
    episode_prompt: str,
    episode_length: int,
) -> tuple[ActionSegment, ...]:
    if candidate.language_annotation_path is None:
        raise ConversionError(
            f"Missing language_annotation.json for {candidate.task_name}/episode{candidate.episode_id}."
        )
    payload = load_json(candidate.language_annotation_path)
    episode_key = f"episode_{candidate.episode_id}"
    annotations = payload.get(episode_key)
    if not isinstance(annotations, list) or not annotations:
        raise ConversionError(
            f"{candidate.language_annotation_path} is missing a non-empty `{episode_key}` entry."
        )

    cursor = 0
    segments: list[ActionSegment] = []
    for raw_item in annotations:
        if not isinstance(raw_item, (list, tuple)) or len(raw_item) < 2:
            raise ConversionError(
                f"Invalid language annotation item for {episode_key}: {raw_item!r}"
            )
        text = str(raw_item[0]).strip() or episode_prompt
        span = int(raw_item[1])
        if span <= 0:
            continue
        end_frame = min(cursor + span, episode_length)
        if end_frame - cursor >= 2:
            segments.append(
                ActionSegment(
                    start_frame=cursor,
                    end_frame=end_frame,
                    action_text=text,
                    source="language_annotation",
                )
            )
        cursor = end_frame

    if cursor < episode_length:
        tail_text = segments[-1].action_text if segments else episode_prompt
        if episode_length - cursor >= 2:
            segments.append(
                ActionSegment(
                    start_frame=cursor,
                    end_frame=episode_length,
                    action_text=tail_text,
                    source="language_annotation_tail",
                )
            )

    if not segments:
        raise ConversionError(
            f"language_annotation produced no valid segments for {candidate.task_name}/episode{candidate.episode_id}."
        )
    return tuple(segments)


def build_segments(
    candidate: EpisodeCandidate,
    episode_prompt: str,
    episode_length: int,
    segment_source: str,
) -> tuple[ActionSegment, ...]:
    if segment_source == "episode_instruction":
        return (
            ActionSegment(
                start_frame=0,
                end_frame=episode_length,
                action_text=episode_prompt,
                source="episode_instruction",
            ),
        )
    if segment_source == "language_annotation":
        return load_language_annotation_segments(candidate, episode_prompt, episode_length)
    raise ConversionError(f"Unsupported segment_source={segment_source!r}")


def resolve_frame_stride(source_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        raise ConversionError(f"target_fps must be > 0, got {target_fps}.")
    if source_fps <= 0:
        raise ConversionError(f"source_fps must be > 0, got {source_fps}.")

    # Downstream action alignment assumes a fixed integer frame stride.
    return max(int(round(source_fps / target_fps)), 1)


def sample_frame_ids(
    start_frame: int,
    end_frame: int,
    source_fps: float,
    target_fps: float,
) -> list[int]:
    if end_frame <= start_frame:
        return []
    frame_stride = resolve_frame_stride(source_fps=source_fps, target_fps=target_fps)
    frame_ids = list(range(start_frame, end_frame, frame_stride))
    if len(frame_ids) < 2 and end_frame - start_frame >= 2:
        frame_ids = [start_frame, end_frame - 1]
    return frame_ids


def validate_episode(candidate: EpisodeCandidate, args: argparse.Namespace) -> ValidatedEpisode:
    h5py = _load_h5py()
    episode_prompt = load_instruction_prompt(candidate.instruction_path, args.desc_type)
    source_fps = infer_source_fps(candidate.video_path, args.source_fps)

    with h5py.File(candidate.hdf5_path, "r") as root:
        available_keys = set(list_hdf5_keys(root))
        required_rgb_keys = {f"/observation/{camera}/rgb" for camera in CAMERA_KEY_MAP}
        required_endpose_keys = {
            "/endpose/left_endpose",
            "/endpose/left_gripper",
            "/endpose/right_endpose",
            "/endpose/right_gripper",
        }
        missing_keys = sorted((required_rgb_keys | required_endpose_keys) - available_keys)
        if missing_keys:
            raise ConversionError(
                f"{candidate.hdf5_path} is missing required keys:\n"
                + "\n".join(f"- {key}" for key in missing_keys)
            )

        frame_counts = {
            camera_name: int(root[f"/observation/{camera_name}/rgb"].shape[0])
            for camera_name in CAMERA_KEY_MAP
        }
        left_eef = root["/endpose/left_endpose"]
        right_eef = root["/endpose/right_endpose"]
        left_gripper = root["/endpose/left_gripper"]
        right_gripper = root["/endpose/right_gripper"]
        endpose_count = int(left_eef.shape[0])
        expected_counts = {
            "left_endpose": int(left_eef.shape[0]),
            "right_endpose": int(right_eef.shape[0]),
            "left_gripper": int(left_gripper.shape[0]),
            "right_gripper": int(right_gripper.shape[0]),
        }
        unique_counts = set(frame_counts.values()) | set(expected_counts.values())
        if len(unique_counts) != 1:
            raise ConversionError(
                f"Frame count mismatch in {candidate.hdf5_path}: "
                f"camera_counts={frame_counts}, endpose_counts={expected_counts}"
            )
        if tuple(left_eef.shape[1:]) != (7,) or tuple(right_eef.shape[1:]) != (7,):
            raise ConversionError(
                f"{candidate.hdf5_path} has invalid endpose shape: "
                f"left={tuple(left_eef.shape)}, right={tuple(right_eef.shape)}"
            )

        segments = build_segments(candidate, episode_prompt, endpose_count, args.segment_source)
        for segment in segments:
            frame_ids = sample_frame_ids(
                segment.start_frame,
                segment.end_frame,
                source_fps=source_fps,
                target_fps=args.target_fps,
            )
            if len(frame_ids) < 2:
                raise ConversionError(
                    f"{candidate.hdf5_path} segment ({segment.start_frame}, {segment.end_frame}) "
                    f"produces fewer than 2 sampled frames at target_fps={args.target_fps}."
                )

    return ValidatedEpisode(
        candidate=candidate,
        episode_prompt=episode_prompt,
        frame_count=endpose_count,
        source_fps=source_fps,
        segments=segments,
    )


def decode_rgb_frame(raw_frame: Any) -> np.ndarray:
    if isinstance(raw_frame, np.ndarray) and raw_frame.ndim == 3:
        frame = raw_frame
    else:
        raw_bytes = raw_frame.tobytes() if hasattr(raw_frame, "tobytes") else bytes(raw_frame)
        raw_bytes = raw_bytes.rstrip(b"\0")
        if not raw_bytes:
            raise ConversionError("Encountered an empty encoded RGB frame.")
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ConversionError("cv2.imdecode failed for an RMBench RGB frame.")
        # RMBench stores RGB arrays and pkl2hdf5 encodes them directly with cv2.imencode.
        # Keeping the decoded channel order reproduces the original numeric RGB tensor.
        frame = decoded

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ConversionError(f"Decoded frame has invalid shape {frame.shape}. Expected HxWx3.")
    return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def decode_rgb_dataset(dataset: Any) -> np.ndarray:
    frames = [decode_rgb_frame(dataset[idx]) for idx in range(int(dataset.shape[0]))]
    return np.stack(frames, axis=0)


def build_compact_endpose_action(root: Any) -> np.ndarray:
    left_endpose = np.asarray(root["/endpose/left_endpose"], dtype=np.float32)
    right_endpose = np.asarray(root["/endpose/right_endpose"], dtype=np.float32)
    left_gripper = np.asarray(root["/endpose/left_gripper"], dtype=np.float32).reshape(-1, 1)
    right_gripper = np.asarray(root["/endpose/right_gripper"], dtype=np.float32).reshape(-1, 1)
    if left_endpose.shape[1] != 7 or right_endpose.shape[1] != 7:
        raise ConversionError(
            f"Expected 7D endpose arrays, got left={left_endpose.shape}, right={right_endpose.shape}."
        )
    action = np.concatenate(
        [left_endpose, left_gripper, right_endpose, right_gripper],
        axis=1,
    )
    if action.shape[1] != 16:
        raise ConversionError(f"Expected compact 16D action, got shape {action.shape}.")
    return action.astype(np.float32, copy=False)


def load_episode_content(validated_episode: ValidatedEpisode) -> tuple[dict[str, np.ndarray], np.ndarray]:
    h5py = _load_h5py()
    with h5py.File(validated_episode.candidate.hdf5_path, "r") as root:
        frames_by_camera = {
            output_camera_key: decode_rgb_dataset(root[f"/observation/{source_camera}/rgb"])
            for source_camera, output_camera_key in CAMERA_KEY_MAP.items()
        }
        action = build_compact_endpose_action(root)
    for camera_key, frames in frames_by_camera.items():
        if frames.shape[0] != action.shape[0]:
            raise ConversionError(
                f"{validated_episode.candidate.hdf5_path} has camera/action mismatch for {camera_key}: "
                f"{frames.shape[0]} vs {action.shape[0]}"
            )
    return frames_by_camera, action


def resize_rgb_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(resized, axis=0)


def save_debug_images(
    debug_root: Path,
    task_name: str,
    episode_index: int,
    sampled_frames: dict[str, np.ndarray],
) -> None:
    for camera_key, frames in sampled_frames.items():
        if len(frames) == 0:
            continue
        path = debug_root / task_name / f"episode_{episode_index:06d}_{camera_key.replace('.', '_')}.png"
        ensure_dir(path.parent)
        cv2.imwrite(str(path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))


class LatentAndTextExtractor:
    def __init__(self, model_path: Path, device: str):
        for subdir in ("vae", "tokenizer", "text_encoder"):
            required_path = model_path / subdir
            if not required_path.is_dir():
                raise ConversionError(f"Missing model component: {required_path}")

        try:
            from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # type: ignore
        except ModuleNotFoundError:
            prompt_clean = lambda text: text

        from einops import rearrange
        from wan_va.modules.utils import (
            WanVAEStreamingWrapper,
            load_text_encoder,
            load_tokenizer,
            load_vae,
        )

        self._rearrange = rearrange
        self.prompt_clean = prompt_clean
        self.device = torch.device(device)
        if self.device.type.startswith("cuda") and not torch.cuda.is_available():
            raise ConversionError(f"Requested device `{device}` but CUDA is not available.")
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.save_dtype = torch.bfloat16

        self.vae = load_vae(
            str(model_path / "vae"),
            torch_dtype=self.compute_dtype,
            torch_device=self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(str(model_path / "tokenizer"))
        self.text_encoder = load_text_encoder(
            str(model_path / "text_encoder"),
            torch_dtype=self.compute_dtype,
            torch_device=self.device,
        )
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device)
        self.latents_std = torch.tensor(self.vae.config.latents_std, device=self.device)
        self.prompt_cache: dict[str, torch.Tensor] = {}

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * (1.0 / latents_std)).to(latents.dtype)

    @torch.inference_mode()
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt = self.prompt_clean(prompt)
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt].clone()

        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=MAX_TEXT_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        hidden = self.text_encoder(input_ids, attention_mask).last_hidden_state
        hidden = hidden.to(device=self.device, dtype=self.compute_dtype)
        trimmed = [seq[: seq_len.item()] for seq, seq_len in zip(hidden, seq_lens)]
        padded = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        seq.new_zeros(MAX_TEXT_SEQUENCE_LENGTH - seq.size(0), seq.size(1)),
                    ]
                )
                for seq in trimmed
            ],
            dim=0,
        )
        encoded = padded[0].to(self.save_dtype).cpu()
        self.prompt_cache[prompt] = encoded
        return encoded.clone()

    @torch.inference_mode()
    def encode_video(self, frames_rgb: np.ndarray) -> tuple[torch.Tensor, int, int, int]:
        self.streaming_vae.clear_cache()
        video_tensor = (
            torch.from_numpy(frames_rgb)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        video_tensor = video_tensor / 255.0 * 2.0 - 1.0
        encoded = self.streaming_vae.encode_chunk(video_tensor.to(self.compute_dtype))
        mu, _ = torch.chunk(encoded, 2, dim=1)
        mu = self.normalize_latents(mu)
        _, channel_dim, latent_num_frames, latent_height, latent_width = mu.shape
        flat = self._rearrange(mu[0], "c f h w -> (f h w) c").to(self.save_dtype).cpu()
        if channel_dim <= 0 or latent_num_frames <= 0 or latent_height <= 0 or latent_width <= 0:
            raise ConversionError(f"Invalid latent shape returned by VAE: {tuple(mu.shape)}")
        return flat, latent_num_frames, latent_height, latent_width


def build_parquet_rows(
    action: np.ndarray,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    coarse_task_index: int,
    task_name: str,
    episode_prompt: str,
    source_fps: float,
) -> dict[str, list[Any]]:
    rows: dict[str, list[Any]] = {key: [] for key in DATASET_FEATURES}
    for frame_index in range(action.shape[0]):
        rows["action"].append(action[frame_index].astype(np.float32).tolist())
        rows["timestamp"].append(np.float32(frame_index / source_fps).item())
        rows["frame_index"].append(frame_index)
        rows["episode_index"].append(episode_index)
        rows["index"].append(global_start_index + frame_index)
        rows["task_index"].append(task_index)
        rows["coarse_task_index"].append(coarse_task_index)
        rows["quality_index"].append(QUALITY_INDEX)
        rows["coarse_quality_index"].append(QUALITY_INDEX)
        rows["task_name"].append(task_name)
        rows["language_instruction"].append(episode_prompt)
    return rows


def feature_to_arrow_type(pa_module, spec: dict[str, Any]):
    dtype = spec["dtype"]
    if dtype == "float32":
        base = pa_module.float32()
    elif dtype == "int64":
        base = pa_module.int64()
    elif dtype == "int32":
        base = pa_module.int32()
    elif dtype == "bool":
        base = pa_module.bool_()
    elif dtype == "string":
        return pa_module.string()
    else:
        raise ConversionError(f"Unsupported feature dtype in info.json: {dtype}")

    shape = spec["shape"]
    if len(shape) == 1 and shape[0] == 1:
        return base
    if len(shape) == 1:
        return pa_module.list_(base, shape[0])
    raise ConversionError(f"Only scalar and 1D fixed-size features are supported, got shape={shape}")


def write_episode_parquet(path: Path, rows: dict[str, list[Any]]) -> None:
    pa = _load_pyarrow()
    import pyarrow.parquet as pq

    ensure_dir(path.parent)
    schema = pa.schema(
        [
            pa.field(name, feature_to_arrow_type(pa, spec))
            for name, spec in DATASET_FEATURES.items()
        ]
    )
    table = pa.Table.from_pydict(rows, schema=schema)
    pq.write_table(table, path)


def latent_file_path(dataset_root: Path, episode_index: int, camera_key: str, start_frame: int, end_frame: int) -> Path:
    chunk_index = episode_index // DEFAULT_CHUNK_SIZE
    return (
        dataset_root
        / "latents"
        / f"chunk-{chunk_index:03d}"
        / camera_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def parquet_file_path(dataset_root: Path, episode_index: int) -> Path:
    chunk_index = episode_index // DEFAULT_CHUNK_SIZE
    return dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"


def build_info(total_episodes: int, total_frames: int, total_tasks: int, fps: float) -> dict[str, Any]:
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "robotwin_tshape",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 0,
        "total_chunks": math.ceil(total_episodes / DEFAULT_CHUNK_SIZE) if total_episodes else 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": DATASET_FEATURES,
    }


def build_dataset_readme(output_dir: Path) -> str:
    return (
        f"# RMBench LingBot-VA Dataset\n\n"
        f"This dataset was converted into the minimal LeRobot v2.1 layout expected by LingBot-VA.\n\n"
        f"- Root: `{output_dir}`\n"
        f"- Actions: compact 16D dual-arm end-effector actions\n"
        f"- Latents: `latents/chunk-XXX/<camera>/episode_XXXXXX_<start>_<end>.pth`\n"
        f"- Empty prompt embedding: `empty_emb.pt`\n"
    )


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise ConversionError(
                f"Output directory already exists: {output_dir}. Re-run with `--overwrite` to replace it."
            )
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rmbench_root = args.rmbench_root.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()

    task_roots = find_task_roots(
        rmbench_root=rmbench_root,
        setting=args.setting,
        task_names=parse_task_names(args.tasks),
    )

    all_candidates: list[EpisodeCandidate] = []
    for task_root in task_roots:
        all_candidates.extend(scan_episode_candidates(task_root))

    valid_episodes: list[ValidatedEpisode] = []
    skipped_episodes: list[dict[str, Any]] = []
    for candidate in all_candidates:
        try:
            valid_episodes.append(validate_episode(candidate, args))
        except ConversionError as exc:
            skipped_episodes.append(
                {
                    "task_name": candidate.task_name,
                    "source_episode_id": candidate.episode_id,
                    "hdf5_path": str(candidate.hdf5_path),
                    "reason": str(exc),
                }
            )
            if not args.allow_skip_invalid:
                raise

    if not valid_episodes:
        raise ConversionError("No valid RMBench episodes found after validation.")

    reset_output_dir(output_dir, args.overwrite)
    extractor = LatentAndTextExtractor(model_path=model_path, device=args.device)
    empty_emb_path = output_dir / "empty_emb.pt"
    torch.save(extractor.encode_text(""), empty_emb_path)

    task_to_index: dict[str, int] = {}
    coarse_task_to_index: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    debug_saved_per_task: dict[str, int] = {}

    global_frame_index = 0
    total_latent_files = 0

    for episode_index, validated in enumerate(valid_episodes):
        candidate = validated.candidate
        task_index = task_to_index.setdefault(validated.episode_prompt, len(task_to_index))
        coarse_task_index = coarse_task_to_index.setdefault(candidate.task_name, len(coarse_task_to_index))

        frames_by_camera, action = load_episode_content(validated)
        parquet_rows = build_parquet_rows(
            action=action,
            episode_index=episode_index,
            global_start_index=global_frame_index,
            task_index=task_index,
            coarse_task_index=coarse_task_index,
            task_name=candidate.task_name,
            episode_prompt=validated.episode_prompt,
            source_fps=validated.source_fps,
        )
        write_episode_parquet(parquet_file_path(output_dir, episode_index), parquet_rows)

        episode_stats = compute_episode_stats(parquet_rows, DATASET_FEATURES)
        all_episode_stats.append(episode_stats)
        episode_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": serialize_stats(episode_stats),
            }
        )

        action_config: list[dict[str, Any]] = []
        debug_frames_for_task: dict[str, np.ndarray] = {}
        for segment in validated.segments:
            frame_ids = sample_frame_ids(
                segment.start_frame,
                segment.end_frame,
                source_fps=validated.source_fps,
                target_fps=args.target_fps,
            )
            if len(frame_ids) < 2:
                raise ConversionError(
                    f"Episode {candidate.task_name}/episode{candidate.episode_id} segment "
                    f"({segment.start_frame}, {segment.end_frame}) sampled fewer than 2 frames."
                )
            segment_frame_stride = max(frame_ids[1] - frame_ids[0], 1)
            action_config.append(
                {
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "action_text": segment.action_text,
                    "segment_source": segment.source,
                }
            )
            text_emb = extractor.encode_text(segment.action_text)
            for camera_key, frames in frames_by_camera.items():
                sampled_frames = frames[frame_ids]
                resize_height, resize_width = CAMERA_RESIZE[camera_key]
                resized_frames = resize_rgb_frames(sampled_frames, resize_height, resize_width)
                latent, latent_num_frames, latent_height, latent_width = extractor.encode_video(resized_frames)
                payload = {
                    "latent": latent,
                    "latent_num_frames": latent_num_frames,
                    "latent_height": latent_height,
                    "latent_width": latent_width,
                    "video_num_frames": len(frame_ids),
                    "video_height": resize_height,
                    "video_width": resize_width,
                    "text_emb": text_emb,
                    "text": segment.action_text,
                    "frame_ids": frame_ids,
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "fps": float(validated.source_fps / segment_frame_stride),
                    "ori_fps": float(validated.source_fps),
                }
                path = latent_file_path(
                    output_dir,
                    episode_index=episode_index,
                    camera_key=camera_key,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                )
                ensure_dir(path.parent)
                torch.save(payload, path)
                total_latent_files += 1
                if debug_saved_per_task.get(candidate.task_name, 0) < args.debug_image_limit:
                    debug_frames_for_task[camera_key] = resized_frames

        if debug_saved_per_task.get(candidate.task_name, 0) < args.debug_image_limit:
            save_debug_images(
                output_dir / "debug_samples",
                candidate.task_name,
                episode_index,
                debug_frames_for_task,
            )
            debug_saved_per_task[candidate.task_name] = debug_saved_per_task.get(candidate.task_name, 0) + 1

        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [validated.episode_prompt],
                "task_name": candidate.task_name,
                "source_episode_id": candidate.episode_id,
                "source_hdf5": str(candidate.hdf5_path),
                "length": int(action.shape[0]),
                "action_config": action_config,
            }
        )
        global_frame_index += int(action.shape[0])

    total_tasks = len(task_to_index)
    info = build_info(
        total_episodes=len(episodes_meta),
        total_frames=global_frame_index,
        total_tasks=total_tasks,
        fps=float(args.target_fps),
    )
    write_json(output_dir / "meta" / "info.json", info)
    write_jsonl(output_dir / "meta" / "episodes.jsonl", episodes_meta)
    write_jsonl(
        output_dir / "meta" / "tasks.jsonl",
        (
            {"task_index": task_index, "task": task}
            for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
        ),
    )
    write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", episode_stats_rows)
    write_json(output_dir / "meta" / "stats.json", serialize_stats(aggregate_stats(all_episode_stats)))
    (output_dir / "README.md").write_text(build_dataset_readme(output_dir), encoding="utf-8")

    train_command = (
        "NGPU=8 CONFIG_NAME=rmbench_train bash script/run_va_posttrain.sh "
        f"--dataset-path {output_dir} "
        f"--pretrained-model-path {model_path} "
        "--save-root /path/to/train_out"
    )
    report = {
        "rmbench_root": str(rmbench_root),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "setting": args.setting,
        "desc_type": args.desc_type,
        "segment_source": args.segment_source,
        "target_fps": float(args.target_fps),
        "source_fps_fallback": float(args.source_fps),
        "device": args.device,
        "task_count": len(task_roots),
        "valid_episode_count": len(episodes_meta),
        "skipped_episode_count": len(skipped_episodes),
        "latent_file_count": total_latent_files,
        "empty_emb_path": str(empty_emb_path),
        "train_command_example": train_command,
        "skipped_episodes": skipped_episodes,
    }
    write_json(output_dir / "conversion_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RMBench demo_clean HDF5 data into LingBot-VA trainable LeRobot + latent format."
    )
    parser.add_argument("--rmbench-root", type=Path, required=True, help="Root containing RMBench task folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--model-path", type=Path, required=True, help="LingBot-VA pretrained model path.")
    parser.add_argument("--setting", type=str, default="demo_clean", help="RMBench setting directory name.")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated task list or `all`.")
    parser.add_argument("--desc-type", type=str, default="seen", help="Instruction key inside episode JSON.")
    parser.add_argument(
        "--segment-source",
        type=str,
        default="episode_instruction",
        choices=["episode_instruction", "language_annotation"],
        help="How to build action_config segments.",
    )
    parser.add_argument("--target-fps", type=float, default=7.5, help="Target FPS used for latent extraction.")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Fallback source FPS when video/episodeN.mp4 is unavailable.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for VAE/text encoding.")
    parser.add_argument(
        "--debug-image-limit",
        type=int,
        default=1,
        help="Save up to N debug RGB samples per task under output_dir/debug_samples. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-skip-invalid",
        action="store_true",
        help="Skip invalid episodes instead of failing the entire conversion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = convert_dataset(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
