#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path("/kpfs-intern/chenyandu/data/rmbench-lingbot")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "norm_stats" / "rmbench_norm_stat.json"
DEFAULT_QUANTILES = [0.01, 0.99]

# RMBench actions are stored as compact 16D robotwin actions:
# [left_pose(7), left_gripper, right_pose(7), right_gripper]
USED_ACTION_CHANNEL_IDS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
RAW_ACTION_DIM = len(USED_ACTION_CHANNEL_IDS)
MODEL_ACTION_DIM = 30


class RunningQuantileStats:
    """Streaming stats with histogram-based quantile estimation."""

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins
        self._quantile_list = quantile_list or DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch).reshape(-1, batch.shape[-1])
        batch = batch.astype(np.result_type(batch.dtype, np.float32), copy=False)
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, key in enumerate(self._quantile_keys):
            stats[key] = quantile_results[i]
        return stats

    def _adjust_histograms(self) -> None:
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1)
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_values.append(self._compute_single_quantile(hist, edges, target_count))
            results.append(np.array(q_values))
        return results

    @staticmethod
    def _compute_single_quantile(hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before
        if count_in_bin == 0:
            return edges[idx]

        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RMBench action q01/q99 stats for LingBot-VA training."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root of the converted RMBench LingBot-VA dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the padded 30D normalization stats JSON.",
    )
    parser.add_argument(
        "--num-quantile-bins",
        type=int,
        default=5000,
        help="Histogram bins for RunningQuantileStats. Matches lerobot's default.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on parquet files, useful for quick smoke tests.",
    )
    return parser.parse_args()


def list_parquet_files(dataset_root: Path) -> list[Path]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}.")
    return parquet_files


def load_action_batch(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != RAW_ACTION_DIM:
        raise ValueError(
            f"{parquet_path} has invalid action shape {actions.shape}; expected [N, {RAW_ACTION_DIM}]."
        )
    return actions


def pad_to_model_action_dim(raw_values: np.ndarray, fill_value: float = 0.0) -> list[float]:
    padded = np.full(MODEL_ACTION_DIM, fill_value, dtype=np.float32)
    padded[USED_ACTION_CHANNEL_IDS] = raw_values.astype(np.float32, copy=False)
    return padded.tolist()


def to_serializable_stats(stats: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "count": int(np.asarray(stats["count"]).reshape(-1)[0]),
        "min": stats["min"].astype(np.float32).tolist(),
        "max": stats["max"].astype(np.float32).tolist(),
        "mean": stats["mean"].astype(np.float32).tolist(),
        "std": stats["std"].astype(np.float32).tolist(),
        "q01": stats["q01"].astype(np.float32).tolist(),
        "q99": stats["q99"].astype(np.float32).tolist(),
    }


def build_output_payload(
    dataset_root: Path,
    parquet_files: list[Path],
    raw_stats: dict[str, np.ndarray],
    num_quantile_bins: int,
) -> dict[str, Any]:
    raw_stats_json = to_serializable_stats(raw_stats)
    padded_norm_stat = {
        "q01": pad_to_model_action_dim(raw_stats["q01"]),
        "q99": pad_to_model_action_dim(raw_stats["q99"]),
    }
    return {
        "dataset_root": str(dataset_root),
        "parquet_file_count": len(parquet_files),
        "raw_action_dim": RAW_ACTION_DIM,
        "model_action_dim": MODEL_ACTION_DIM,
        "used_action_channel_ids": USED_ACTION_CHANNEL_IDS,
        "quantile_method": "histogram_running_quantile",
        "num_quantile_bins": num_quantile_bins,
        "raw_stats": raw_stats_json,
        "norm_stat": padded_norm_stat,
        "q01": padded_norm_stat["q01"],
        "q99": padded_norm_stat["q99"],
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output_path.resolve()

    parquet_files = list_parquet_files(dataset_root)
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]
        if not parquet_files:
            raise ValueError("--max-files resulted in an empty parquet file list.")

    running_stats = RunningQuantileStats(
        quantile_list=DEFAULT_QUANTILES,
        num_quantile_bins=args.num_quantile_bins,
    )

    total_frames = 0
    for parquet_path in parquet_files:
        actions = load_action_batch(parquet_path)
        running_stats.update(actions)
        total_frames += int(actions.shape[0])

    raw_stats = running_stats.get_statistics()
    payload = build_output_payload(dataset_root, parquet_files, raw_stats, args.num_quantile_bins)
    payload["total_frames"] = total_frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({"output_path": str(output_path), "parquet_file_count": len(parquet_files), "total_frames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
