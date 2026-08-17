# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate Kimi K3 MXFP4 dequantization against a dense BF16 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import multiprocessing
import os
import platform
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import paddle

from paddleformers.quantization.hf_checkpoint import (
    HFDequantLoadTransform,
    TensorMetadata,
    _read_safetensors_header,
    resolve_hf_quantization_adapter,
)
from paddleformers.utils.safetensors import fast_safe_open


DEFAULT_FP4_MODEL_PATH = "/root/paddlejob/share-storage/gpfs/system-public/zhuxinming/survey/kimi-k3/model_weights"
DEFAULT_REFERENCE_MODEL_PATH = (
    "/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/kimik3_workspace/Models/Kimi-K3"
)
DEFAULT_LOG_FILE = Path(__file__).with_name("kimi_k3_dequant_validation.log")
INDEX_FILE = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp4-model-path", default=DEFAULT_FP4_MODEL_PATH)
    parser.add_argument("--reference-model-path", default=DEFAULT_REFERENCE_MODEL_PATH)
    parser.add_argument("--log-file", default=str(DEFAULT_LOG_FILE))
    parser.add_argument(
        "--logical-key",
        action="append",
        default=[],
        help="Logical .weight key to validate. May be specified multiple times.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of deterministic, evenly distributed weights to test when --logical-key is omitted.",
    )
    parser.add_argument("--all", action="store_true", help="Validate every comparable MXFP4 logical weight.")
    parser.add_argument(
        "--all-limit",
        type=int,
        default=0,
        help="Debug only: limit --all to the first N comparable weights; zero means no limit.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker processes used by --all.")
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="OMP/MKL threads assigned to each worker process.",
    )
    parser.add_argument("--max-failure-details", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    return parser.parse_args()


def configure_logging(log_file: str) -> logging.Logger:
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kimi_k3_dequant_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Log file: %s", path)
    return logger


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def read_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / INDEX_FILE
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(key, str) or not isinstance(file_name, str) for key, file_name in weight_map.items()
    ):
        raise ValueError(f"Invalid weight_map in {index_path}.")
    return weight_map


def get_quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    quantization_config = config.get("quantization_config")
    if quantization_config is None:
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            quantization_config = text_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("FP4 config does not define a quantization_config JSON object.")
    return quantization_config


def find_logical_keys(fp4_weight_map: dict[str, str], reference_weight_map: dict[str, str]) -> list[str]:
    logical_keys = []
    for packed_key in fp4_weight_map:
        if not packed_key.endswith(".weight_packed"):
            continue
        logical_key = packed_key[: -len("_packed")]
        if logical_key + "_scale" in fp4_weight_map and logical_key in reference_weight_map:
            logical_keys.append(logical_key)
    return sorted(logical_keys)


def select_logical_keys(candidates: list[str], requested: list[str], num_samples: int) -> list[str]:
    if requested:
        missing = sorted(set(requested) - set(candidates))
        if missing:
            raise KeyError(f"Requested logical keys are unavailable: {missing}.")
        return list(dict.fromkeys(requested))
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if num_samples >= len(candidates):
        return candidates
    if num_samples == 1:
        return [candidates[0]]

    indices = [round(index * (len(candidates) - 1) / (num_samples - 1)) for index in range(num_samples)]
    return [candidates[index] for index in dict.fromkeys(indices)]


def load_selected_metadata(
    model_path: Path,
    weight_map: dict[str, str],
    keys: list[str],
) -> dict[str, TensorMetadata]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        keys_by_file[weight_map[key]].append(key)

    selected: dict[str, TensorMetadata] = {}
    for file_name, file_keys in keys_by_file.items():
        file_path = model_path / file_name
        metadata = _read_safetensors_header(str(file_path))
        for key in file_keys:
            if key not in metadata:
                raise KeyError(f"{key!r} is declared in the index but absent from {file_path}.")
            selected[key] = metadata[key]
    return selected


def load_tensors(
    model_path: Path,
    weight_map: dict[str, str],
    keys: list[str],
) -> dict[str, np.ndarray]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        keys_by_file[weight_map[key]].append(key)

    tensors: dict[str, np.ndarray] = {}
    for file_name, file_keys in keys_by_file.items():
        file_path = model_path / file_name
        with fast_safe_open(str(file_path), framework="np") as file:
            available_keys = set(file.keys())
            for key in file_keys:
                if key not in available_keys:
                    raise KeyError(f"{key!r} is declared in the index but absent from {file_path}.")
                tensors[key] = file.get_tensor(key)
    return tensors


def build_sample_transform(
    fp4_model_path: Path,
    fp4_weight_map: dict[str, str],
    logical_keys: list[str],
) -> tuple[str, HFDequantLoadTransform]:
    config = read_json(fp4_model_path / "config.json")
    quantization_config = get_quantization_config(config)
    adapter_name, adapter = resolve_hf_quantization_adapter(quantization_config)
    physical_keys = [component for key in logical_keys for component in (key + "_packed", key + "_scale")]
    source_metadata = load_selected_metadata(fp4_model_path, fp4_weight_map, physical_keys)
    manifest = adapter.build_manifest(quantization_config, source_metadata)
    return adapter_name, HFDequantLoadTransform(manifest, paddle.bfloat16)


def compare_weight(
    logical_key: str,
    transform: HFDequantLoadTransform,
    fp4_model_path: Path,
    reference_model_path: Path,
    fp4_weight_map: dict[str, str],
    reference_weight_map: dict[str, str],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    source_keys = transform.source_keys(logical_key)
    source_arrays = load_tensors(fp4_model_path, fp4_weight_map, source_keys)
    source_tensors = {key: paddle.to_tensor(value) for key, value in source_arrays.items()}
    output = transform.apply(logical_key, source_tensors, paddle.bfloat16)

    reference_array = load_tensors(reference_model_path, reference_weight_map, [logical_key])[logical_key]
    reference = np.asarray(reference_array, dtype=np.float32)
    actual = output.astype("float32").numpy()
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch for {logical_key}: actual={actual.shape}, reference={reference.shape}.")

    difference = np.abs(actual - reference)
    mismatch_count = int(np.count_nonzero(actual != reference))
    close = bool(np.allclose(actual, reference, rtol=rtol, atol=atol, equal_nan=True))
    return {
        "logical_key": logical_key,
        "source_keys": source_keys,
        "fp4_shards": sorted({fp4_weight_map[key] for key in source_keys}),
        "reference_shard": reference_weight_map[logical_key],
        "packed_shape": tuple(source_arrays[logical_key + "_packed"].shape),
        "scale_shape": tuple(source_arrays[logical_key + "_scale"].shape),
        "reference_storage_dtype": str(reference_array.dtype),
        "output_shape": tuple(output.shape),
        "output_dtype": str(output.dtype),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "rmse": float(math.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
        "mismatch_count": mismatch_count,
        "element_count": int(actual.size),
        "actual_min": float(actual.min()),
        "actual_max": float(actual.max()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "passed": close,
    }


def build_group_tasks(
    logical_keys: list[str],
    fp4_weight_map: dict[str, str],
    reference_weight_map: dict[str, str],
    fp4_model_path: Path,
    reference_model_path: Path,
    quantization_config: dict[str, Any],
    rtol: float,
    atol: float,
    max_failure_details: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for logical_key in logical_keys:
        packed_key = logical_key + "_packed"
        scale_key = logical_key + "_scale"
        group_key = (
            fp4_weight_map[packed_key],
            fp4_weight_map[scale_key],
            reference_weight_map[logical_key],
        )
        grouped[group_key].append(logical_key)

    tasks = []
    for shard_names, group_logical_keys in sorted(grouped.items()):
        packed_shard, scale_shard, reference_shard = shard_names
        group_fp4_weight_map = {}
        group_reference_weight_map = {}
        for logical_key in group_logical_keys:
            group_fp4_weight_map[logical_key + "_packed"] = packed_shard
            group_fp4_weight_map[logical_key + "_scale"] = scale_shard
            group_reference_weight_map[logical_key] = reference_shard
        tasks.append(
            {
                "fp4_model_path": str(fp4_model_path),
                "reference_model_path": str(reference_model_path),
                "quantization_config": quantization_config,
                "logical_keys": group_logical_keys,
                "fp4_weight_map": group_fp4_weight_map,
                "reference_weight_map": group_reference_weight_map,
                "rtol": rtol,
                "atol": atol,
                "max_failure_details": max_failure_details,
                "shards": shard_names,
            }
        )
    return tasks


def validate_weight_group(task: dict[str, Any]) -> dict[str, Any]:
    started_at = time.monotonic()
    fp4_model_path = Path(task["fp4_model_path"])
    reference_model_path = Path(task["reference_model_path"])
    logical_keys = task["logical_keys"]
    fp4_weight_map = task["fp4_weight_map"]
    reference_weight_map = task["reference_weight_map"]
    physical_keys = [component for key in logical_keys for component in (key + "_packed", key + "_scale")]

    source_metadata = load_selected_metadata(fp4_model_path, fp4_weight_map, physical_keys)
    adapter_name, adapter = resolve_hf_quantization_adapter(task["quantization_config"])
    manifest = adapter.build_manifest(task["quantization_config"], source_metadata)
    transform = HFDequantLoadTransform(manifest, paddle.bfloat16)

    summary = {
        "adapter_name": adapter_name,
        "shards": task["shards"],
        "weight_count": 0,
        "element_count": 0,
        "mismatched_weight_count": 0,
        "mismatched_element_count": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "sum_squared_error": 0.0,
        "reference_storage_dtypes": set(),
        "failure_details": [],
    }

    with ExitStack() as stack:
        fp4_readers = {
            file_name: stack.enter_context(fast_safe_open(str(fp4_model_path / file_name), framework="np"))
            for file_name in sorted(set(fp4_weight_map.values()))
        }
        reference_readers = {
            file_name: stack.enter_context(fast_safe_open(str(reference_model_path / file_name), framework="np"))
            for file_name in sorted(set(reference_weight_map.values()))
        }

        for index, logical_key in enumerate(logical_keys, start=1):
            source_keys = transform.source_keys(logical_key)
            source_arrays = {key: fp4_readers[fp4_weight_map[key]].get_tensor(key) for key in source_keys}
            source_tensors = {key: paddle.to_tensor(value) for key, value in source_arrays.items()}
            output = transform.apply(logical_key, source_tensors, paddle.bfloat16)

            reference_array = reference_readers[reference_weight_map[logical_key]].get_tensor(logical_key)
            reference = np.asarray(reference_array, dtype=np.float32)
            actual = output.astype("float32").numpy()
            if actual.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for {logical_key}: actual={actual.shape}, reference={reference.shape}."
                )

            difference = np.abs(actual - reference)
            mismatch_count = int(np.count_nonzero(actual != reference))
            passed = bool(np.allclose(actual, reference, rtol=task["rtol"], atol=task["atol"], equal_nan=True))
            max_abs_error = float(difference.max())
            summary["weight_count"] += 1
            summary["element_count"] += int(actual.size)
            summary["mismatched_element_count"] += mismatch_count
            summary["max_abs_error"] = max(summary["max_abs_error"], max_abs_error)
            summary["sum_abs_error"] += float(difference.sum(dtype=np.float64))
            summary["sum_squared_error"] += float(np.square(difference, dtype=np.float64).sum())
            summary["reference_storage_dtypes"].add(str(reference_array.dtype))
            if not passed:
                summary["mismatched_weight_count"] += 1
                if len(summary["failure_details"]) < task["max_failure_details"]:
                    summary["failure_details"].append(
                        {
                            "logical_key": logical_key,
                            "shape": tuple(actual.shape),
                            "max_abs_error": max_abs_error,
                            "mismatch_count": mismatch_count,
                        }
                    )

            del source_arrays, source_tensors, output, reference_array, reference, actual, difference
            if index % 128 == 0:
                gc.collect()

    summary["reference_storage_dtypes"] = sorted(summary["reference_storage_dtypes"])
    summary["elapsed_seconds"] = time.monotonic() - started_at
    return summary


def run_full_validation(
    logger: logging.Logger,
    args: argparse.Namespace,
    candidates: list[str],
    fp4_weight_map: dict[str, str],
    reference_weight_map: dict[str, str],
    fp4_model_path: Path,
    reference_model_path: Path,
) -> bool:
    if args.logical_key:
        raise ValueError("--logical-key cannot be combined with --all.")
    if args.workers <= 0 or args.threads_per_worker <= 0:
        raise ValueError("workers and threads-per-worker must be positive.")
    if args.max_failure_details < 0:
        raise ValueError("max-failure-details must not be negative.")
    if args.all_limit < 0:
        raise ValueError("all-limit must not be negative.")
    if args.all_limit:
        logger.warning("Debug all-limit is active: validating only the first %d weights", args.all_limit)
        candidates = candidates[: args.all_limit]

    fp4_config = read_json(fp4_model_path / "config.json")
    quantization_config = get_quantization_config(fp4_config)
    tasks = build_group_tasks(
        logical_keys=candidates,
        fp4_weight_map=fp4_weight_map,
        reference_weight_map=reference_weight_map,
        fp4_model_path=fp4_model_path,
        reference_model_path=reference_model_path,
        quantization_config=quantization_config,
        rtol=args.rtol,
        atol=args.atol,
        max_failure_details=args.max_failure_details,
    )
    logger.info(
        "Full validation plan: %d weights in %d shard groups, workers=%d, threads-per-worker=%d",
        len(candidates),
        len(tasks),
        args.workers,
        args.threads_per_worker,
    )

    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(args.threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads_per_worker)
    aggregate = {
        "weight_count": 0,
        "element_count": 0,
        "mismatched_weight_count": 0,
        "mismatched_element_count": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "sum_squared_error": 0.0,
        "failure_details": [],
    }
    started_at = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        future_to_task = {executor.submit(validate_weight_group, task): task for task in tasks}
        for completed, future in enumerate(as_completed(future_to_task), start=1):
            result = future.result()
            for key in (
                "weight_count",
                "element_count",
                "mismatched_weight_count",
                "mismatched_element_count",
                "sum_abs_error",
                "sum_squared_error",
            ):
                aggregate[key] += result[key]
            aggregate["max_abs_error"] = max(aggregate["max_abs_error"], result["max_abs_error"])
            remaining_failure_slots = max(0, args.max_failure_details - len(aggregate["failure_details"]))
            aggregate["failure_details"].extend(result["failure_details"][:remaining_failure_slots])
            elapsed = time.monotonic() - started_at
            logger.info(
                "Shard group %d/%d complete: shards=%s, weights=%d, elements=%d, "
                "mismatched_weights=%d, max_abs_error=%g, group_seconds=%.3f, total_seconds=%.3f",
                completed,
                len(tasks),
                result["shards"],
                result["weight_count"],
                result["element_count"],
                result["mismatched_weight_count"],
                result["max_abs_error"],
                result["elapsed_seconds"],
                elapsed,
            )

    if aggregate["element_count"]:
        aggregate["mean_abs_error"] = aggregate["sum_abs_error"] / aggregate["element_count"]
        aggregate["rmse"] = math.sqrt(aggregate["sum_squared_error"] / aggregate["element_count"])
    else:
        aggregate["mean_abs_error"] = 0.0
        aggregate["rmse"] = 0.0
    aggregate["elapsed_seconds"] = time.monotonic() - started_at
    aggregate["passed"] = aggregate["mismatched_weight_count"] == 0
    logger.info("Full validation summary: %s", json.dumps(aggregate, sort_keys=True))
    return aggregate["passed"]


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.log_file)
    fp4_model_path = Path(args.fp4_model_path).expanduser().resolve()
    reference_model_path = Path(args.reference_model_path).expanduser().resolve()
    started_at = time.monotonic()

    try:
        logger.info("Validation started")
        logger.info("Python: %s", sys.version.replace("\n", " "))
        logger.info("Platform: %s", platform.platform())
        logger.info("Paddle: %s", paddle.__version__)
        logger.info("Paddle device: %s", paddle.device.get_device())
        logger.info("FP4 model path: %s", fp4_model_path)
        logger.info("Reference model path: %s", reference_model_path)
        logger.info("Tolerance: rtol=%g, atol=%g", args.rtol, args.atol)

        for model_path in (fp4_model_path, reference_model_path):
            if not model_path.is_dir():
                raise FileNotFoundError(f"Model directory does not exist: {model_path}.")
            for required_name in ("config.json", INDEX_FILE):
                required_path = model_path / required_name
                if not required_path.is_file():
                    raise FileNotFoundError(f"Required file does not exist: {required_path}.")

        fp4_weight_map = read_weight_map(fp4_model_path)
        reference_weight_map = read_weight_map(reference_model_path)
        candidates = find_logical_keys(fp4_weight_map, reference_weight_map)
        if not candidates:
            raise ValueError("No matching MXFP4 and dense reference weights were found.")
        logger.info("FP4 index tensors: %d", len(fp4_weight_map))
        logger.info("Reference index tensors: %d", len(reference_weight_map))
        logger.info("Comparable MXFP4 logical weights: %d", len(candidates))

        if args.all:
            passed = run_full_validation(
                logger=logger,
                args=args,
                candidates=candidates,
                fp4_weight_map=fp4_weight_map,
                reference_weight_map=reference_weight_map,
                fp4_model_path=fp4_model_path,
                reference_model_path=reference_model_path,
            )
            elapsed = time.monotonic() - started_at
            if not passed:
                logger.error("Validation FAILED after %.3f seconds", elapsed)
                return 1
            logger.info("Validation PASSED after %.3f seconds", elapsed)
            return 0

        logical_keys = select_logical_keys(candidates, args.logical_key, args.num_samples)
        logger.info("Selected logical weights (%d): %s", len(logical_keys), logical_keys)

        adapter_name, transform = build_sample_transform(fp4_model_path, fp4_weight_map, logical_keys)
        logger.info("Resolved quantization adapter: %s", adapter_name)

        failures = []
        for index, logical_key in enumerate(logical_keys, start=1):
            weight_started_at = time.monotonic()
            logger.info("[%d/%d] Validating %s", index, len(logical_keys), logical_key)
            result = compare_weight(
                logical_key=logical_key,
                transform=transform,
                fp4_model_path=fp4_model_path,
                reference_model_path=reference_model_path,
                fp4_weight_map=fp4_weight_map,
                reference_weight_map=reference_weight_map,
                rtol=args.rtol,
                atol=args.atol,
            )
            result["elapsed_seconds"] = time.monotonic() - weight_started_at
            logger.info("[%d/%d] Result: %s", index, len(logical_keys), json.dumps(result, sort_keys=True))
            if not result["passed"]:
                failures.append(result)
            del result
            gc.collect()

        elapsed = time.monotonic() - started_at
        if failures:
            logger.error(
                "Validation FAILED: %d/%d weights mismatched in %.3f seconds",
                len(failures),
                len(logical_keys),
                elapsed,
            )
            return 1
        logger.info(
            "Validation PASSED: %d/%d weights matched in %.3f seconds",
            len(logical_keys),
            len(logical_keys),
            elapsed,
        )
        return 0
    except Exception:
        logger.exception("Validation aborted after %.3f seconds", time.monotonic() - started_at)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
