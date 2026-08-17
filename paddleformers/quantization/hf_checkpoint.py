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

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

import paddle

from .checkpoint_dequant import CheckpointDequantSpec, checkpoint_dequantize


_SUPPORTED_LOAD_MODES = {"auto", "off", "dequantize_bf16"}
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def normalize_hf_quantized_load_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError(f"hf_quantized_load_mode must be a string, got {type(mode).__name__}.")
    normalized = mode.strip().lower()
    if normalized not in _SUPPORTED_LOAD_MODES:
        raise ValueError(
            f"Invalid hf_quantized_load_mode {normalized!r}; " f"expected one of {sorted(_SUPPORTED_LOAD_MODES)}."
        )
    return normalized


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    file_name: str | None = None


try:
    LoadTensorMetadata = paddle.distributed.LoadTensorMetadata
except AttributeError:
    # Keep importing PaddleFormers with Paddle versions before this public type.
    @dataclass(frozen=True)
    class LoadTensorMetadata:
        global_shape: tuple[int, ...]
        dtype: str


@dataclass
class HFQuantizedWeightSpec:
    logical_name: str
    logical_shape: tuple[int, ...]
    components: dict[str, str]
    quant_method: str
    bits: int | None = None
    value_format: str | None = None
    scale_format: str | None = None
    block_axes: tuple[int, ...] | None = None
    block_shape: tuple[int, ...] | None = None
    storage_layout: str | None = None
    scale_layout: str | None = None


@dataclass
class HFQuantizationManifest:
    weights: dict[str, HFQuantizedWeightSpec]

    def __post_init__(self) -> None:
        for logical_key, spec in self.weights.items():
            _validate_weight_spec(logical_key, spec)


class HFQuantizationAdapter(Protocol):
    def matches(self, raw_quantization_config: dict[str, Any]) -> bool:
        ...

    def build_manifest(
        self,
        raw_quantization_config: dict[str, Any],
        source_keys: dict[str, TensorMetadata],
    ) -> HFQuantizationManifest:
        ...


_HF_QUANTIZATION_ADAPTERS: dict[str, HFQuantizationAdapter] = {}


def register_hf_quantization_adapter(name: str, adapter: HFQuantizationAdapter) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("HF quantization adapter name must not be empty.")
    name = name.strip().lower()
    if name in _HF_QUANTIZATION_ADAPTERS:
        raise ValueError(f"HF quantization adapter {name!r} is already registered.")
    if not callable(getattr(adapter, "matches", None)) or not callable(getattr(adapter, "build_manifest", None)):
        raise TypeError("HF quantization adapter must provide matches() and build_manifest() methods.")
    _HF_QUANTIZATION_ADAPTERS[name] = adapter


def resolve_hf_quantization_adapter(
    raw_quantization_config: dict[str, Any],
) -> tuple[str, HFQuantizationAdapter]:
    matched = [
        (name, adapter)
        for name, adapter in _HF_QUANTIZATION_ADAPTERS.items()
        if adapter.matches(raw_quantization_config)
    ]
    if not matched:
        registered = sorted(_HF_QUANTIZATION_ADAPTERS)
        raise ValueError(
            "Unsupported HF quantization_config " f"{raw_quantization_config!r}; registered adapters: {registered}."
        )
    if len(matched) != 1:
        names = [name for name, _ in matched]
        raise ValueError(f"HF quantization_config matches multiple adapters: {names}.")
    return matched[0]


def _validate_weight_spec(
    logical_key: str,
    spec: HFQuantizedWeightSpec,
) -> None:
    if spec.logical_name != logical_key:
        raise ValueError(f"Manifest key {logical_key!r} does not match weight logical_name {spec.logical_name!r}.")
    if not spec.logical_shape or any(not isinstance(dim, int) or dim <= 0 for dim in spec.logical_shape):
        raise ValueError(f"Invalid logical shape for {logical_key!r}: {spec.logical_shape}.")
    if "qweight" not in spec.components:
        raise ValueError(f"Quantized weight {logical_key!r} does not define a qweight component.")
    if any(not role or not source_key for role, source_key in spec.components.items()):
        raise ValueError(f"Quantized weight {logical_key!r} contains an empty component role or source key.")
    if not spec.quant_method:
        raise ValueError(f"Quantized weight {logical_key!r} does not define a quantization method.")
    if (spec.block_axes is None) != (spec.block_shape is None):
        raise ValueError(f"Quantized weight {logical_key!r} must define block_axes and block_shape together.")
    if spec.block_axes is not None:
        if len(spec.block_axes) != len(spec.block_shape) or len(set(spec.block_axes)) != len(spec.block_axes):
            raise ValueError(f"Invalid block geometry for {logical_key!r}.")
        if any(axis < 0 or axis >= len(spec.logical_shape) for axis in spec.block_axes):
            raise ValueError(f"Block axes for {logical_key!r} are outside logical shape {spec.logical_shape}.")
        if any(size <= 0 for size in spec.block_shape):
            raise ValueError(f"Block sizes for {logical_key!r} must be positive.")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path!r}.")
    return value


def _read_safetensors_header(file_path: str) -> dict[str, TensorMetadata]:
    file_size = os.path.getsize(file_path)
    with open(file_path, "rb") as file:
        raw_header_size = file.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"Invalid safetensors file {file_path!r}: missing the 8-byte header size.")
        header_size = int.from_bytes(raw_header_size, byteorder="little", signed=False)
        if header_size <= 0 or header_size > file_size - 8:
            raise ValueError(f"Invalid safetensors header size {header_size} for {file_path!r} with size {file_size}.")
        raw_header = file.read(header_size)

    try:
        header = json.loads(raw_header)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors JSON header in {file_path!r}.") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {file_path!r}: expected a JSON object.")

    data_size = file_size - 8 - header_size
    metadata: dict[str, TensorMetadata] = {}
    for key, item in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Invalid metadata for tensor {key!r} in {file_path!r}.")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise ValueError(f"Incomplete metadata for tensor {key!r} in {file_path!r}.")
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"Invalid shape for tensor {key!r} in {file_path!r}: {shape}.")
        if len(offsets) != 2 or any(not isinstance(offset, int) for offset in offsets):
            raise ValueError(f"Invalid data offsets for tensor {key!r} in {file_path!r}: {offsets}.")
        start, end = offsets
        if start < 0 or start > end or end > data_size:
            raise ValueError(f"Out-of-range data offsets for tensor {key!r} in {file_path!r}: {offsets}.")
        metadata[key] = TensorMetadata(
            shape=tuple(shape),
            dtype=dtype,
            file_name=os.path.basename(file_path),
        )
    return metadata


def _find_safetensors_index(checkpoint_path: str) -> str | None:
    standard_path = os.path.join(checkpoint_path, _SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(standard_path):
        return standard_path
    candidates = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors.index.json")))
    if len(candidates) > 1:
        raise ValueError(f"Found multiple safetensors index files in {checkpoint_path!r}: {candidates}.")
    return candidates[0] if candidates else None


def _gather_tensor_metadata(local_metadata: dict[str, TensorMetadata]) -> dict[str, TensorMetadata]:
    if not paddle.distributed.is_initialized() or paddle.distributed.get_world_size() <= 1:
        return local_metadata

    gathered: list[dict[str, TensorMetadata]] = []
    paddle.distributed.all_gather_object(gathered, local_metadata)
    merged: dict[str, TensorMetadata] = {}
    for rank_metadata in gathered:
        for key, item in rank_metadata.items():
            previous = merged.get(key)
            if previous is not None and previous != item:
                raise ValueError(f"Conflicting safetensors metadata for {key!r}: {previous!r} versus {item!r}.")
            merged[key] = item
    return merged


def read_hf_safetensors_metadata(checkpoint_path: str) -> dict[str, TensorMetadata]:
    index_path = _find_safetensors_index(checkpoint_path)
    weight_map: dict[str, str] | None = None
    if index_path is not None:
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or any(
            not isinstance(key, str) or not isinstance(file_name, str) for key, file_name in weight_map.items()
        ):
            raise ValueError(f"Invalid weight_map in {index_path!r}.")
        candidate_files = sorted(set(weight_map.values()))
    else:
        candidate_files = [
            os.path.basename(path) for path in sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
        ]

    local_metadata: dict[str, TensorMetadata] = {}
    for file_name in candidate_files:
        file_path = os.path.join(checkpoint_path, file_name)
        if not os.path.isfile(file_path):
            continue
        for key, item in _read_safetensors_header(file_path).items():
            if weight_map is not None and weight_map.get(key) != file_name:
                continue
            previous = local_metadata.get(key)
            if previous is not None and previous != item:
                raise ValueError(f"Tensor {key!r} occurs in multiple local safetensors files.")
            local_metadata[key] = item

    metadata = _gather_tensor_metadata(local_metadata)
    if weight_map is not None:
        missing = set(weight_map) - set(metadata)
        if missing:
            sample = sorted(missing)[:10]
            raise FileNotFoundError(
                f"No rank can read metadata for {len(missing)} tensors declared by {index_path!r}; "
                f"first missing keys: {sample}."
            )
    if not metadata:
        raise FileNotFoundError(f"No readable safetensors weights were found in {checkpoint_path!r}.")
    return metadata


def _storage_format(metadata: TensorMetadata) -> str:
    return metadata.dtype.split(".")[-1].upper()


def _scale_format(metadata: TensorMetadata) -> str:
    storage_format = _storage_format(metadata)
    if storage_format in {"F8_E8M0", "FLOAT8_E8M0FNU"}:
        return "ue8m0"
    # compressed-tensors stores MXFP4 E8M0 scales as raw uint8 bytes.
    if storage_format in {"U8", "UINT8", "I8", "INT8"}:
        return "ue8m0"
    if storage_format in {"F16", "FLOAT16"}:
        return "fp16"
    if storage_format in {"F32", "FLOAT32"}:
        return "fp32"
    if storage_format in {"BF16", "BFLOAT16", "U16"}:
        return "bf16"
    raise ValueError(f"Unsupported checkpoint scale storage format {storage_format!r}.")


def _expected_scale_shape(
    logical_shape: tuple[int, ...],
    block_axes: tuple[int, ...],
    block_shape: tuple[int, ...],
) -> tuple[int, ...]:
    axis_to_block = dict(zip(block_axes, block_shape))
    return tuple(
        math.ceil(dim / axis_to_block[axis]) if axis in axis_to_block else dim
        for axis, dim in enumerate(logical_shape)
    )


class FineGrainedFP8HFQuantizationAdapter:
    def matches(self, raw_quantization_config: dict[str, Any]) -> bool:
        method = str(raw_quantization_config.get("quant_method", "")).strip().lower()
        return method in {"fp8", "finegrained_fp8", "mxfp8"}

    def build_manifest(
        self,
        raw_quantization_config: dict[str, Any],
        source_keys: dict[str, TensorMetadata],
    ) -> HFQuantizationManifest:
        raw_block_shape = raw_quantization_config.get("weight_block_size")
        if not isinstance(raw_block_shape, (list, tuple)) or not raw_block_shape:
            raise ValueError("Fine-grained FP8 quantization_config must define weight_block_size.")
        block_shape = tuple(raw_block_shape)
        if any(not isinstance(size, int) or size <= 0 for size in block_shape):
            raise ValueError(f"Invalid weight_block_size: {raw_block_shape!r}.")

        weights: dict[str, HFQuantizedWeightSpec] = {}
        for weight_key in sorted(source_keys):
            if not weight_key.endswith(".weight"):
                continue
            scale_key = weight_key[: -len(".weight")] + ".scale"
            if scale_key not in source_keys:
                continue

            weight_metadata = source_keys[weight_key]
            scale_metadata = source_keys[scale_key]
            weight_shape = weight_metadata.shape
            scale_shape = scale_metadata.shape
            weight_format = _storage_format(weight_metadata)
            scale_format = _scale_format(scale_metadata)

            if weight_format in {"F8_E4M3", "FLOAT8_E4M3FN"}:
                if len(weight_shape) < len(block_shape):
                    raise ValueError(
                        f"Weight {weight_key!r} has rank {len(weight_shape)}, "
                        f"smaller than block rank {len(block_shape)}."
                    )
                block_axes = tuple(range(len(weight_shape) - len(block_shape), len(weight_shape)))
                expected_scale_shape = _expected_scale_shape(weight_shape, block_axes, block_shape)
                if scale_shape != expected_scale_shape:
                    raise ValueError(
                        f"Invalid FP8 scale shape for {weight_key!r}: "
                        f"expected {expected_scale_shape}, got {scale_shape}."
                    )
                spec = HFQuantizedWeightSpec(
                    logical_name=weight_key,
                    logical_shape=weight_shape,
                    components={"qweight": weight_key, "scale": scale_key},
                    quant_method="fp8_block",
                    bits=8,
                    value_format="e4m3",
                    scale_format=scale_format,
                    block_axes=block_axes,
                    block_shape=block_shape,
                    storage_layout="plain",
                    scale_layout="block_grid",
                )
            elif weight_format in {"I8", "INT8"}:
                if not weight_shape or weight_shape[-1] <= 0:
                    raise ValueError(f"Invalid packed FP4 weight shape for {weight_key!r}: {weight_shape}.")
                logical_shape = (*weight_shape[:-1], weight_shape[-1] * 2)
                if len(scale_shape) != len(logical_shape) or scale_shape[:-1] != logical_shape[:-1]:
                    raise ValueError(
                        f"Invalid MXFP4 scale shape for {weight_key!r}: weight={weight_shape}, scale={scale_shape}."
                    )
                if scale_shape[-1] <= 0 or logical_shape[-1] % scale_shape[-1] != 0:
                    raise ValueError(
                        f"Cannot infer an integral MXFP4 group size for {weight_key!r}: "
                        f"logical={logical_shape}, scale={scale_shape}."
                    )
                group_size = logical_shape[-1] // scale_shape[-1]
                spec = HFQuantizedWeightSpec(
                    logical_name=weight_key,
                    logical_shape=logical_shape,
                    components={"qweight": weight_key, "scale": scale_key},
                    quant_method="mxfp4_group",
                    bits=4,
                    value_format="e2m1",
                    scale_format=scale_format,
                    block_axes=(len(logical_shape) - 1,),
                    block_shape=(group_size,),
                    storage_layout="two_nibbles_last_axis_low_high",
                    scale_layout="row_group_grid",
                )
            else:
                continue
            weights[weight_key] = spec

        if not weights:
            raise ValueError(
                "The HF checkpoint declares fine-grained FP8 quantization, but no supported "
                "F8_E4M3/I8 weight and scale pairs were found."
            )
        return HFQuantizationManifest(weights=weights)


class CompressedTensorsMXFP4HFQuantizationAdapter:
    """Build a manifest for compressed-tensors MXFP4 checkpoints.

    Kimi K3 uses the compressed-tensors on-disk representation: an FP4
    weight is packed two values per uint8 and its per-group E8M0 scales are
    stored as uint8 tensors.  The quantization configuration can be nested
    under ``text_config`` in multimodal model configs.
    """

    _SUPPORTED_FORMATS = {"mxfp4-pack-quantized", "mxfp4_pack_quantized"}

    def matches(self, raw_quantization_config: dict[str, Any]) -> bool:
        method = str(raw_quantization_config.get("quant_method", "")).strip().lower()
        if method not in {"compressed-tensors", "compressed_tensors"}:
            return False
        if str(raw_quantization_config.get("format", "")).strip().lower() in self._SUPPORTED_FORMATS:
            return True
        config_groups = raw_quantization_config.get("config_groups")
        if not isinstance(config_groups, dict):
            return False
        return any(
            isinstance(group, dict) and str(group.get("format", "")).strip().lower() in self._SUPPORTED_FORMATS
            for group in config_groups.values()
        )

    def _get_mxfp4_group(self, raw_quantization_config: dict[str, Any]) -> dict[str, Any]:
        config_groups = raw_quantization_config.get("config_groups")
        if not isinstance(config_groups, dict):
            raise ValueError("compressed-tensors MXFP4 config must define config_groups.")
        groups = [
            group
            for group in config_groups.values()
            if isinstance(group, dict) and str(group.get("format", "")).strip().lower() in self._SUPPORTED_FORMATS
        ]
        if len(groups) != 1:
            raise ValueError("Expected exactly one compressed-tensors MXFP4 config group, " f"found {len(groups)}.")
        return groups[0]

    def build_manifest(
        self,
        raw_quantization_config: dict[str, Any],
        source_keys: dict[str, TensorMetadata],
    ) -> HFQuantizationManifest:
        group = self._get_mxfp4_group(raw_quantization_config)
        weights_config = group.get("weights")
        if not isinstance(weights_config, dict):
            raise ValueError("compressed-tensors MXFP4 config group must define weights.")

        bits = weights_config.get("num_bits", 4)
        group_size = weights_config.get("group_size", 32)
        strategy = str(weights_config.get("strategy", "group")).strip().lower()
        symmetric = weights_config.get("symmetric", True)
        if bits != 4 or group_size != 32 or strategy != "group":
            raise ValueError(
                "Only compressed-tensors MXFP4 group quantization with num_bits=4, "
                f"group_size=32, strategy='group' is supported; got bits={bits}, "
                f"group_size={group_size}, strategy={strategy!r}."
            )
        if symmetric is not True:
            raise ValueError("Compressed-tensors MXFP4 dequantization currently requires symmetric weights.")

        weights: dict[str, HFQuantizedWeightSpec] = {}
        for packed_key in sorted(source_keys):
            if not packed_key.endswith(".weight_packed"):
                continue
            scale_key = packed_key[: -len("_packed")] + "_scale"
            if scale_key not in source_keys:
                continue

            packed_metadata = source_keys[packed_key]
            scale_metadata = source_keys[scale_key]
            packed_shape = packed_metadata.shape
            scale_shape = scale_metadata.shape
            packed_format = _storage_format(packed_metadata)
            if packed_format not in {"U8", "UINT8", "I8", "INT8"}:
                raise ValueError(f"Unsupported MXFP4 packed storage format for {packed_key!r}: {packed_format}.")
            if not packed_shape or packed_shape[-1] <= 0:
                raise ValueError(f"Invalid packed MXFP4 shape for {packed_key!r}: {packed_shape}.")

            logical_shape = (*packed_shape[:-1], packed_shape[-1] * 2)
            block_axes = (len(logical_shape) - 1,)
            block_shape = (group_size,)
            expected_scale_shape = _expected_scale_shape(logical_shape, block_axes, block_shape)
            if scale_shape != expected_scale_shape:
                raise ValueError(
                    f"Invalid MXFP4 scale shape for {packed_key!r}: "
                    f"expected {expected_scale_shape}, got {scale_shape}."
                )

            logical_key = packed_key[: -len("_packed")]
            weights[logical_key] = HFQuantizedWeightSpec(
                logical_name=logical_key,
                logical_shape=logical_shape,
                components={"qweight": packed_key, "scale": scale_key},
                quant_method="mxfp4_group",
                bits=4,
                value_format="e2m1",
                scale_format=_scale_format(scale_metadata),
                block_axes=block_axes,
                block_shape=block_shape,
                storage_layout="two_nibbles_last_axis_low_high",
                scale_layout="row_group_grid",
            )

        if not weights:
            raise ValueError(
                "The HF checkpoint declares compressed-tensors MXFP4 quantization, but no "
                "weight_packed/weight_scale pairs were found."
            )
        return HFQuantizationManifest(weights=weights)


def _to_dequant_spec(spec: HFQuantizedWeightSpec, method: str) -> CheckpointDequantSpec:
    return CheckpointDequantSpec(
        method=method,
        bits=spec.bits,
        value_format=spec.value_format,
        scale_format=spec.scale_format,
        block_axes=spec.block_axes,
        block_shape=spec.block_shape,
        storage_layout=spec.storage_layout,
        scale_layout=spec.scale_layout,
    )


class HFDequantLoadTransform:
    def __init__(self, manifest: HFQuantizationManifest, output_dtype: paddle.dtype):
        self.manifest = manifest
        self.output_dtype = output_dtype

    def logical_metadata(self) -> dict[str, LoadTensorMetadata]:
        dtype = str(self.output_dtype).split(".")[-1]
        return {
            logical_key: LoadTensorMetadata(global_shape=spec.logical_shape, dtype=dtype)
            for logical_key, spec in self.manifest.weights.items()
        }

    def source_keys(self, logical_key: str) -> list[str]:
        try:
            spec = self.manifest.weights[logical_key]
        except KeyError as exc:
            raise KeyError(f"Logical weight {logical_key!r} is not managed by this load transform.") from exc
        return list(dict.fromkeys(spec.components.values()))

    def apply(
        self,
        logical_key: str,
        source_tensors: dict[str, paddle.Tensor],
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        try:
            spec = self.manifest.weights[logical_key]
        except KeyError as exc:
            raise KeyError(f"Logical weight {logical_key!r} is not managed by this load transform.") from exc

        missing = set(spec.components.values()) - set(source_tensors)
        if missing:
            raise KeyError(f"Missing components for {logical_key!r}: {sorted(missing)}.")
        components = {role: source_tensors[source_key] for role, source_key in spec.components.items()}
        method = spec.quant_method
        output = checkpoint_dequantize(
            method=method,
            components=components,
            spec=_to_dequant_spec(spec, method),
            output_dtype=output_dtype,
        )
        if tuple(output.shape) != spec.logical_shape:
            raise ValueError(
                f"Invalid dequantized shape for {logical_key!r}: expected {spec.logical_shape}, "
                f"got {tuple(output.shape)}."
            )
        return output


def build_hf_dequant_load_transform(
    checkpoint_path: str,
    mode: str,
) -> HFDequantLoadTransform | None:
    mode = normalize_hf_quantized_load_mode(mode)
    if mode == "off":
        return None

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.isfile(config_path):
        if mode == "auto":
            return None
        raise FileNotFoundError(f"HF quantized checkpoint config was not found at {config_path!r}.")
    raw_config = _read_json(config_path)
    raw_quantization_config = raw_config.get("quantization_config")
    if raw_quantization_config is None:
        text_config = raw_config.get("text_config")
        if isinstance(text_config, dict):
            raw_quantization_config = text_config.get("quantization_config")
    if raw_quantization_config is None:
        if mode == "auto":
            return None
        raise ValueError(f"HF config {config_path!r} does not define quantization_config.")
    if not isinstance(raw_quantization_config, dict):
        raise ValueError(f"quantization_config in {config_path!r} must be a JSON object.")

    _, adapter = resolve_hf_quantization_adapter(raw_quantization_config)
    source_metadata = read_hf_safetensors_metadata(checkpoint_path)
    manifest = adapter.build_manifest(raw_quantization_config, source_metadata)

    return HFDequantLoadTransform(manifest=manifest, output_dtype=paddle.bfloat16)


register_hf_quantization_adapter("fine_grained_fp8", FineGrainedFP8HFQuantizationAdapter())
register_hf_quantization_adapter("compressed_tensors_mxfp4", CompressedTensorsMXFP4HFQuantizationAdapter())
