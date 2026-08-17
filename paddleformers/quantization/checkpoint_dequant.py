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

import math
from dataclasses import dataclass
from typing import Protocol

import paddle


@dataclass(frozen=True)
class CheckpointDequantSpec:
    method: str
    bits: int | None = None
    value_format: str | None = None
    scale_format: str | None = None
    block_axes: tuple[int, ...] | None = None
    block_shape: tuple[int, ...] | None = None
    storage_layout: str | None = None
    scale_layout: str | None = None


class CheckpointDequantizer(Protocol):
    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        spec: CheckpointDequantSpec,
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        ...


_CHECKPOINT_DEQUANTIZERS: dict[str, CheckpointDequantizer] = {}


def _build_e4m3_lut() -> tuple[float, ...]:
    values = []
    for code in range(256):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
            continue
        magnitude = mantissa * 2.0**-9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values.append(-magnitude if code & 0x80 else magnitude)
    return tuple(values)


_E4M3_LUT = _build_e4m3_lut()
_UE8M0_LUT = tuple(float("inf") if exponent == 255 else 2.0 ** (exponent - 127) for exponent in range(256))


def _normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError(f"Checkpoint quantization method must be a string, got {type(method).__name__}.")
    normalized = method.strip().lower()
    if not normalized:
        raise ValueError("Checkpoint quantization method must not be empty.")
    return normalized


def register_checkpoint_dequantizer(method: str, dequantizer: CheckpointDequantizer) -> None:
    method = _normalize_method(method)
    if method in _CHECKPOINT_DEQUANTIZERS:
        raise ValueError(f"Checkpoint dequantizer {method!r} is already registered.")
    if not callable(getattr(dequantizer, "dequantize", None)):
        raise TypeError("Checkpoint dequantizer must provide a callable dequantize() method.")
    _CHECKPOINT_DEQUANTIZERS[method] = dequantizer


def get_checkpoint_dequantizer(method: str) -> CheckpointDequantizer:
    method = _normalize_method(method)
    try:
        return _CHECKPOINT_DEQUANTIZERS[method]
    except KeyError as exc:
        supported = sorted(_CHECKPOINT_DEQUANTIZERS)
        raise ValueError(
            f"Unsupported checkpoint quantization method {method!r}; registered methods: {supported}."
        ) from exc


def checkpoint_dequantize(
    method: str,
    components: dict[str, paddle.Tensor],
    spec: CheckpointDequantSpec,
    output_dtype: paddle.dtype,
) -> paddle.Tensor:
    return get_checkpoint_dequantizer(method).dequantize(components, spec, output_dtype)


def _dtype_name(tensor: paddle.Tensor) -> str:
    return str(tensor.dtype).split(".")[-1].lower()


def _as_uint8_codes(tensor: paddle.Tensor, component_name: str) -> paddle.Tensor:
    dtype = _dtype_name(tensor)
    if dtype not in {"uint8", "int8"}:
        raise TypeError(
            f"{component_name} must use raw uint8/int8 storage, got {tensor.dtype}. "
            "The Paddle safetensors reader must preserve unsupported 8-bit formats as raw bytes."
        )
    return tensor.astype("uint8").astype("int32")


def _decode_e4m3(tensor: paddle.Tensor) -> paddle.Tensor:
    dtype = _dtype_name(tensor)
    if "float8_e4m3" in dtype:
        return tensor.astype("float32")

    raw = _as_uint8_codes(tensor, "qweight")
    lut = paddle.to_tensor(_E4M3_LUT, dtype="float32", place=tensor.place)
    return paddle.gather(lut, raw.flatten()).reshape(raw.shape)


def _decode_ue8m0(tensor: paddle.Tensor) -> paddle.Tensor:
    raw = _as_uint8_codes(tensor, "scale")
    lut = paddle.to_tensor(_UE8M0_LUT, dtype="float32", place=tensor.place)
    return paddle.gather(lut, raw.flatten()).reshape(raw.shape)


def _decode_e2m1_packed(tensor: paddle.Tensor) -> paddle.Tensor:
    raw = _as_uint8_codes(tensor, "qweight")
    low = paddle.bitwise_and(raw, paddle.full_like(raw, 0xF))
    high = paddle.bitwise_and(paddle.bitwise_right_shift(raw, paddle.full_like(raw, 4)), paddle.full_like(raw, 0xF))
    codes = paddle.stack([low, high], axis=-1).reshape([*tensor.shape[:-1], tensor.shape[-1] * 2])
    lut = paddle.to_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype="float32",
        place=tensor.place,
    )
    return paddle.gather(lut, codes.flatten()).reshape(codes.shape)


def _require_components(components: dict[str, paddle.Tensor], required: set[str]) -> None:
    missing = required - set(components)
    if missing:
        raise KeyError(f"Missing checkpoint quantization components: {sorted(missing)}.")


def _normalize_block_geometry(spec: CheckpointDequantSpec, rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if spec.block_axes is None or spec.block_shape is None:
        raise ValueError(f"{spec.method} requires block_axes and block_shape.")
    if len(spec.block_axes) != len(spec.block_shape) or not spec.block_axes:
        raise ValueError("block_axes and block_shape must be non-empty and have the same length.")

    axes = tuple(axis + rank if axis < 0 else axis for axis in spec.block_axes)
    if len(set(axes)) != len(axes) or any(axis < 0 or axis >= rank for axis in axes):
        raise ValueError(f"Invalid block_axes {spec.block_axes} for a rank-{rank} tensor.")
    if any(size <= 0 for size in spec.block_shape):
        raise ValueError(f"block_shape values must be positive, got {spec.block_shape}.")
    return axes, tuple(spec.block_shape)


def _expand_scale_grid(
    scales: paddle.Tensor,
    logical_shape: tuple[int, ...],
    spec: CheckpointDequantSpec,
) -> paddle.Tensor:
    axes, block_shape = _normalize_block_geometry(spec, len(logical_shape))
    if len(scales.shape) != len(logical_shape):
        raise ValueError(
            f"Scale rank must match logical weight rank: scale={tuple(scales.shape)}, logical={logical_shape}."
        )

    axis_to_block = dict(zip(axes, block_shape))
    expected_scale_shape = tuple(
        math.ceil(dim / axis_to_block[axis]) if axis in axis_to_block else dim
        for axis, dim in enumerate(logical_shape)
    )
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"Invalid scale grid shape: expected {expected_scale_shape}, got {tuple(scales.shape)} "
            f"for logical shape {logical_shape}."
        )

    expanded = scales
    for axis, block_size in sorted(axis_to_block.items()):
        expanded = paddle.repeat_interleave(expanded, repeats=block_size, axis=axis)
    slices = tuple(slice(0, dim) for dim in logical_shape)
    return expanded[slices]


def _decode_scales(scale: paddle.Tensor, spec: CheckpointDequantSpec) -> paddle.Tensor:
    scale_format = (spec.scale_format or "").lower()
    if scale_format in {"ue8m0", "e8m0"}:
        return _decode_ue8m0(scale)
    if scale_format in {"float", "fp16", "fp32", "bf16"}:
        return scale.astype("float32")
    raise ValueError(f"Unsupported scale_format {spec.scale_format!r} for {spec.method}.")


class FP8BlockCheckpointDequantizer:
    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        spec: CheckpointDequantSpec,
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        _require_components(components, {"qweight", "scale"})
        if spec.value_format not in {"e4m3", "fp8_e4m3"}:
            raise ValueError(f"fp8_block requires value_format='e4m3', got {spec.value_format!r}.")
        if spec.storage_layout != "plain" or spec.scale_layout != "block_grid":
            raise ValueError(
                "fp8_block requires storage_layout='plain' and scale_layout='block_grid', got "
                f"{spec.storage_layout!r} and {spec.scale_layout!r}."
            )

        qweight = _decode_e4m3(components["qweight"])
        logical_shape = tuple(qweight.shape)
        scales = _expand_scale_grid(_decode_scales(components["scale"], spec), logical_shape, spec)
        return (qweight * scales).astype(output_dtype)


class MXFP4GroupCheckpointDequantizer:
    def dequantize(
        self,
        components: dict[str, paddle.Tensor],
        spec: CheckpointDequantSpec,
        output_dtype: paddle.dtype,
    ) -> paddle.Tensor:
        _require_components(components, {"qweight", "scale"})
        if spec.bits != 4 or spec.value_format != "e2m1":
            raise ValueError(
                f"mxfp4_group requires bits=4 and value_format='e2m1', got {spec.bits} and {spec.value_format!r}."
            )
        if spec.storage_layout != "two_nibbles_last_axis_low_high":
            raise ValueError(f"Unsupported MXFP4 storage layout {spec.storage_layout!r}.")
        if spec.scale_layout != "row_group_grid":
            raise ValueError(f"Unsupported MXFP4 scale layout {spec.scale_layout!r}.")

        qweight = _decode_e2m1_packed(components["qweight"])
        logical_shape = tuple(qweight.shape)
        scales = _expand_scale_grid(_decode_scales(components["scale"], spec), logical_shape, spec)
        return (qweight * scales).astype(output_dtype)


register_checkpoint_dequantizer("fp8_block", FP8BlockCheckpointDequantizer())
register_checkpoint_dequantizer("mxfp4_group", MXFP4GroupCheckpointDequantizer())
