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

import json
import os
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.quantization.checkpoint_dequant import CheckpointDequantSpec, checkpoint_dequantize
from paddleformers.quantization.hf_checkpoint import (
    CompressedTensorsMXFP4HFQuantizationAdapter,
    FineGrainedFP8HFQuantizationAdapter,
    HFDequantLoadTransform,
    TensorMetadata,
    build_hf_dequant_load_transform,
    read_hf_safetensors_metadata,
)


def _write_safetensors_file(path, tensors):
    header = {}
    offset = 0
    for key, dtype, shape, byte_size in tensors:
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    with open(path, "wb") as file:
        file.write(len(raw_header).to_bytes(8, byteorder="little"))
        file.write(raw_header)
        file.write(bytes(offset))


class TestCheckpointDequantization(unittest.TestCase):
    def test_e4m3_raw_decoder_handles_subnormal_max_and_nan(self):
        qweight = paddle.to_tensor([[0x00, 0x01, 0x38, 0x7E, 0xFE, 0x7F]], dtype="uint8")
        scale = paddle.to_tensor([[127]], dtype="uint8")
        spec = CheckpointDequantSpec(
            method="fp8_block",
            bits=8,
            value_format="e4m3",
            scale_format="ue8m0",
            block_axes=(0, 1),
            block_shape=(1, 6),
            storage_layout="plain",
            scale_layout="block_grid",
        )

        output = checkpoint_dequantize(
            "fp8_block",
            {"qweight": qweight, "scale": scale},
            spec,
            paddle.float32,
        ).numpy()

        np.testing.assert_array_equal(output[0, :5], [0.0, 2.0**-9, 1.0, 448.0, -448.0])
        self.assertTrue(np.isnan(output[0, 5]))

    def test_fp8_block_with_raw_e4m3_and_ue8m0(self):
        qweight = paddle.to_tensor([[0x38] * 4, [0x38] * 4], dtype="uint8")
        scale = paddle.to_tensor([[127, 128], [126, 127]], dtype="uint8")
        spec = CheckpointDequantSpec(
            method="fp8_block",
            bits=8,
            value_format="e4m3",
            scale_format="ue8m0",
            block_axes=(0, 1),
            block_shape=(1, 2),
            storage_layout="plain",
            scale_layout="block_grid",
        )

        output = checkpoint_dequantize(
            "fp8_block",
            {"qweight": qweight, "scale": scale},
            spec,
            paddle.float32,
        )

        expected = np.array([[1.0, 1.0, 2.0, 2.0], [0.5, 0.5, 1.0, 1.0]], dtype="float32")
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_mxfp4_group_unpacks_low_nibble_first(self):
        # HF stores packed FP4 as I8, so bytes >= 128 arrive as negative int8.
        qweight = paddle.to_tensor([[0x21, -87]], dtype="int8")
        scale = paddle.to_tensor([[127, 128]], dtype="uint8")
        spec = CheckpointDequantSpec(
            method="mxfp4_group",
            bits=4,
            value_format="e2m1",
            scale_format="ue8m0",
            block_axes=(1,),
            block_shape=(2,),
            storage_layout="two_nibbles_last_axis_low_high",
            scale_layout="row_group_grid",
        )

        output = checkpoint_dequantize(
            "mxfp4_group",
            {"qweight": qweight, "scale": scale},
            spec,
            paddle.float32,
        )

        expected = np.array([[0.5, 1.0, -1.0, -2.0]], dtype="float32")
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_invalid_scale_grid_fails_before_broadcast(self):
        qweight = paddle.full([2, 4], 0x38, dtype="uint8")
        scale = paddle.full([1, 1], 127, dtype="uint8")
        spec = CheckpointDequantSpec(
            method="fp8_block",
            bits=8,
            value_format="e4m3",
            scale_format="ue8m0",
            block_axes=(0, 1),
            block_shape=(1, 2),
            storage_layout="plain",
            scale_layout="block_grid",
        )

        with self.assertRaisesRegex(ValueError, "Invalid scale grid shape"):
            checkpoint_dequantize(
                "fp8_block",
                {"qweight": qweight, "scale": scale},
                spec,
                paddle.float32,
            )


class TestHFQuantizationAdapter(unittest.TestCase):
    def setUp(self):
        self.quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [2, 2],
        }
        self.source_metadata = {
            "layers.0.attn.wq_a.weight": TensorMetadata((2, 4), "F8_E4M3"),
            "layers.0.attn.wq_a.scale": TensorMetadata((1, 2), "F8_E8M0"),
            "layers.0.ffn.experts.0.w1.weight": TensorMetadata((2, 2), "I8"),
            "layers.0.ffn.experts.0.w1.scale": TensorMetadata((2, 1), "F8_E8M0"),
            "layers.0.attn_norm.weight": TensorMetadata((4,), "BF16"),
        }

    def test_builds_mixed_fp8_fp4_manifest(self):
        manifest = FineGrainedFP8HFQuantizationAdapter().build_manifest(
            self.quantization_config,
            self.source_metadata,
        )

        self.assertEqual(
            set(manifest.weights),
            {
                "layers.0.attn.wq_a.weight",
                "layers.0.ffn.experts.0.w1.weight",
            },
        )
        fp8_spec = manifest.weights["layers.0.attn.wq_a.weight"]
        self.assertEqual(fp8_spec.quant_method, "fp8_block")
        self.assertEqual(fp8_spec.logical_shape, (2, 4))
        self.assertEqual(fp8_spec.block_axes, (0, 1))
        fp4_spec = manifest.weights["layers.0.ffn.experts.0.w1.weight"]
        self.assertEqual(fp4_spec.quant_method, "mxfp4_group")
        self.assertEqual(fp4_spec.logical_shape, (2, 4))
        self.assertEqual(fp4_spec.block_shape, (4,))

    def test_transform_maps_physical_keys_and_validates_shape(self):
        manifest = FineGrainedFP8HFQuantizationAdapter().build_manifest(
            self.quantization_config,
            self.source_metadata,
        )
        transform = HFDequantLoadTransform(manifest, paddle.bfloat16)
        logical_key = "layers.0.attn.wq_a.weight"

        self.assertEqual(
            transform.source_keys(logical_key),
            [logical_key, "layers.0.attn.wq_a.scale"],
        )
        self.assertEqual(transform.logical_metadata()[logical_key].global_shape, (2, 4))
        output = transform.apply(
            logical_key,
            {
                logical_key: paddle.full([2, 4], 0x38, dtype="uint8"),
                "layers.0.attn.wq_a.scale": paddle.full([1, 2], 127, dtype="uint8"),
            },
            paddle.float32,
        )
        np.testing.assert_array_equal(output.numpy(), np.ones([2, 4], dtype="float32"))


class TestKimiK3MXFP4QuantizationAdapter(unittest.TestCase):
    def setUp(self):
        self.quantization_config = {
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": False,
                        "group_size": 32,
                        "num_bits": 4,
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": True,
                        "type": "float",
                    },
                }
            },
            "format": "mxfp4-pack-quantized",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }
        self.logical_key = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        self.packed_key = f"{self.logical_key}_packed"
        self.scale_key = f"{self.logical_key}_scale"
        self.source_metadata = {
            self.packed_key: TensorMetadata((2, 16), "U8"),
            self.scale_key: TensorMetadata((2, 1), "U8"),
            "language_model.model.layers.1.self_attn.q_proj.weight": TensorMetadata((2, 32), "BF16"),
        }

    def test_builds_kimi_k3_mxfp4_manifest(self):
        adapter = CompressedTensorsMXFP4HFQuantizationAdapter()
        self.assertTrue(adapter.matches(self.quantization_config))

        manifest = adapter.build_manifest(self.quantization_config, self.source_metadata)

        self.assertEqual(set(manifest.weights), {self.logical_key})
        spec = manifest.weights[self.logical_key]
        self.assertEqual(spec.logical_shape, (2, 32))
        self.assertEqual(spec.components, {"qweight": self.packed_key, "scale": self.scale_key})
        self.assertEqual(spec.quant_method, "mxfp4_group")
        self.assertEqual(spec.scale_format, "ue8m0")
        self.assertEqual(spec.block_axes, (1,))
        self.assertEqual(spec.block_shape, (32,))

    def test_transform_dequantizes_kimi_k3_packed_weight(self):
        manifest = CompressedTensorsMXFP4HFQuantizationAdapter().build_manifest(
            self.quantization_config,
            self.source_metadata,
        )
        transform = HFDequantLoadTransform(manifest, paddle.bfloat16)
        packed = paddle.to_tensor([[0x21] * 16, [0xBA] * 16], dtype="uint8")
        scale = paddle.to_tensor([[127], [128]], dtype="uint8")

        output = transform.apply(
            self.logical_key,
            {self.packed_key: packed, self.scale_key: scale},
            paddle.float32,
        )

        expected = np.stack(
            [
                np.tile([0.5, 1.0], 16),
                np.tile([-1.0, -1.5], 16) * 2.0,
            ]
        ).astype("float32")
        np.testing.assert_array_equal(output.numpy(), expected)

    def test_build_transform_reads_nested_text_quantization_config(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(os.path.join(checkpoint_path, "config.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {"model_type": "kimi_k3", "text_config": {"quantization_config": self.quantization_config}},
                    file,
                )
            _write_safetensors_file(
                os.path.join(checkpoint_path, "model.safetensors"),
                [
                    (self.packed_key, "U8", (2, 16), 32),
                    (self.scale_key, "U8", (2, 1), 2),
                ],
            )

            transform = build_hf_dequant_load_transform(
                checkpoint_path=checkpoint_path,
                mode="dequantize_bf16",
            )

            self.assertIsInstance(transform, HFDequantLoadTransform)
            self.assertEqual(set(transform.logical_metadata()), {self.logical_key})
            self.assertEqual(transform.source_keys(self.logical_key), [self.packed_key, self.scale_key])


class TestHFSafetensorsMetadata(unittest.TestCase):
    def test_raw_header_reader_and_transform_builder(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            config = {
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [2, 2],
                }
            }
            with open(os.path.join(checkpoint_path, "config.json"), "w", encoding="utf-8") as file:
                json.dump(config, file)

            weight_file = "model-00001-of-00002.safetensors"
            scale_file = "model-00002-of-00002.safetensors"
            _write_safetensors_file(
                os.path.join(checkpoint_path, weight_file),
                [("layers.0.attn.wq_a.weight", "F8_E4M3", (2, 4), 8)],
            )
            _write_safetensors_file(
                os.path.join(checkpoint_path, scale_file),
                [("layers.0.attn.wq_a.scale", "F8_E8M0", (1, 2), 2)],
            )
            with open(
                os.path.join(checkpoint_path, "model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "metadata": {"total_size": 10},
                        "weight_map": {
                            "layers.0.attn.wq_a.weight": weight_file,
                            "layers.0.attn.wq_a.scale": scale_file,
                        },
                    },
                    file,
                )

            metadata = read_hf_safetensors_metadata(checkpoint_path)
            self.assertEqual(metadata["layers.0.attn.wq_a.weight"].dtype, "F8_E4M3")
            self.assertEqual(metadata["layers.0.attn.wq_a.scale"].file_name, scale_file)

            transform = build_hf_dequant_load_transform(
                checkpoint_path=checkpoint_path,
                mode="dequantize_bf16",
            )
            self.assertIsInstance(transform, HFDequantLoadTransform)
            self.assertEqual(set(transform.logical_metadata()), {"layers.0.attn.wq_a.weight"})

    def test_auto_mode_preserves_unquantized_hf_checkpoint(self):
        with tempfile.TemporaryDirectory() as checkpoint_path:
            with open(os.path.join(checkpoint_path, "config.json"), "w", encoding="utf-8") as file:
                json.dump({"model_type": "deepseek_v4"}, file)

            transform = build_hf_dequant_load_transform(
                checkpoint_path=checkpoint_path,
                mode="auto",
            )
            self.assertIsNone(transform)


if __name__ == "__main__":
    unittest.main()
