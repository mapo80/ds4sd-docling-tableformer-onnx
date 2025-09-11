import os

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.transformers.float16 import convert_float_to_float16
from transformers import AutoModelForObjectDetection


class RandomDataReader(CalibrationDataReader):
    """Supply random images for static quantization calibration."""

    def __init__(self, num_batches: int = 5):
        self.num_batches = num_batches
        self.batch = 0

    def get_next(self):
        if self.batch >= self.num_batches:
            return None
        self.batch += 1
        data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        return {"pixel_values": data}

    def rewind(self):
        self.batch = 0


def convert(
    output_path: str = "docling_layout_heron.onnx",
    quantized_path: str = "docling_layout_heron_quant.onnx",
    fp16_path: str = "docling_layout_heron_fp16.onnx",
    static_quant_path: str = "docling_layout_heron_static.onnx",
) -> None:
    """Export the ds4sd/docling-layout-heron model to ONNX and produce
    smaller variants.

    Besides the full precision model, a dynamically quantized version
    (``int8`` MatMul weights), a float16 converted model and a statically
    quantized model are created. Static quantization uses random images for
    calibration and quantizes both weights and activations to ``int8``.

    All models expect an input tensor ``pixel_values`` of shape
    ``(batch, 3, 640, 640)`` and return ``logits`` and ``pred_boxes``.
    """

    model = AutoModelForObjectDetection.from_pretrained("ds4sd/docling-layout-heron")
    model.eval()
    model.config.return_dict = False

    dummy = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved ONNX model to {output_path} ({size_mb:.1f} MB)")

    # Simplify the graph to fold constants and remove unused nodes
    try:
        import onnxsim

        model_simp, check = onnxsim.simplify(output_path)
        if check:
            onnx.save(model_simp, output_path)
            size_mb = os.path.getsize(output_path) / 1024**2
            print(f"Simplified model saved to {output_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"ONNX simplification failed: {e}")

    # Dynamically quantize MatMul nodes (int8 weights)
    try:
        quantize_dynamic(
            output_path,
            quantized_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        q_size_mb = os.path.getsize(quantized_path) / 1024**2
        print(f"Saved quantized model to {quantized_path} ({q_size_mb:.1f} MB)")
    except Exception as e:
        print(f"Quantization failed: {e}")

    # Convert the full-precision model to float16
    try:
        fp32_model = onnx.load(output_path)
        fp16_model = convert_float_to_float16(
            fp32_model, keep_io_types=True, op_block_list=["MatMul"]
        )
        onnx.save(fp16_model, fp16_path)
        fp16_size_mb = os.path.getsize(fp16_path) / 1024**2
        print(f"Saved float16 model to {fp16_path} ({fp16_size_mb:.1f} MB)")
    except Exception as e:
        print(f"FP16 conversion failed: {e}")

    # Statically quantize weights and activations to int8
    try:
        dr = RandomDataReader()
        quantize_static(
            output_path,
            static_quant_path,
            dr,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
        s_size_mb = os.path.getsize(static_quant_path) / 1024**2
        print(f"Saved static int8 model to {static_quant_path} ({s_size_mb:.1f} MB)")
    except Exception as e:
        print(f"Static quantization failed: {e}")
    return None


if __name__ == "__main__":
    convert()
