"""https://github.com/aselsan-research-imaging-team/bicubic-plusplus
├─ bicubic-plusplus-main/
   ├─ Bicubicpp_export.py
"""

import torch
import torch.nn as nn
import os
import sys
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.SR_models import Bicubic_plus_plus


def convert_to_single_file(onnx_path):
    model = onnx.load(onnx_path)

    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

    load_external_data_for_model(model, os.path.dirname(onnx_path))

    for tensor in model.graph.initializer:
        if tensor.HasField("data_location"):
            tensor.ClearField("data_location")

    onnx.save(model, onnx_path)

    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"已删除外部数据文件: {data_path}")


def export_to_onnx():

    sr_rate = 3
    model = Bicubic_plus_plus(sr_rate=sr_rate)
    
    pretrained_path = "./pretrained/bicubic_pp_x3.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    model.eval()

    output_dir = "./pretrained"
    os.makedirs(output_dir, exist_ok=True)

    dynamic_axes = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    }

    dummy_input = torch.randn(1, 3, 64, 64)

    print("导出 FP32 ONNX 模型...")
    fp32_path = os.path.join(output_dir, "bicubic_pp_x3_fp32.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"FP32 模型已保存至: {fp32_path}")

    convert_to_single_file(fp32_path)

    print("\n导出 FP16 ONNX 模型...")

    model_fp16 = Bicubic_plus_plus(sr_rate=sr_rate)
    model_fp16.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
    model_fp16.half()
    model_fp16.eval()

    dummy_input_fp16 = dummy_input.half()
    
    fp16_path = os.path.join(output_dir, "bicubic_pp_x3_fp16.onnx")
    
    torch.onnx.export(
        model_fp16,
        dummy_input_fp16,
        fp16_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"FP16 模型已保存至: {fp16_path}")

    convert_to_single_file(fp16_path)

    print("\n========== 导出完成 ==========")
    for path in [fp32_path, fp16_path]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{os.path.basename(path)}: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_to_onnx()
