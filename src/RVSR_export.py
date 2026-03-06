"""https://github.com/huai-chang/RVSR
├─ RVSR-main/
   ├─ RVSR_export.py
"""

import torch
import torch.onnx
import sys
import os
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.inference_arch import RVSR
from argparse import ArgumentParser

def export_onnx(model, output_path, opset_version=18, fp16=False):

    dummy_input = torch.randn(1, 3, 64, 64)
    
    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()
    
    model.eval()

    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height_out', 3: 'width_out'}
    }

    temp_output_path = output_path + ".temp"

    torch.onnx.export(
        model,
        dummy_input,
        temp_output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    external_data_path = temp_output_path + ".data"
    
    if os.path.exists(external_data_path):
        print("Merging external data into single file...")
        onnx_model = onnx.load(temp_output_path)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        os.remove(temp_output_path)
        os.remove(external_data_path)
    else:
        os.rename(temp_output_path, output_path)
    
    print(f"Successfully exported to {output_path}")


def main():
    parser = ArgumentParser(description='Export RVSR model to ONNX format')
    parser.add_argument('--model_path', type=str, default='pretrianed/RVSR_rep.pth',
                        help='Path to the reparameterized model')
    parser.add_argument('--output_dir', type=str, default='pretrianed',
                        help='Output directory for ONNX files')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version')
    args = parser.parse_args()

    print("Loading model...")
    model = RVSR(sr_rate=4, N=16)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print("\nExporting FP32 ONNX model...")
    fp32_output = f"{args.output_dir}/RVSR_rep_fp32_op{args.opset}.onnx"
    export_onnx(model, fp32_output, opset_version=args.opset, fp16=False)

    model_fp16 = RVSR(sr_rate=4, N=16)
    model_fp16.load_state_dict(state_dict)
    model_fp16.eval()

    print("\nExporting FP16 ONNX model...")
    fp16_output = f"{args.output_dir}/RVSR_rep_fp16_op{args.opset}.onnx"
    export_onnx(model_fp16, fp16_output, opset_version=args.opset, fp16=True)
    
    print("\n" + "="*50)
    print("Export completed!")
    print(f"FP32 model: {fp32_output}")
    print(f"FP16 model: {fp16_output}")
    print("="*50)

if __name__ == '__main__':
    main()
