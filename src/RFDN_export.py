"""https://github.com/njulj/RFDN
├─ RFDN-master/
   ├─ RFDN_export.py
"""

import argparse
import os
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RFDN class from RFDN.py
from RFDN import RFDN

def export_onnx(
    model_path: str,
    output_path: str = None,
    fp16: bool = False,
    opset_version: int = 18,
    input_height: int = 64,
    input_width: int = 64
):
    """
    Export RFDN model to ONNX format.
    
    Args:
        model_path: Path to the .pth model file
        output_path: Output ONNX file path (auto-generated if None)
        fp16: If True, export model in FP16 precision
        opset_version: ONNX opset version
        input_height: Dummy input height for tracing
        input_width: Dummy input width for tracing
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    model = RFDN()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    dtype = torch.float32
    if fp16:
        print("Converting model to FP16...")
        model = model.half()
        dtype = torch.float16

    dummy_input = torch.randn(1, 3, input_height, input_width, dtype=dtype, device=device)
    print(f"Dummy input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")

    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape} (expected 4x upscale)")

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        suffix = "_fp16" if fp16 else ""
        output_path = os.path.join(os.path.dirname(model_path), f"{base_name}{suffix}.onnx")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    dynamic_axes = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height_x4', 3: 'width_x4'}
    }
    
    # Export to ONNX (dynamo=False ensures single-file output without external data)
    print(f"Exporting to ONNX (opset {opset_version}, single file)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=False  # Use legacy exporter for single-file ONNX
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")

        print(f"\nModel Info:")
        print(f"  - IR version: {onnx_model.ir_version}")
        print(f"  - Opset version: {onnx_model.opset_import[0].version}")
        print(f"  - Producer: {onnx_model.producer_name}")

        for input_tensor in onnx_model.graph.input:
            print(f"  - Input: {input_tensor.name}, shape: {[d.dim_param or d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        for output_tensor in onnx_model.graph.output:
            print(f"  - Output: {output_tensor.name}, shape: {[d.dim_param or d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
            
    except ImportError:
        print("Note: Install 'onnx' package for model verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Export RFDN model to ONNX format')
    parser.add_argument('--model', type=str, default='trained_model/RFDN_AIM.pth',
                        help='Path to the .pth model file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path (auto-generated if not specified)')
    parser.add_argument('--fp16', action='store_true',
                        help='Export model in FP16 precision using PyTorch half()')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version (default: 18)')
    parser.add_argument('--height', type=int, default=64,
                        help='Dummy input height for tracing (default: 64)')
    parser.add_argument('--width', type=int, default=64,
                        help='Dummy input width for tracing (default: 64)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    export_onnx(
        model_path=args.model,
        output_path=args.output,
        fp16=args.fp16,
        opset_version=args.opset,
        input_height=args.height,
        input_width=args.width
    )

if __name__ == '__main__':
    main()
