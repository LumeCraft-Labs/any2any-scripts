"""https://github.com/saiFarmer/GRDFNet
├─ GRDFNet-main/
   ├─ GRDFNet_export.py

Export GRDFNet model to ONNX format.

Supports three recommended configurations:
- Standard:    num_sets=3, feature_channels=32 (balanced quality/speed)
- HighQuality: num_sets=6, feature_channels=48 (best quality, ~4x slower)
- Lightweight: num_sets=3, feature_channels=24 (fast, for video inference)
"""

import sys
import os

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
from GRDFNet import GRDFNet


# Recommended configurations from the author
CONFIGS = {
    "standard": {
        "num_sets": 3,
        "feature_channels": 32,
        "description": "Balanced quality/speed for desktop workloads",
    },
    "high_quality": {
        "num_sets": 6,
        "feature_channels": 48,
        "description": "Highest quality, ~4x slower than standard",
    },
    "lightweight": {
        "num_sets": 3,
        "feature_channels": 24,
        "description": "Fast inference for video, 50-75% faster than standard",
    },
}


def get_config_name(num_sets: int, feature_channels: int) -> str:
    """Get configuration name based on parameters."""
    for name, cfg in CONFIGS.items():
        if cfg["num_sets"] == num_sets and cfg["feature_channels"] == feature_channels:
            return name
    return f"custom_s{num_sets}_c{feature_channels}"


def export_to_onnx(
    pth_path: str,
    onnx_path: str,
    input_height: int = 256,
    input_width: int = 256,
    opset_version: int = 18,
    dynamic_axes: bool = True,
):

    # Load checkpoint
    print(f"Loading checkpoint from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params", checkpoint)
    
    # Infer model configuration from state_dict
    # head.weight shape: [feature_channels, num_in_ch, 3, 3]
    # tail.weight shape: [num_out_ch, feature_channels, 3, 3]
    feature_channels = state_dict["head.weight"].shape[0]
    num_in_ch = state_dict["head.weight"].shape[1]
    num_out_ch = state_dict["tail.weight"].shape[0]
    
    # Count body layers to determine num_sets
    # body structure: 3xLWGRB + 2xLWDRB + num_sets x [LWGRB + LWDRB]
    # = 5 + 2*num_sets blocks
    body_indices = set()
    for key in state_dict.keys():
        if key.startswith("body."):
            idx = int(key.split(".")[1])
            body_indices.add(idx)
    num_body_blocks = len(body_indices)
    num_sets = (num_body_blocks - 5) // 2
    
    # Check for upsample layers
    has_upsample = any("upsample0.expand" in k for k in state_dict.keys())
    if has_upsample:
        # Infer scale from upsample0.expand.weight shape
        # shape: [out_ch * scale * scale, in_ch, 3, 3]
        expand_weight = state_dict["upsample0.expand.weight"]
        scale_squared = expand_weight.shape[0] // num_out_ch
        upscale = int(scale_squared ** 0.5)
    else:
        upscale = 1
    
    print(f"Inferred model config:")
    print(f"  num_in_ch: {num_in_ch}")
    print(f"  num_out_ch: {num_out_ch}")
    print(f"  feature_channels: {feature_channels}")
    print(f"  upscale: {upscale}")
    print(f"  num_sets: {num_sets}")
    print(f"  num_body_blocks: {num_body_blocks}")
    
    # Create model
    model = GRDFNet(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        upscale=upscale,
        num_sets=num_sets,
        bias=True,
        norm=False,
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, num_in_ch, input_height, input_width)
    
    # Test forward pass
    print(f"Testing forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    print(f"Exporting to ONNX: {onnx_path}")
    
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    else:
        dynamic_axes_dict = None
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # Convert external data to single file
    try:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        # Remove external data and embed weights in the model
        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=False,
        )

        external_data_path = onnx_path + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
            print(f"Removed external data file: {external_data_path}")
        print(f"Converted to single-file ONNX")
    except Exception as e:
        print(f"Warning: Could not convert to single file: {e}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Note: Install 'onnx' package to verify the exported model")
    except Exception as e:
        print(f"ONNX verification warning: {e}")
    
    return onnx_path


def export_from_weights(
    pth_path: str,
    output_dir: str = ".",
    input_height: int = 256,
    input_width: int = 256,
    opset_version: int = 18,
    dynamic_axes: bool = True,
):
    """
    Export a single .pth checkpoint to ONNX with auto-generated name.
    
    Returns:
        Tuple of (onnx_path, config_name)
    """
    # Load and infer config
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params", checkpoint)
    
    feature_channels = state_dict["head.weight"].shape[0]
    body_indices = set()
    for key in state_dict.keys():
        if key.startswith("body."):
            idx = int(key.split(".")[1])
            body_indices.add(idx)
    num_body_blocks = len(body_indices)
    num_sets = (num_body_blocks - 5) // 2
    
    has_upsample = any("upsample0.expand" in k for k in state_dict.keys())
    if has_upsample:
        expand_weight = state_dict["upsample0.expand.weight"]
        num_out_ch = state_dict["tail.weight"].shape[0]
        scale_squared = expand_weight.shape[0] // num_out_ch
        upscale = int(scale_squared ** 0.5)
    else:
        upscale = 1
    
    # Generate output filename
    config_name = get_config_name(num_sets, feature_channels)
    base_name = os.path.splitext(os.path.basename(pth_path))[0]
    onnx_filename = f"GRDFNet_{upscale}x_{config_name}_s{num_sets}_c{feature_channels}.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)
    
    export_to_onnx(
        pth_path=pth_path,
        onnx_path=onnx_path,
        input_height=input_height,
        input_width=input_width,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )
    
    return onnx_path, config_name


def export_model_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    opset_version: int = 18,
    dynamic_axes: bool = True,
    fp16: bool = False,
):

    model.eval()
    
    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()
    
    if dynamic_axes:
        dynamic_axes_dict = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    else:
        dynamic_axes_dict = None
    
    # Use dynamo=False for FP16 to avoid compatibility issues
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        dynamo=not fp16,  # Disable dynamo for FP16 due to compatibility issues
    )
    
    # Convert external data to single file
    try:
        import onnx
        onnx_model = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(
            onnx_model,
            onnx_path,
            save_as_external_data=False,
        )

        external_data_path = onnx_path + ".data"
        if os.path.exists(external_data_path):
            os.remove(external_data_path)
    except Exception as e:
        print(f"Warning: Could not convert to single file: {e}")
    
    return onnx_path


def export_all_configs(
    pth_path: str,
    output_dir: str = ".",
    input_height: int = 256,
    input_width: int = 256,
    opset_version: int = 18,
    dynamic_axes: bool = True,
    export_fp16: bool = True,
):
    """
    Export ONNX model from the provided .pth checkpoint.
    Only exports the configuration that matches the checkpoint's architecture.
    Exports both FP32 and FP16 variants.
    
    Note: Each .pth file corresponds to a specific configuration.
    To export other configurations, you need their respective .pth files.
    """
    # Load checkpoint
    print(f"Loading checkpoint from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params", checkpoint)
    
    # Infer model configuration from state_dict
    feature_channels = state_dict["head.weight"].shape[0]
    num_in_ch = state_dict["head.weight"].shape[1]
    num_out_ch = state_dict["tail.weight"].shape[0]
    
    # Count body layers to determine num_sets
    body_indices = set()
    for key in state_dict.keys():
        if key.startswith("body."):
            idx = int(key.split(".")[1])
            body_indices.add(idx)
    num_body_blocks = len(body_indices)
    num_sets = (num_body_blocks - 5) // 2
    
    # Infer upscale from original model
    has_upsample = any("upsample0.expand" in k for k in state_dict.keys())
    if has_upsample:
        expand_weight = state_dict["upsample0.expand.weight"]
        scale_squared = expand_weight.shape[0] // num_out_ch
        upscale = int(scale_squared ** 0.5)
    else:
        upscale = 1

    config_name = get_config_name(num_sets, feature_channels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    exported = []
    
    print(f"\n{'='*60}")
    print(f"Exporting {config_name.upper()} configuration (from checkpoint)")
    print(f"  num_in_ch: {num_in_ch}")
    print(f"  num_out_ch: {num_out_ch}")
    print(f"  feature_channels: {feature_channels}")
    print(f"  upscale: {upscale}")
    print(f"  num_sets: {num_sets}")
    print(f"  num_body_blocks: {num_body_blocks}")
    if config_name in CONFIGS:
        print(f"  {CONFIGS[config_name]['description']}")
    print(f"{'='*60}")
    
    # Create model with this configuration
    model = GRDFNet(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        feature_channels=feature_channels,
        upscale=upscale,
        num_sets=num_sets,
        bias=True,
        norm=False,
    )
    
    # Load the actual weights from checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded weights from checkpoint successfully!")

    dummy_input = torch.randn(1, num_in_ch, input_height, input_width)
    
    # Test forward pass
    print(f"Testing forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Export FP32 variant
    onnx_filename_fp32 = f"GRDFNet_{upscale}x_{config_name}_s{num_sets}_c{feature_channels}_fp32.onnx"
    onnx_path_fp32 = os.path.join(output_dir, onnx_filename_fp32)
    print(f"Exporting FP32 to ONNX: {onnx_path_fp32}")
    
    export_model_to_onnx(
        model=model,
        onnx_path=onnx_path_fp32,
        dummy_input=dummy_input.clone(),
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        fp16=False,
    )
    
    # Verify FP32 model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path_fp32)
        onnx.checker.check_model(onnx_model)
        print("FP32 ONNX model verification passed!")
    except Exception as e:
        print(f"FP32 ONNX verification warning: {e}")
    
    file_size_fp32 = os.path.getsize(onnx_path_fp32) / 1024  # KB
    exported.append({
        "config": config_name,
        "precision": "fp32",
        "path": onnx_path_fp32,
        "size_kb": file_size_fp32,
        "num_sets": num_sets,
        "feature_channels": feature_channels,
    })
    print(f"Saved: {onnx_path_fp32} ({file_size_fp32:.1f} KB)")

    if export_fp16:
        onnx_filename_fp16 = f"GRDFNet_{upscale}x_{config_name}_s{num_sets}_c{feature_channels}_fp16.onnx"
        onnx_path_fp16 = os.path.join(output_dir, onnx_filename_fp16)
        print(f"Exporting FP16 to ONNX: {onnx_path_fp16}")
        
        # Create a fresh model for FP16 export
        model_fp16 = GRDFNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=upscale,
            num_sets=num_sets,
            bias=True,
            norm=False,
        )
        model_fp16.load_state_dict(state_dict)
        
        try:
            export_model_to_onnx(
                model=model_fp16,
                onnx_path=onnx_path_fp16,
                dummy_input=dummy_input.clone(),
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                fp16=True,
            )
            
            # Verify FP16 model
            try:
                import onnx
                onnx_model = onnx.load(onnx_path_fp16)
                onnx.checker.check_model(onnx_model)
                print("FP16 ONNX model verification passed!")
            except Exception as e:
                print(f"FP16 ONNX verification warning: {e}")
            
            file_size_fp16 = os.path.getsize(onnx_path_fp16) / 1024  # KB
            exported.append({
                "config": config_name,
                "precision": "fp16",
                "path": onnx_path_fp16,
                "size_kb": file_size_fp16,
                "num_sets": num_sets,
                "feature_channels": feature_channels,
            })
            print(f"Saved: {onnx_path_fp16} ({file_size_fp16:.1f} KB)")
        except Exception as e:
            print(f"Warning: FP16 export failed: {e}")

    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'Precision':<10} {'num_sets':<10} {'channels':<10} {'Size (KB)':<12} {'File'}")
    print("-" * 100)
    for e in exported:
        print(f"{e['config']:<15} {e['precision']:<10} {e['num_sets']:<10} {e['feature_channels']:<10} {e['size_kb']:<12.1f} {os.path.basename(e['path'])}")
    print(f"\nExported from checkpoint: {pth_path}")
    print(f"\nFP32: Full precision, maximum accuracy")
    print(f"FP16: Half precision, ~50% smaller, faster on GPUs with FP16 support")
    
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GRDFNet to ONNX")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="1x_pretrain.pth",
        help="Input .pth checkpoint path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .onnx path (default: same name as input)"
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=256,
        help="Input height for tracing"
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=256,
        help="Input width for tracing"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18, recommended 17-21 for TensorRT)"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static input/output shapes (no dynamic axes)"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Export all three recommended configurations (standard, high_quality, lightweight)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for ONNX files (used with --all-configs)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip FP16 export (only export FP32)"
    )
    
    args = parser.parse_args()
    
    if args.all_configs:
        # Export all three configurations
        export_all_configs(
            pth_path=args.input,
            output_dir=args.output_dir,
            input_height=args.height,
            input_width=args.width,
            opset_version=args.opset,
            dynamic_axes=not args.static,
            export_fp16=not args.no_fp16,
        )
    else:
        # Export single model from weights
        if args.output is None:
            args.output = args.input.replace(".pth", ".onnx")
        
        export_to_onnx(
            pth_path=args.input,
            onnx_path=args.output,
            input_height=args.height,
            input_width=args.width,
            opset_version=args.opset,
            dynamic_axes=not args.static,
        )
