#!/usr/bin/env python3
"""
NCNN to PyTorch PTH Converter for SRVGGNetCompact Models

Converts NCNN .param/.bin files back to PyTorch .pth format.
"""

import os
import sys
import struct
import argparse
import numpy as np
from pathlib import Path

# Add ESRGAN-Compact to path for importing the architecture
SCRIPT_DIR = Path(__file__).parent
ESRGAN_DIR = SCRIPT_DIR.parent / "ESRGAN-Compact"
sys.path.insert(0, str(ESRGAN_DIR))

try:
    import torch
    from torch import nn
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    
    Copied from srvgg_arch.py to avoid basicsr dependency.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        import torch.nn.functional as F
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

def parse_param_file(param_path):
    layers = []
    
    with open(param_path, 'r') as f:
        lines = f.readlines()
    
    # First line: magic number (7767517)
    # Second line: layer_count blob_count
    layer_count, blob_count = map(int, lines[1].strip().split())
    
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        layer_type = parts[0]
        layer_name = parts[1]
        input_count = int(parts[2])
        output_count = int(parts[3])
        
        # Parse layer-specific params
        params = {}
        param_start = 4 + input_count + output_count
        for param in parts[param_start:]:
            if '=' in param:
                key, value = param.split('=')
                key = int(key)
                # Try to parse as int, then float
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                params[key] = value
        
        layers.append({
            'type': layer_type,
            'name': layer_name,
            'params': params
        })
    
    return layers

def align_size(size, alignment=4):
    return (size + alignment - 1) // alignment * alignment

def read_weight_type0(f, count):
    """Read weight data with type=0 (may have flag header).
    """
    # Read 4-byte flag
    flag_bytes = f.read(4)
    if len(flag_bytes) < 4:
        raise ValueError("Unexpected end of file reading flag")
    
    flag = struct.unpack('<I', flag_bytes)[0]
    
    if flag == 0x01306B47:
        # float16 data
        align_data_size = align_size(count * 2, 4)
        data = np.frombuffer(f.read(align_data_size), dtype=np.float16)[:count]
        return data.astype(np.float32)
    
    elif flag == 0x000D4B38:
        # int8 data
        align_data_size = align_size(count, 4)
        data = np.frombuffer(f.read(align_data_size), dtype=np.int8)[:count]
        return data.astype(np.float32)
    
    elif flag == 0x0002C056:
        # float32 with extra scaling (raw data after tag)
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
        return data.copy()
    
    else:
        # Check if it's quantized or raw float32
        f0 = flag & 0xFF
        f1 = (flag >> 8) & 0xFF
        f2 = (flag >> 16) & 0xFF
        f3 = (flag >> 24) & 0xFF
        flag_sum = f0 + f1 + f2 + f3
        
        if flag_sum != 0:
            # Quantized data: 256 float quantization table + indices
            # First, seek back and read quantization table (flag was first 4 bytes of table)
            f.seek(-4, 1)
            quant_table = np.frombuffer(f.read(256 * 4), dtype=np.float32)
            align_data_size = align_size(count, 4)
            indices = np.frombuffer(f.read(align_data_size), dtype=np.uint8)[:count]
            return quant_table[indices].copy()
        else:
            # Raw float32 data (flag bytes were 0x00000000)
            # flag is already part of the alignment, but we need to check if
            # the low byte (f0) is 0, meaning raw float32
            if f0 == 0:
                data = np.frombuffer(f.read(count * 4), dtype=np.float32)
                return data.copy()
            else:
                raise ValueError(f"Unknown weight format flag: 0x{flag:08X}")

def read_weight_type1(f, count):
    """Read weight data with type=1 (raw float32, no flag).
    """
    data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.copy()

def detect_model_params(layers):
    """Detect model parameters from .param layers."""
    conv_layers = [l for l in layers if l['type'] == 'Convolution']
    prelu_layers = [l for l in layers if l['type'] == 'PReLU']
    
    if not conv_layers:
        raise ValueError("No Convolution layers found")
    
    # First conv: num_in_ch -> num_feat
    first_conv = conv_layers[0]
    num_feat = first_conv['params'].get(0, 64)  # 0 = num_output
    
    # Calculate num_conv from middle convolutions (all with same num_feat in/out)
    num_conv = 0
    for conv in conv_layers[1:-1]:  # Exclude first and last
        if conv['params'].get(0) == num_feat:
            num_conv += 1
    
    # Last conv: num_feat -> num_out_ch * upscale^2
    last_conv = conv_layers[-1]
    last_output = last_conv['params'].get(0)
    
    # Detect upscale from PixelShuffle
    pixelshuffle_layers = [l for l in layers if l['type'] == 'PixelShuffle']
    if pixelshuffle_layers:
        upscale = pixelshuffle_layers[0]['params'].get(0, 2)
    else:
        # Fallback: calculate from last conv output channels
        upscale = int(np.sqrt(last_output / 3))
    
    # Detect activation type from PReLU presence
    act_type = 'prelu' if prelu_layers else 'relu'
    
    return {
        'num_in_ch': 3,
        'num_out_ch': 3,
        'num_feat': num_feat,
        'num_conv': num_conv,
        'upscale': upscale,
        'act_type': act_type
    }

def convert_ncnn_to_pth(param_path, bin_path, output_path=None, verify=False):

    param_path = Path(param_path)
    bin_path = Path(bin_path)
    
    if output_path is None:
        output_path = param_path.with_suffix('.pth')
    else:
        output_path = Path(output_path)
    
    print(f"Parsing {param_path.name}...")
    layers = parse_param_file(param_path)
    
    # Detect model parameters
    model_params = detect_model_params(layers)
    print(f"Detected model parameters:")
    print(f"  num_feat={model_params['num_feat']}, num_conv={model_params['num_conv']}, "
          f"upscale={model_params['upscale']}, act_type={model_params['act_type']}")
    
    # Create PyTorch model
    model = SRVGGNetCompact(**model_params)
    
    # Read weights from .bin file
    print(f"Reading weights from {bin_path.name}...")
    state_dict = {}
    
    with open(bin_path, 'rb') as f:
        # Map NCNN layers to PyTorch state_dict keys
        body_idx = 0
        
        for layer in layers:
            if layer['type'] == 'Convolution':
                params = layer['params']
                num_output = params.get(0)
                kernel_w = params.get(1)
                kernel_h = params.get(11, kernel_w)
                weight_data_size = params.get(6)
                bias_term = params.get(5, 0)
                
                # Read weight (type=0)
                weight = read_weight_type0(f, weight_data_size)
                
                # Reshape weight: NCNN stores as (num_output, num_input, kH, kW) flattened
                num_input = weight_data_size // (num_output * kernel_w * kernel_h)
                weight = weight.reshape(num_output, num_input, kernel_h, kernel_w)
                
                state_dict[f'body.{body_idx}.weight'] = torch.from_numpy(weight)
                
                if bias_term:
                    bias = read_weight_type1(f, num_output)
                    state_dict[f'body.{body_idx}.bias'] = torch.from_numpy(bias)
                
                print(f"  Conv {layer['name']}: weight shape {weight.shape}, bias={bias_term}")
                body_idx += 1
                
            elif layer['type'] == 'PReLU':
                num_slope = layer['params'].get(0, 1)
                
                # Read slope data (type=1)
                slope = read_weight_type1(f, num_slope)
                
                state_dict[f'body.{body_idx}.weight'] = torch.from_numpy(slope)
                
                print(f"  PReLU {layer['name']}: {num_slope} slopes")
                body_idx += 1
    
    # Load state dict into model
    print("Loading weights into model...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    
    # Save model
    print(f"Saving to {output_path}...")
    
    # Save in the same format as typical ESRGAN models
    save_dict = {
        'params': model.state_dict(),
        'model_params': model_params
    }
    torch.save(save_dict, output_path)
    
    # Also save just the state_dict for direct loading
    state_dict_path = output_path.with_name(output_path.stem + '_state_dict.pth')
    torch.save(model.state_dict(), state_dict_path)
    print(f"Also saved state_dict to {state_dict_path}")
    
    if verify:
        print("\nVerifying model can be loaded...")
        test_model = SRVGGNetCompact(**model_params)
        test_model.load_state_dict(torch.load(state_dict_path, weights_only=True))
        test_model.eval()
        
        # Test forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 3, 64, 64)
            output = test_model(test_input)
            expected_size = 64 * model_params['upscale']
            print(f"  Input: {test_input.shape} -> Output: {output.shape}")
            print(f"  Expected output size: (1, 3, {expected_size}, {expected_size})")
            if output.shape == (1, 3, expected_size, expected_size):
                print("  ✓ Forward pass successful!")
            else:
                print("  ✗ Output shape mismatch!")
    
    print("\nConversion complete!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert NCNN models to PyTorch PTH')
    parser.add_argument('--param', type=str, help='Path to .param file')
    parser.add_argument('--bin', type=str, help='Path to .bin file')
    parser.add_argument('--output', type=str, help='Output .pth path')
    parser.add_argument('--all', action='store_true', help='Convert all models in current directory')
    parser.add_argument('--verify', action='store_true', help='Verify converted model')
    
    args = parser.parse_args()
    
    if args.all:
        # Find all .param files in current directory
        param_files = list(Path('.').glob('*.param'))
        if not param_files:
            print("No .param files found in current directory")
            return
        
        for param_path in param_files:
            bin_path = param_path.with_suffix('.bin')
            if bin_path.exists():
                print(f"\n{'='*60}")
                print(f"Converting: {param_path.name}")
                print('='*60)
                try:
                    convert_ncnn_to_pth(param_path, bin_path, verify=args.verify)
                except Exception as e:
                    print(f"Error converting {param_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Skipping {param_path.name}: no matching .bin file")
    
    elif args.param:
        param_path = Path(args.param)
        bin_path = Path(args.bin) if args.bin else param_path.with_suffix('.bin')
        convert_ncnn_to_pth(param_path, bin_path, args.output, verify=args.verify)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
