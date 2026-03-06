"""
fNNEDI3
"""

import argparse
import os
import struct
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Neurons(Enum):
    nns16 = 0
    nns32 = 1
    nns64 = 2
    nns128 = 3
    nns256 = 4

    def get_neurons(self) -> int:
        return [16, 32, 64, 128, 256][self.value]

class Window(Enum):
    win8x4 = 0
    win8x6 = 1

    def get_width(self) -> int:
        return 8

    def get_height(self) -> int:
        return [4, 6][self.value]

class NNEDI3Weights:
    """Load and extract weights from nnedi3_weights.bin"""
    
    weights_file = "nnedi3_weights.bin"
    weights_filesize = 83328 * 4
    weights = None
    
    weight_offsets = [0, 1088, 3264, 7616, 16320, 33728, 35328, 38528, 44928, 57728]
    
    weights_dirs = [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights"),
        os.path.realpath(os.getcwd())
    ]
    
    @classmethod
    def load_weights(cls):
        if cls.weights is not None:
            return
        for weights_dir in cls.weights_dirs:
            try:
                path = os.path.join(weights_dir, cls.weights_file)
                with open(path, "rb") as f:
                    cls.weights = f.read()
                assert len(cls.weights) == cls.weights_filesize
                return
            except IOError:
                pass
        raise Exception(f"Unable to load {cls.weights_file}")
    
    @classmethod
    def weight_at(cls, ptr: int) -> int:
        return struct.unpack_from("<i", cls.weights, ptr * 4)[0]
    
    @classmethod
    def int_bits_to_float(cls, x: int) -> float:
        return struct.unpack("<f", struct.pack("<i", x))[0]
    
    @classmethod
    def get_weight_matrix(cls, neurons: Neurons, window: Window):
        """
        Extract weight matrices for given configuration.
        
        Returns:
            W1: [nns, win_h, win_w] - first projection weights (as conv kernel)
            W2: [nns, win_h, win_w] - second projection weights (as conv kernel)
            WS: [nns, 2] - softmax weights
        """
        cls.load_weights()
        
        nns = neurons.get_neurons()
        win_w = window.get_width()
        win_h = window.get_height()
        window_size = win_w * win_h
        
        offset = cls.weight_offsets[window.value * len(Neurons) + neurons.value]
        
        W1 = np.zeros((nns, win_h, win_w), dtype=np.float32)
        W2 = np.zeros((nns, win_h, win_w), dtype=np.float32)
        WS = np.zeros((nns, 2), dtype=np.float32)
        
        for n in range(nns):
            base = offset + (window_size * 2 + 4) * n
            for y in range(win_h):
                for x in range(win_w):
                    idx = x + y * win_w
                    W1[n, y, x] = cls.int_bits_to_float(cls.weight_at(base + idx))
                    W2[n, y, x] = cls.int_bits_to_float(cls.weight_at(base + window_size + idx))
            WS[n, 0] = cls.int_bits_to_float(cls.weight_at(base + window_size * 2))
            WS[n, 1] = cls.int_bits_to_float(cls.weight_at(base + window_size * 2 + 1))
        
        return W1, W2, WS

class NNEDI3FullImageFast(nn.Module):
    """
    Fast NNEDI3 vertical doubling using pure convolutions.
    
    Instead of extracting windows then doing matrix multiply,
    we use convolutions directly for weight projection.
    """
    
    def __init__(self, neurons: Neurons, window: Window):
        super().__init__()
        
        self.nns = neurons.get_neurons()
        self.win_w = window.get_width()
        self.win_h = window.get_height()
        self.window_size = self.win_w * self.win_h
        
        # Load weights
        W1, W2, WS = NNEDI3Weights.get_weight_matrix(neurons, window)
        
        # W1, W2 are [nns, win_h, win_w], convert to conv weights [nns, 1, win_h, win_w]
        self.register_buffer('conv_w1', torch.from_numpy(W1[:, None, :, :]))
        self.register_buffer('conv_w2', torch.from_numpy(W2[:, None, :, :]))
        self.register_buffer('WS0', torch.from_numpy(WS[:, 0]))
        self.register_buffer('WS1', torch.from_numpy(WS[:, 1]))
        
        # Precompute sum of weights for mean calculation
        # sum_kernel: [1, 1, win_h, win_w] all ones
        sum_kernel = torch.ones(1, 1, self.win_h, self.win_w) / self.window_size
        self.register_buffer('sum_kernel', sum_kernel)
        
        # sq_sum_kernel is the same (just apply to x^2)
        sq_sum_kernel = torch.ones(1, 1, self.win_h, self.win_w) / self.window_size
        self.register_buffer('sq_sum_kernel', sq_sum_kernel)
        
        self.eps = 1.192092896e-7
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Double the image vertically using NNEDI3
        
        Args:
            x: Input image [B, 1, H, W], values in [0, 1]
            
        Returns:
            Output image [B, 1, H*2, W], values in [0, 1]
        """
        B, C, H, W = x.shape
        
        # Padding
        pad_left = self.win_w // 2 - 1
        pad_right = self.win_w - pad_left - 1
        pad_top = self.win_h // 2 - 1
        pad_bottom = self.win_h - pad_top - 1
        
        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        # Compute mean using convolution: [B, 1, H, W]
        mean = F.conv2d(x_pad, self.sum_kernel)
        
        # Compute mean of squares
        sq_mean = F.conv2d(x_pad * x_pad, self.sq_sum_kernel)
        
        # Variance
        var = sq_mean - mean * mean  # [B, 1, H, W]
        
        # mstd2 = 1/sqrt(var) if var >= eps else 0
        mstd2 = torch.where(var >= self.eps, torch.rsqrt(var), torch.zeros_like(var))
        mstd1 = var * mstd2
        
        # Projection using convolution
        # sum1, sum2: [B, nns, H, W]
        sum1 = F.conv2d(x_pad, self.conv_w1)
        sum2 = F.conv2d(x_pad, self.conv_w2)
        
        # Apply normalization: sum * mstd2 + WS
        # mstd2: [B, 1, H, W], WS0: [nns] -> [1, nns, 1, 1]
        sum1 = sum1 * mstd2 + self.WS0.view(1, -1, 1, 1)
        sum2 = sum2 * mstd2 + self.WS1.view(1, -1, 1, 1)
        
        # Numerically stable exp
        sum1_max = sum1.max(dim=1, keepdim=True).values  # [B, 1, H, W]
        sum1 = torch.exp(sum1 - sum1_max)
        
        # Soft activation
        sum2_act = sum2 / (1.0 + torch.abs(sum2))
        
        # Weighted sum: reduce over nns dimension
        wsum = sum1.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        vsum = (sum1 * sum2_act).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Final interpolation
        interp = mean + 5.0 * vsum / wsum * mstd1
        interp = torch.clamp(interp, 0.0, 1.0)
        
        # Interleave original and interpolated rows
        # x: [B, 1, H, W], interp: [B, 1, H, W]
        stacked = torch.stack([x, interp], dim=4)  # [B, 1, H, W, 2]
        output = stacked.permute(0, 1, 2, 4, 3).reshape(B, 1, H * 2, W)
        
        return output

class NNEDI3Upscale2xFast(nn.Module):
    """
    Complete 2x upscaling using NNEDI3 (fast version)
    """
    
    def __init__(self, neurons: Neurons, window: Window):
        super().__init__()
        self.nnedi3 = NNEDI3FullImageFast(neurons, window)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        2x upscale an image
        
        Args:
            x: Input [B, 1, H, W]
            
        Returns:
            Output [B, 1, H*2, W*2]
        """
        # Y-doubling
        x = self.nnedi3(x)  # [B, 1, H*2, W]
        
        # Transpose for X-doubling
        x = x.permute(0, 1, 3, 2)  # [B, 1, W, H*2]
        
        # X-doubling (actually Y-doubling on transposed image)
        x = self.nnedi3(x)  # [B, 1, W*2, H*2]
        
        # Transpose back
        x = x.permute(0, 1, 3, 2)  # [B, 1, H*2, W*2]
        
        return x

def export_onnx(neurons: Neurons, window: Window, output_path: str, opset: int = 17, fp16: bool = False):

    model = NNEDI3Upscale2xFast(neurons, window)
    model.eval()
    
    if fp16:
        model = model.half()
        dummy_input = torch.randn(1, 1, 64, 64, dtype=torch.float16)
    else:
        dummy_input = torch.randn(1, 1, 64, 64)
    
    dynamic_axes = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height_2x', 3: 'width_2x'}
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    try:
        import onnx
        model_proto = onnx.load(output_path)
        onnx.save(model_proto, output_path, save_as_external_data=False)
    except Exception:
        pass
    
    print(f"Exported: {output_path}")
    print(f"  Opset: {opset}")
    print(f"  Precision: {'fp16' if fp16 else 'fp32'}")
    print(f"  Neurons: {neurons.get_neurons()}")
    print(f"  Window: {window.get_width()}x{window.get_height()}")

def test_model(onnx_path: str):
    """Test the exported model"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime to test: pip install onnxruntime")
        return
    
    session = ort.InferenceSession(onnx_path)
    
    input_info = session.get_inputs()[0]
    is_fp16 = input_info.type == 'tensor(float16)'
    
    dtype = np.float16 if is_fp16 else np.float32
    test_input = np.random.rand(1, 1, 32, 32).astype(dtype)
    
    output = session.run(None, {'input': test_input})[0]
    
    print(f"\nTest:")
    print(f"  Input dtype: {dtype.__name__}")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

def main():
    neurons_map = {
        16: Neurons.nns16,
        32: Neurons.nns32,
        64: Neurons.nns64,
        128: Neurons.nns128,
        256: Neurons.nns256
    }
    windows_map = {"8x4": Window.win8x4, "8x6": Window.win8x6}

    parser = argparse.ArgumentParser(description="Export fNNEDI3 2x upscaling model to ONNX (Fast version)")
    parser.add_argument('-n', '--nns', type=int, choices=sorted(neurons_map.keys()),
                        default=32, help='Neurons (default: 32)')
    parser.add_argument('-w', '--win', choices=sorted(windows_map.keys()),
                        default="8x4", help='Window (default: 8x4)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset')
    parser.add_argument('--fp16', action='store_true', help='Export in fp16 precision')
    parser.add_argument('--test', action='store_true', help='Test exported model')

    args = parser.parse_args()
    
    neurons = neurons_map[args.nns]
    window = windows_map[args.win]
    
    if args.output is None:
        suffix = "_fp16" if args.fp16 else ""
        args.output = f"fNNEDI3_nns{args.nns}_win{args.win}{suffix}.onnx"
    
    export_onnx(neurons, window, args.output, args.opset, args.fp16)
    
    if args.test:
        test_model(args.output)

if __name__ == "__main__":
    main()
