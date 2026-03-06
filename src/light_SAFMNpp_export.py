"""https://github.com/sunny2109/SAFMN/tree/main/AIS2024-RTSR
├─ SAFMN-main/
   ├─ scripts/
      ├─ to_onnx/
         ├─ light_SAFMNpp_export.py

"""

import os
import sys
import cv2 
import glob
import numpy as np 
import onnx
import onnxruntime as ort
import torch  
import torch.onnx 
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Try to import SAFMN from basicsr, but it's optional (only needed for SAFMN model conversion)
try:
    from basicsr.archs.safmn_arch import SAFMN
    SAFMN_AVAILABLE = True
except ImportError:
    SAFMN_AVAILABLE = False
    print("Warning: basicsr.archs.safmn_arch not available. Only light_safmnpp conversion is supported.")

class SimpleSAFM_ONNX(nn.Module):
    """SimpleSAFM modified for ONNX export - uses max_pool2d instead of adaptive_max_pool2d"""
    def __init__(self, dim, pool_size=8):
        super().__init__()
        self.pool_size = pool_size
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(dim//2, dim//2, 3, 1, 1, groups=dim//2, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        x0, x1 = self.proj(x).chunk(2, dim=1)

        # Use max_pool2d instead of adaptive_max_pool2d for ONNX compatibility
        x2 = F.max_pool2d(x0, kernel_size=self.pool_size, stride=self.pool_size)
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM_ONNX(nn.Module):
    def __init__(self, dim, ffn_scale):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, int(dim*ffn_scale), 3, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(int(dim*ffn_scale), dim, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class AttBlock_ONNX(nn.Module):
    def __init__(self, dim, ffn_scale, pool_size=8):
        super().__init__()
        self.conv1 = SimpleSAFM_ONNX(dim, pool_size)
        self.conv2 = CCM_ONNX(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class LightSAFMNPP_ONNX(nn.Module):
    """Light SAFMN++ model modified for ONNX export with dynamic input shapes"""
    def __init__(self, dim=8, n_blocks=1, ffn_scale=2.0, upscaling_factor=4, pool_size=8):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        self.feats = nn.Sequential(*[AttBlock_ONNX(dim, ffn_scale, pool_size) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        return self.to_img(x)


def convert_onnx(model, output_folder, output_name, is_dynamic=True, fp16=False): 
    model.eval()
    
    if fp16:
        model = model.half()
        fake_x = torch.rand(1, 3, 256, 256, requires_grad=False).half()
    else:
        fake_x = torch.rand(1, 3, 256, 256, requires_grad=False)
    
    output_path = os.path.join(output_folder, output_name)
    
    dynamic_params = None
    if is_dynamic:
        dynamic_params = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 
            'output': {0: 'batch_size', 2: 'out_height', 3: 'out_width'}
        }

    torch.onnx.export(
        model,                          # model being run 
        fake_x,                         # model input (or a tuple for multiple inputs) 
        output_path,                    # where to save the model  
        export_params=True,             # store the trained parameter weights inside the model file 
        opset_version=18,               # the ONNX version to export the model to 
        do_constant_folding=True,       # whether to execute constant folding for optimization 
        input_names=['input'],          # the model's input names 
        output_names=['output'],        # the model's output names 
        dynamic_axes=dynamic_params
    )

    model_onnx = onnx.load(output_path, load_external_data=True)
    onnx.save(model_onnx, output_path, save_as_external_data=False)

    data_file = output_path + '.data'
    if os.path.exists(data_file):
        os.remove(data_file)

    print(f'Model has been converted to ONNX: {output_path}')


def convert_pt(model, output_folder, output_name): 
    model.eval() 

    fake_x = torch.rand(1, 3, 256, 256, requires_grad=False)
    output_path = os.path.join(output_folder, output_name)

    traced_module = torch.jit.trace(model, fake_x)
    traced_module.save(output_path)
    print(f'Model has been converted to pt: {output_path}')


def load_light_safmnpp_weights(onnx_model, pretrained_path):
    """Load weights from original light_safmnpp.pth to ONNX-compatible model"""
    state_dict = torch.load(pretrained_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    onnx_model.load_state_dict(state_dict, strict=True)
    return onnx_model


def test_onnx(onnx_model, input_path, save_path, target_size=None):
    # for GPU inference
    # ort_session = ort.InferenceSession(onnx_model, ['CUDAExecutionProvider'])

    ort_session = ort.InferenceSession(onnx_model)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(input_path, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]

        print(f'Testing......idx: {idx}, img: {imgname}')

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
        if target_size is not None and img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> CHW
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        output = ort_session.run(None, {"input": img})

        # save image
        print('Saving!')
        output = np.squeeze(output[0], axis=0)
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            
        output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_SAFMN.png'), output)


def convert_light_safmnpp_to_onnx():
    """Convert light_safmnpp.pth to ONNX with dynamic input dimensions"""
    
    # Light SAFMN++ parameters (detected from pretrained weights)
    # to_feat.weight: [32, 3, 3, 3] -> dim=32
    # feats.0, feats.1 -> n_blocks=2
    # conv2.conv.0: [48, 32, 3, 3] -> ffn_scale = 48/32 = 1.5
    # to_img.0: [48, 32, 3, 3] -> 48 = 3 * scale^2, scale = 4
    dim = 32
    n_blocks = 2
    ffn_scale = 1.5
    upscaling_factor = 4
    pool_size = 8  # Original uses h//8, w//8 in adaptive_max_pool2d

    model = LightSAFMNPP_ONNX(
        dim=dim, 
        n_blocks=n_blocks, 
        ffn_scale=ffn_scale, 
        upscaling_factor=upscaling_factor,
        pool_size=pool_size
    )

    pretrained_model = os.path.join(project_root, 'AIS2024-RTSR/pretrained_model/sunny2109_light_safmnpp_pretrain_x4.pth')
    model = load_light_safmnpp_weights(model, pretrained_model)
    model.eval()

    output_folder = os.path.join(project_root, 'scripts/to_onnx/output')
    os.makedirs(output_folder, exist_ok=True)

    convert_onnx(model, output_folder, 'light_safmnpp_dynamic_x4.onnx', is_dynamic=True, fp16=False)

    convert_onnx(model, output_folder, 'light_safmnpp_dynamic_x4_fp16.onnx', is_dynamic=True, fp16=True)
    
    return os.path.join(output_folder, 'light_safmnpp_dynamic_x4.onnx')


def convert_safmn_to_onnx():
    """Convert original SAFMN model to ONNX (fixed dimensions due to adaptive_max_pool2d)"""
    if not SAFMN_AVAILABLE:
        print("Error: SAFMN model requires basicsr package. Please install it first.")
        print("Run: pip install -e . (in the project root)")
        return
    
    model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2) 

    pretrained_model = 'experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2.pth'
    model.load_state_dict(torch.load(pretrained_model)['params'], strict=True)
    
    output_folder = 'scripts/to_onnx/output'
    os.makedirs(output_folder, exist_ok=True)

    convert_onnx(model, output_folder, 'SAFMN_640_960_x2.onnx', is_dynamic=False)
    convert_pt(model, output_folder, 'SAFMN_640_960_x2.pt')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SAFMN models to ONNX')
    parser.add_argument('--model', type=str, default='light_safmnpp', 
                        choices=['light_safmnpp', 'safmn'],
                        help='Model to convert: light_safmnpp or safmn')
    parser.add_argument('--test', action='store_true', help='Test the converted ONNX model')
    parser.add_argument('--input_path', type=str, default='datasets/real_test', 
                        help='Input path for testing')
    parser.add_argument('--save_path', type=str, default='results/onnx_results', 
                        help='Save path for test results')
    
    args = parser.parse_args()
    
    if args.model == 'light_safmnpp':
        print("Converting Light SAFMN++ to ONNX with dynamic dimensions...")
        onnx_model_path = convert_light_safmnpp_to_onnx()
        print(f"\nConversion complete!")
        print(f"ONNX model saved to: {onnx_model_path}")
        print("\nNote: This model supports dynamic input dimensions.")
        print("Input must have height and width divisible by pool_size (8).")
        
        if args.test:
            os.makedirs(args.save_path, exist_ok=True)
            test_onnx(onnx_model_path, args.input_path, args.save_path)
    else:
        print("Converting SAFMN to ONNX with fixed dimensions...")
        convert_safmn_to_onnx()
        
        if args.test:
            onnx_model = 'scripts/to_onnx/output/SAFMN_640_960_x2.onnx'
            os.makedirs(args.save_path, exist_ok=True)
            test_onnx(onnx_model, args.input_path, args.save_path, target_size=(640, 960))



