"""https://github.com/xinntao/Real-ESRGAN
├─ Real-ESRGAN-master/
│  ├─ ESRGAN_SRVGGNET_convert.py
│
├─ esrgan-models/
│  ├─ ESRGAN_SRVGGNET_convert2.py
│
├─ ncnn-20250916/

SRVGGNetCompact (.pth) to mpv GLSL Shader Converter

Usage:
  python ESRGAN_SRVGGNET_convert.py input.pth [-o output.glsl] [-n ModelName] [-t compute|fragment]
"""

import math
import torch
import numpy as np

def load_model(model_path):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Handle different checkpoint formats
    if 'params_ema' in state_dict:
        params = state_dict['params_ema']
    elif 'params' in state_dict:
        params = state_dict['params']
    else:
        params = state_dict
    
    return params

def find_rect(area):
    """Find optimal rectangle dimensions for tiling."""
    s = int(math.sqrt(area))
    for w in range(s, 0, -1):
        if area % w == 0:
            h = area // w
            return int(max(w, h)), int(min(w, h))
    return area, 1

def get_tile_off(i, w):
    x = i % w
    y = i // w
    return x, y

def fmt(val, precision=8):
    return f"{val:.{precision}g}"

def fmt_vec(arr, precision=8):
    return ", ".join(fmt(v, precision) for v in arr.flatten())

def generate_header(model_name):
    return ""

COMPUTE_HEADER = """
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#ifdef GL_EXT_shader_explicit_arithmetic_types_float16
#	define V4 f16vec4
#	define M4 f16mat4
#	define F float16_t
#else
#	define V4 vec4
#	define M4 mat4
#	define F float
#endif

"""

FRAGMENT_HEADER = """
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#ifdef GL_EXT_shader_explicit_arithmetic_types_float16
#	define V4 f16vec4
#	define M4 f16mat4
#	define F float16_t
#else
#	define V4 vec4
#	define M4 mat4
#	define F float
#endif

"""

# ============================================================================
# COMPUTE SHADER GENERATORS (Original)
# ============================================================================

def generate_first_conv(weights, biases, prelu_slopes, layer_name, num_feat, model_name, upscale=2):
    """Generate GLSL for first convolution (3 -> num_feat channels) with PReLU."""
    
    # weights shape: [num_feat, 3, 3, 3] -> [out_ch, in_ch, kH, kW]
    passes_out = int(np.ceil(num_feat / 4.0))
    tile_w, tile_h = find_rect(passes_out)
    
    threads_w, threads_h = 8, 8
    
    code = "//!HOOK MAIN\n"
    code += "//!BIND HOOKED\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}+PReLU\n"
    code += f"//!WIDTH HOOKED.w {float(tile_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_h)} *\n"
    if upscale > 1:
        code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += f"//!COMPUTE {threads_w * tile_w} {threads_h * tile_h} {threads_w} {threads_h}\n"
    code += COMPUTE_HEADER
    
    code += "const ivec2 ksize = ivec2(3, 3);\n"
    code += "const ivec2 offset = ksize / 2;\n"
    code += "const ivec2 wg_size = ivec2(gl_WorkGroupSize);\n"
    code += "const ivec2 isize = wg_size + ksize - 1;\n"
    code += "shared V4 inp[isize.y][isize.x];\n"  # RGB input as V4 (only xyz used)
    code += "void hook() {\n"
    code += "    const uvec2 local_xy = gl_LocalInvocationID.xy;\n"
    code += "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;\n"
    code += "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {\n"
    code += "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {\n"
    code += "            ivec2 input_pos = base + ivec2(x, y) - offset;\n"
    code += "            inp[y][x] = V4(HOOKED_mul * texelFetch(HOOKED_raw, input_pos, 0));\n"
    code += "        }\n"
    code += "    }\n"
    code += "    barrier();\n\n"
    
    # Initialize results with biases
    for p in range(passes_out):
        bias_slice = biases[p*4:min((p+1)*4, num_feat)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        code += f"    V4 res{p} = V4({fmt_vec(bias_slice)});\n"
    
    # Load kernel positions
    code += "\n"
    for ky in range(3):
        for kx in range(3):
            code += f"    V4 i_{kx}_{ky} = inp[local_xy.y + {ky}][local_xy.x + {kx}];\n"
    
    code += "\n"
    # Convolution: for each output pass
    for p in range(passes_out):
        out_start = p * 4
        out_end = min((p + 1) * 4, num_feat)
        out_size = out_end - out_start
        
        for ky in range(3):
            for kx in range(3):
                # weights[out_ch, in_ch=3, kH, kW] 
                # We need mat4 for 4 output channels x 4 input channels (padded)
                w_slice = weights[out_start:out_end, :, ky, kx]  # [out_size, 3]
                
                # Pad to 4x4 and TRANSPOSE for GLSL column-major M4 * V4
                # In GLSL: result = M * v, where result[i] = dot(M[i], v) = sum(M[j][i] * v[j])
                # M4(a,b,c,d, e,f,g,h, ...) creates columns: col0=(a,b,c,d), col1=(e,f,g,h), ...
                # So M[j][i] = element at column j, row i
                # We want: result[out] = sum(weight[out,in] * input[in])
                # So M[in][out] = weight[out,in], meaning M = weight.T
                w_padded = np.zeros((4, 4), dtype=np.float32)
                w_padded[:out_size, :3] = w_slice
                w_transposed = w_padded.T  # Now [in, out] order for GLSL
                
                code += f"    res{p} += M4({fmt_vec(w_transposed)}) * i_{kx}_{ky};\n"
    
    # Apply PReLU and store
    code += f"\n    ivec2 obase = ivec2(gl_GlobalInvocationID) * ivec2({tile_w}, {tile_h});\n"
    
    for p in range(passes_out):
        tx, ty = get_tile_off(p, tile_w)
        slope_slice = prelu_slopes[p*4:min((p+1)*4, num_feat)]
        if len(slope_slice) < 4:
            slope_slice = np.pad(slope_slice, (0, 4 - len(slope_slice)), constant_values=0.25)
        code += f"    V4 s{p} = V4({fmt_vec(slope_slice)});\n"
        code += f"    res{p} = max(res{p}, V4(0.0)) + s{p} * min(res{p}, V4(0.0));\n"
        code += f"    imageStore(out_image, obase + ivec2({tx}, {ty}), res{p});\n"
    
    code += "}\n\n"
    return code

def generate_mid_conv(weights, biases, prelu_slopes, layer_name, prev_layer, num_feat, model_name, upscale=2):
    """Generate GLSL for middle convolution (num_feat -> num_feat) with PReLU."""
    
    passes = int(np.ceil(num_feat / 4.0))
    tile_w, tile_h = find_rect(passes)
    
    threads_w, threads_h = 8, 8
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}+PReLU\n"
    code += f"//!WIDTH HOOKED.w {float(tile_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_h)} *\n"
    if upscale > 1:
        code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += f"//!COMPUTE {threads_w * tile_w} {threads_h * tile_h} {threads_w} {threads_h}\n"
    code += COMPUTE_HEADER
    
    code += "const ivec2 ksize = ivec2(3, 3);\n"
    code += "const ivec2 offset = ksize / 2;\n"
    code += "const ivec2 wg_size = ivec2(gl_WorkGroupSize);\n"
    code += "const ivec2 isize = wg_size + ksize - 1;\n"
    code += f"shared V4 inp[{passes}][isize.y][isize.x];\n"
    code += "void hook() {\n"
    code += "    const uvec2 local_xy = gl_LocalInvocationID.xy;\n"
    code += "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;\n"
    code += "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {\n"
    code += "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {\n"
    code += f"            ivec2 ipos = (base + ivec2(x, y) - offset) * ivec2({tile_w}, {tile_h});\n"
    
    for z in range(passes):
        tx, ty = get_tile_off(z, tile_w)
        code += f"            inp[{z}][y][x] = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, ipos + ivec2({tx}, {ty}), 0));\n"
    
    code += "        }\n"
    code += "    }\n"
    code += "    barrier();\n\n"
    
    # Initialize with biases
    for p in range(passes):
        bias_slice = biases[p*4:min((p+1)*4, num_feat)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        code += f"    V4 res{p} = V4({fmt_vec(bias_slice)});\n"
    
    # For each input tile
    for z in range(passes):
        code += f"\n"
        for ky in range(3):
            for kx in range(3):
                code += f"    V4 i{z}_{kx}_{ky} = inp[{z}][local_xy.y + {ky}][local_xy.x + {kx}];\n"
        
        in_start = z * 4
        in_end = min((z + 1) * 4, num_feat)
        in_size = in_end - in_start
        
        for p in range(passes):
            out_start = p * 4
            out_end = min((p + 1) * 4, num_feat)
            out_size = out_end - out_start
            
            for ky in range(3):
                for kx in range(3):
                    # weights[out_ch, in_ch, kH, kW]
                    w_slice = weights[out_start:out_end, in_start:in_end, ky, kx]
                    
                    # Pad and TRANSPOSE for GLSL column-major M4 * V4
                    w_padded = np.zeros((4, 4), dtype=np.float32)
                    w_padded[:out_size, :in_size] = w_slice
                    w_transposed = w_padded.T  # Transpose for GLSL
                    
                    code += f"    res{p} += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
    
    # PReLU and store
    code += f"\n    ivec2 obase = ivec2(gl_GlobalInvocationID) * ivec2({tile_w}, {tile_h});\n"
    
    for p in range(passes):
        tx, ty = get_tile_off(p, tile_w)
        slope_slice = prelu_slopes[p*4:min((p+1)*4, num_feat)]
        if len(slope_slice) < 4:
            slope_slice = np.pad(slope_slice, (0, 4 - len(slope_slice)), constant_values=0.25)
        code += f"    V4 s{p} = V4({fmt_vec(slope_slice)});\n"
        code += f"    res{p} = max(res{p}, V4(0.0)) + s{p} * min(res{p}, V4(0.0));\n"
        code += f"    imageStore(out_image, obase + ivec2({tx}, {ty}), res{p});\n"
    
    code += "}\n\n"
    return code

def generate_last_conv(weights, biases, layer_name, prev_layer, num_feat, out_channels, model_name):
    """Generate GLSL for last convolution (num_feat -> out_channels for PixelShuffle)."""
    
    passes_in = int(np.ceil(num_feat / 4.0))
    passes_out = int(np.ceil(out_channels / 4.0))
    tile_in_w, tile_in_h = find_rect(passes_in)
    tile_out_w, tile_out_h = find_rect(passes_out)
    
    threads_w, threads_h = 8, 8
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}\n"
    code += f"//!WIDTH HOOKED.w {float(tile_out_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_out_h)} *\n"
    code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += f"//!COMPUTE {threads_w * tile_out_w} {threads_h * tile_out_h} {threads_w} {threads_h}\n"
    code += COMPUTE_HEADER
    
    code += "const ivec2 ksize = ivec2(3, 3);\n"
    code += "const ivec2 offset = ksize / 2;\n"
    code += "const ivec2 wg_size = ivec2(gl_WorkGroupSize);\n"
    code += "const ivec2 isize = wg_size + ksize - 1;\n"
    code += f"shared V4 inp[{passes_in}][isize.y][isize.x];\n"
    code += "void hook() {\n"
    code += "    const uvec2 local_xy = gl_LocalInvocationID.xy;\n"
    code += "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;\n"
    code += "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {\n"
    code += "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {\n"
    code += f"            ivec2 ipos = (base + ivec2(x, y) - offset) * ivec2({tile_in_w}, {tile_in_h});\n"
    
    for z in range(passes_in):
        tx, ty = get_tile_off(z, tile_in_w)
        code += f"            inp[{z}][y][x] = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, ipos + ivec2({tx}, {ty}), 0));\n"
    
    code += "        }\n"
    code += "    }\n"
    code += "    barrier();\n\n"
    
    # Initialize
    for p in range(passes_out):
        bias_slice = biases[p*4:min((p+1)*4, out_channels)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        code += f"    V4 res{p} = V4({fmt_vec(bias_slice)});\n"
    
    # Convolution
    for z in range(passes_in):
        code += f"\n"
        for ky in range(3):
            for kx in range(3):
                code += f"    V4 i{z}_{kx}_{ky} = inp[{z}][local_xy.y + {ky}][local_xy.x + {kx}];\n"
        
        in_start = z * 4
        in_end = min((z + 1) * 4, num_feat)
        in_size = in_end - in_start
        
        for p in range(passes_out):
            out_start = p * 4
            out_end = min((p + 1) * 4, out_channels)
            out_size = out_end - out_start
            
            for ky in range(3):
                for kx in range(3):
                    w_slice = weights[out_start:out_end, in_start:in_end, ky, kx]
                    # Pad and TRANSPOSE for GLSL column-major M4 * V4
                    w_padded = np.zeros((4, 4), dtype=np.float32)
                    w_padded[:out_size, :in_size] = w_slice
                    w_transposed = w_padded.T  # Transpose for GLSL
                    code += f"    res{p} += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
    
    # Store (no activation on last conv)
    code += f"\n    ivec2 obase = ivec2(gl_GlobalInvocationID) * ivec2({tile_out_w}, {tile_out_h});\n"
    for p in range(passes_out):
        tx, ty = get_tile_off(p, tile_out_w)
        code += f"    imageStore(out_image, obase + ivec2({tx}, {ty}), res{p});\n"
    
    code += "}\n\n"
    return code

def generate_last_conv_x1(weights, biases, layer_name, prev_layer, num_feat, out_channels, model_name):
    """Generate GLSL for last convolution of x1 models (num_feat -> 3 RGB channels, no tiling)."""
    
    passes_in = int(np.ceil(num_feat / 4.0))
    tile_in_w, tile_in_h = find_rect(passes_in)
    
    threads_w, threads_h = 8, 8
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}\n"
    code += "//!COMPONENTS 4\n"
    code += f"//!COMPUTE {threads_w} {threads_h} {threads_w} {threads_h}\n"
    code += COMPUTE_HEADER
    
    code += "const ivec2 ksize = ivec2(3, 3);\n"
    code += "const ivec2 offset = ksize / 2;\n"
    code += "const ivec2 wg_size = ivec2(gl_WorkGroupSize);\n"
    code += "const ivec2 isize = wg_size + ksize - 1;\n"
    code += f"shared V4 inp[{passes_in}][isize.y][isize.x];\n"
    code += "void hook() {\n"
    code += "    const uvec2 local_xy = gl_LocalInvocationID.xy;\n"
    code += "    ivec2 base = ivec2(gl_WorkGroupID) * wg_size;\n"
    code += "    for (uint y = local_xy.y; y < isize.y; y += wg_size.y) {\n"
    code += "        for (uint x = local_xy.x; x < isize.x; x += wg_size.x) {\n"
    code += f"            ivec2 ipos = (base + ivec2(x, y) - offset) * ivec2({tile_in_w}, {tile_in_h});\n"
    
    for z in range(passes_in):
        tx, ty = get_tile_off(z, tile_in_w)
        code += f"            inp[{z}][y][x] = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, ipos + ivec2({tx}, {ty}), 0));\n"
    
    code += "        }\n"
    code += "    }\n"
    code += "    barrier();\n\n"
    
    # Initialize with biases - only 3 output channels (RGB), pad to vec4
    bias_padded = np.zeros(4, dtype=np.float32)
    bias_padded[:min(3, out_channels)] = biases[:min(3, out_channels)]
    code += f"    V4 res = V4({fmt_vec(bias_padded)});\n"
    
    # Convolution
    for z in range(passes_in):
        code += f"\n"
        for ky in range(3):
            for kx in range(3):
                code += f"    V4 i{z}_{kx}_{ky} = inp[{z}][local_xy.y + {ky}][local_xy.x + {kx}];\n"
        
        in_start = z * 4
        in_end = min((z + 1) * 4, num_feat)
        in_size = in_end - in_start
        
        for ky in range(3):
            for kx in range(3):
                # weights[out_ch=3, in_ch, kH, kW] -> output to RGB
                w_slice = weights[:min(3, out_channels), in_start:in_end, ky, kx]  # [3, in_size]
                # Pad and TRANSPOSE for GLSL column-major M4 * V4
                w_padded = np.zeros((4, 4), dtype=np.float32)
                w_padded[:w_slice.shape[0], :in_size] = w_slice
                w_transposed = w_padded.T  # Transpose for GLSL
                code += f"    res += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
    
    # Store (no activation on last conv, output is single pixel with RGB in xyz)
    code += "\n    imageStore(out_image, ivec2(gl_GlobalInvocationID), res);\n"
    
    code += "}\n\n"
    return code

def generate_output_x1(prev_layer, num_out_ch, model_name):
    """Generate output pass for x1 models (no upscale, just residual)."""
    
    # For x1 models, last conv outputs 3 channels directly (no PixelShuffle)
    # Just add residual and output
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += "//!BIND HOOKED\n"
    code += f"//!DESC [{model_name}] Output + Residual\n"
    code += "//!COMPUTE 32 32 32 32\n"
    
    code += """
void hook() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec4 orig = texelFetch(HOOKED_raw, pos, 0) * HOOKED_mul;
    vec4 conv_out = """ + prev_layer + """_mul * texelFetch(""" + prev_layer + """_raw, pos, 0);
    vec4 result;
    result.rgb = conv_out.rgb + orig.rgb;
    result.a = orig.a;
    imageStore(out_image, pos, clamp(result, 0.0, 1.0));
}

"""
    return code

def generate_pixelshuffle(prev_layer, upscale, num_out_ch, model_name):
    """Generate PixelShuffle + residual connection pass."""
    
    # PyTorch PixelShuffle for upscale=2, num_out_ch=3:
    # Input: 12 channels -> Output: 3 channels at 2x resolution
    # Channel order: [R00, R01, R10, R11, G00, G01, G10, G11, B00, B01, B10, B11]
    # where Cyx = channel C at position (y,x) in the 2x2 block
    #
    # Our texture packing: 3 vec4 tiles in a texture of size (HOOKED.w*3, HOOKED.h)
    # For original pixel at (px, py), the tiles are stored at:
    # - tile0: (px*3 + 0, py) contains channels [0,1,2,3]   = [R00, R01, R10, R11]
    # - tile1: (px*3 + 1, py) contains channels [4,5,6,7]   = [G00, G01, G10, G11]
    # - tile2: (px*3 + 2, py) contains channels [8,9,10,11] = [B00, B01, B10, B11]
    #
    # For high-res position (hx, hy), we need:
    # - Low-res position: (lx, ly) = (floor(hx/2), floor(hy/2))
    # - Position in 2x2 block: (sx, sy) = (hx mod 2, hy mod 2)
    # - Channel index: idx = sy * 2 + sx
    # - Texture pixel for tile t: (lx * 3 + t, ly)
    
    passes = int(np.ceil(num_out_ch * upscale * upscale / 4.0))
    tile_w, tile_h = find_rect(passes)
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += "//!BIND HOOKED\n"
    code += f"//!DESC [{model_name}] PixelShuffle x{upscale} + Residual\n"
    code += f"//!WIDTH HOOKED.w {upscale} *\n"
    code += f"//!HEIGHT HOOKED.h {upscale} *\n"
    code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPUTE 32 32 32 32\n"
    
    if upscale == 2 and num_out_ch == 3:
        # PixelShuffle 2x: 12 channels -> 3 RGB at 2x resolution
        # conv9 texture is (HOOKED.w * 3, HOOKED.h * 1) with 3 vec4 tiles per pixel
        # Output is (HOOKED.w * 2, HOOKED.h * 2)
        code += f"""
void hook() {{
    ivec2 opos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 ipos = opos / 2;
    ivec2 sub = opos % 2;
    int idx = sub.y * 2 + sub.x;
    ivec2 base_pixel = ipos * ivec2({tile_w}, {tile_h});
    vec4 orig = texelFetch(HOOKED_raw, ipos, 0) * HOOKED_mul;
    vec4 tileR = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(0, 0), 0);
    vec4 tileG = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(1, 0), 0);
    vec4 tileB = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(2, 0), 0);
    vec4 result;
    result.r = tileR[idx];
    result.g = tileG[idx];
    result.b = tileB[idx];
    result.a = orig.a;
    result.rgb += orig.rgb;
    imageStore(out_image, opos, clamp(result, 0.0, 1.0));
}}

"""
    else:
        code += f"""
void hook() {{
    ivec2 opos = ivec2(gl_GlobalInvocationID.xy);
    vec4 result = vec4(0.0, 0.0, 0.0, 1.0);
    result.rgb += texelFetch(HOOKED_raw, opos, 0).rgb * HOOKED_mul;
    imageStore(out_image, opos, clamp(result, 0.0, 1.0));
}}

"""
    
    return code

# ============================================================================
# FRAGMENT SHADER GENERATORS (New)
# ============================================================================

def generate_first_conv_frag(weights, biases, prelu_slopes, layer_name, num_feat, model_name, upscale=2):
    """Generate fragment shader for first convolution (3 -> num_feat channels) with PReLU."""
    
    passes_out = int(np.ceil(num_feat / 4.0))
    tile_w, tile_h = find_rect(passes_out)
    
    code = "//!HOOK MAIN\n"
    code += "//!BIND HOOKED\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}+PReLU\n"
    code += f"//!WIDTH HOOKED.w {float(tile_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_h)} *\n"
    if upscale > 1:
        code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += FRAGMENT_HEADER
    
    code += "vec4 hook() {\n"
    # Compute which tile and pixel we're rendering
    code += f"    ivec2 opos = ivec2(gl_FragCoord.xy);\n"
    code += f"    int tile_idx = (opos.x % {tile_w}) + (opos.y % {tile_h}) * {tile_w};\n"
    code += f"    ivec2 ipos = ivec2(opos.x / {tile_w}, opos.y / {tile_h});\n"
    code += "\n"
    
    # Sample 3x3 input neighborhood
    code += "    // Sample 3x3 input neighborhood\n"
    for ky in range(3):
        for kx in range(3):
            ox, oy = kx - 1, ky - 1
            code += f"    V4 i_{kx}_{ky} = V4(HOOKED_texOff(vec2({ox}, {oy})));\n"
    code += "\n"
    
    # Output each tile pass
    for p in range(passes_out):
        out_start = p * 4
        out_end = min((p + 1) * 4, num_feat)
        out_size = out_end - out_start
        
        bias_slice = biases[p*4:min((p+1)*4, num_feat)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        
        code += f"    if (tile_idx == {p}) {{\n"
        code += f"        V4 res = V4({fmt_vec(bias_slice)});\n"
        
        for ky in range(3):
            for kx in range(3):
                w_slice = weights[out_start:out_end, :, ky, kx]
                w_padded = np.zeros((4, 4), dtype=np.float32)
                w_padded[:out_size, :3] = w_slice
                w_transposed = w_padded.T
                code += f"        res += M4({fmt_vec(w_transposed)}) * i_{kx}_{ky};\n"
        
        slope_slice = prelu_slopes[p*4:min((p+1)*4, num_feat)]
        if len(slope_slice) < 4:
            slope_slice = np.pad(slope_slice, (0, 4 - len(slope_slice)), constant_values=0.25)
        code += f"        V4 s = V4({fmt_vec(slope_slice)});\n"
        code += "        res = max(res, V4(0.0)) + s * min(res, V4(0.0));\n"
        code += "        return vec4(res);\n"
        code += "    }\n"
    
    code += "    return vec4(0.0);\n"
    code += "}\n\n"
    return code

def generate_mid_conv_frag(weights, biases, prelu_slopes, layer_name, prev_layer, num_feat, model_name, upscale=2):
    """Generate fragment shader for middle convolution (num_feat -> num_feat) with PReLU."""
    
    passes = int(np.ceil(num_feat / 4.0))
    tile_w, tile_h = find_rect(passes)
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}+PReLU\n"
    code += f"//!WIDTH HOOKED.w {float(tile_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_h)} *\n"
    if upscale > 1:
        code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += FRAGMENT_HEADER
    
    code += "vec4 hook() {\n"
    code += f"    ivec2 opos = ivec2(gl_FragCoord.xy);\n"
    code += f"    int tile_idx = (opos.x % {tile_w}) + (opos.y % {tile_h}) * {tile_w};\n"
    code += f"    ivec2 base_pos = ivec2(opos.x / {tile_w}, opos.y / {tile_h});\n"
    code += "\n"
    
    # Sample all input tiles for 3x3 neighborhood
    code += "    // Sample all input tiles for 3x3 neighborhood\n"
    for z in range(passes):
        tx, ty = get_tile_off(z, tile_w)
        for ky in range(3):
            for kx in range(3):
                ox, oy = kx - 1, ky - 1
                code += f"    V4 i{z}_{kx}_{ky} = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, (base_pos + ivec2({ox}, {oy})) * ivec2({tile_w}, {tile_h}) + ivec2({tx}, {ty}), 0));\n"
    code += "\n"
    
    # Output each tile pass
    for p in range(passes):
        out_start = p * 4
        out_end = min((p + 1) * 4, num_feat)
        out_size = out_end - out_start
        
        bias_slice = biases[p*4:min((p+1)*4, num_feat)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        
        code += f"    if (tile_idx == {p}) {{\n"
        code += f"        V4 res = V4({fmt_vec(bias_slice)});\n"
        
        for z in range(passes):
            in_start = z * 4
            in_end = min((z + 1) * 4, num_feat)
            in_size = in_end - in_start
            
            for ky in range(3):
                for kx in range(3):
                    w_slice = weights[out_start:out_end, in_start:in_end, ky, kx]
                    w_padded = np.zeros((4, 4), dtype=np.float32)
                    w_padded[:out_size, :in_size] = w_slice
                    w_transposed = w_padded.T
                    code += f"        res += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
        
        slope_slice = prelu_slopes[p*4:min((p+1)*4, num_feat)]
        if len(slope_slice) < 4:
            slope_slice = np.pad(slope_slice, (0, 4 - len(slope_slice)), constant_values=0.25)
        code += f"        V4 s = V4({fmt_vec(slope_slice)});\n"
        code += "        res = max(res, V4(0.0)) + s * min(res, V4(0.0));\n"
        code += "        return vec4(res);\n"
        code += "    }\n"
    
    code += "    return vec4(0.0);\n"
    code += "}\n\n"
    return code

def generate_last_conv_frag(weights, biases, layer_name, prev_layer, num_feat, out_channels, model_name):
    """Generate fragment shader for last convolution (num_feat -> out_channels for PixelShuffle)."""
    
    passes_in = int(np.ceil(num_feat / 4.0))
    passes_out = int(np.ceil(out_channels / 4.0))
    tile_in_w, tile_in_h = find_rect(passes_in)
    tile_out_w, tile_out_h = find_rect(passes_out)
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}\n"
    code += f"//!WIDTH HOOKED.w {float(tile_out_w)} *\n"
    code += f"//!HEIGHT HOOKED.h {float(tile_out_h)} *\n"
    code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    code += "//!COMPONENTS 4\n"
    code += FRAGMENT_HEADER
    
    code += "vec4 hook() {\n"
    code += f"    ivec2 opos = ivec2(gl_FragCoord.xy);\n"
    code += f"    int tile_idx = (opos.x % {tile_out_w}) + (opos.y % {tile_out_h}) * {tile_out_w};\n"
    code += f"    ivec2 base_pos = ivec2(opos.x / {tile_out_w}, opos.y / {tile_out_h});\n"
    code += "\n"
    
    # Sample all input tiles for 3x3 neighborhood
    code += "    // Sample all input tiles for 3x3 neighborhood\n"
    for z in range(passes_in):
        tx, ty = get_tile_off(z, tile_in_w)
        for ky in range(3):
            for kx in range(3):
                ox, oy = kx - 1, ky - 1
                code += f"    V4 i{z}_{kx}_{ky} = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, (base_pos + ivec2({ox}, {oy})) * ivec2({tile_in_w}, {tile_in_h}) + ivec2({tx}, {ty}), 0));\n"
    code += "\n"
    
    # Output each tile pass
    for p in range(passes_out):
        out_start = p * 4
        out_end = min((p + 1) * 4, out_channels)
        out_size = out_end - out_start
        
        bias_slice = biases[p*4:min((p+1)*4, out_channels)]
        if len(bias_slice) < 4:
            bias_slice = np.pad(bias_slice, (0, 4 - len(bias_slice)))
        
        code += f"    if (tile_idx == {p}) {{\n"
        code += f"        V4 res = V4({fmt_vec(bias_slice)});\n"
        
        for z in range(passes_in):
            in_start = z * 4
            in_end = min((z + 1) * 4, num_feat)
            in_size = in_end - in_start
            
            for ky in range(3):
                for kx in range(3):
                    w_slice = weights[out_start:out_end, in_start:in_end, ky, kx]
                    w_padded = np.zeros((4, 4), dtype=np.float32)
                    w_padded[:out_size, :in_size] = w_slice
                    w_transposed = w_padded.T
                    code += f"        res += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
        
        code += "        return vec4(res);\n"
        code += "    }\n"
    
    code += "    return vec4(0.0);\n"
    code += "}\n\n"
    return code

def generate_last_conv_x1_frag(weights, biases, layer_name, prev_layer, num_feat, out_channels, model_name):
    """Generate fragment shader for last convolution of x1 models (num_feat -> 3 RGB channels, no tiling)."""
    
    passes_in = int(np.ceil(num_feat / 4.0))
    tile_in_w, tile_in_h = find_rect(passes_in)
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += f"//!SAVE {layer_name}\n"
    code += f"//!DESC [{model_name}] {layer_name}\n"
    code += "//!COMPONENTS 4\n"
    code += FRAGMENT_HEADER
    
    code += "vec4 hook() {\n"
    code += f"    ivec2 ipos = ivec2(gl_FragCoord.xy);\n"
    code += "\n"
    
    # Sample all input tiles for 3x3 neighborhood
    code += "    // Sample all input tiles for 3x3 neighborhood\n"
    for z in range(passes_in):
        tx, ty = get_tile_off(z, tile_in_w)
        for ky in range(3):
            for kx in range(3):
                ox, oy = kx - 1, ky - 1
                code += f"    V4 i{z}_{kx}_{ky} = V4({prev_layer}_mul * texelFetch({prev_layer}_raw, (ipos + ivec2({ox}, {oy})) * ivec2({tile_in_w}, {tile_in_h}) + ivec2({tx}, {ty}), 0));\n"
    code += "\n"
    
    # Initialize with biases - only 3 output channels (RGB), pad to vec4
    bias_padded = np.zeros(4, dtype=np.float32)
    bias_padded[:min(3, out_channels)] = biases[:min(3, out_channels)]
    code += f"    V4 res = V4({fmt_vec(bias_padded)});\n"
    
    # Convolution
    for z in range(passes_in):
        in_start = z * 4
        in_end = min((z + 1) * 4, num_feat)
        in_size = in_end - in_start
        
        for ky in range(3):
            for kx in range(3):
                w_slice = weights[:min(3, out_channels), in_start:in_end, ky, kx]
                w_padded = np.zeros((4, 4), dtype=np.float32)
                w_padded[:w_slice.shape[0], :in_size] = w_slice
                w_transposed = w_padded.T
                code += f"    res += M4({fmt_vec(w_transposed)}) * i{z}_{kx}_{ky};\n"
    
    code += "    return vec4(res);\n"
    code += "}\n\n"
    return code

def generate_output_x1_frag(prev_layer, num_out_ch, model_name):
    """Generate fragment shader output pass for x1 models (no upscale, just residual)."""
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += "//!BIND HOOKED\n"
    code += f"//!DESC [{model_name}] Output + Residual\n"
    
    code += """
vec4 hook() {
    vec4 orig = HOOKED_texOff(0);
    vec4 conv_out = """ + prev_layer + """_texOff(0);
    vec4 result;
    result.rgb = conv_out.rgb + orig.rgb;
    result.a = orig.a;
    return clamp(result, 0.0, 1.0);
}

"""
    return code

def generate_pixelshuffle_frag(prev_layer, upscale, num_out_ch, model_name):
    """Generate fragment shader PixelShuffle + residual connection pass."""
    
    passes = int(np.ceil(num_out_ch * upscale * upscale / 4.0))
    tile_w, tile_h = find_rect(passes)
    
    code = "//!HOOK MAIN\n"
    code += f"//!BIND {prev_layer}\n"
    code += "//!BIND HOOKED\n"
    code += f"//!DESC [{model_name}] PixelShuffle x{upscale} + Residual\n"
    code += f"//!WIDTH HOOKED.w {upscale} *\n"
    code += f"//!HEIGHT HOOKED.h {upscale} *\n"
    code += "//!WHEN OUTPUT.w HOOKED.w 1.200 * > OUTPUT.h HOOKED.h 1.200 * > *\n"
    
    if upscale == 2 and num_out_ch == 3:
        code += f"""
vec4 hook() {{
    ivec2 opos = ivec2(gl_FragCoord.xy);
    ivec2 ipos = opos / 2;
    ivec2 sub = opos % 2;
    int idx = sub.y * 2 + sub.x;
    ivec2 base_pixel = ipos * ivec2({tile_w}, {tile_h});
    vec4 orig = texelFetch(HOOKED_raw, ipos, 0) * HOOKED_mul;
    vec4 tileR = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(0, 0), 0);
    vec4 tileG = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(1, 0), 0);
    vec4 tileB = {prev_layer}_mul * texelFetch({prev_layer}_raw, base_pixel + ivec2(2, 0), 0);
    vec4 result;
    result.r = tileR[idx];
    result.g = tileG[idx];
    result.b = tileB[idx];
    result.a = orig.a;
    result.rgb += orig.rgb;
    return clamp(result, 0.0, 1.0);
}}

"""
    else:
        code += f"""
vec4 hook() {{
    return HOOKED_texOff(0);
}}

"""
    
    return code

# ============================================================================
# MAIN CONVERSION FUNCTIONS
# ============================================================================

def convert_model(model_path, output_path, model_name="SuperUltraCompact", shader_type="compute"):
    """Main conversion function."""
    
    print(f"Loading model: {model_path}")
    params = load_model(model_path)
    
    # Analyze model structure
    # body.0: first conv (3 -> num_feat)
    # body.1: first PReLU
    # body.2, 4, 6, ...: middle convs (num_feat -> num_feat)
    # body.3, 5, 7, ...: PReLUs
    # body.N: last conv (num_feat -> num_out_ch * upscale^2)
    
    first_conv_w = params['body.0.weight'].numpy()
    first_conv_b = params['body.0.bias'].numpy()
    first_prelu = params['body.1.weight'].numpy()
    
    num_feat = first_conv_w.shape[0]  # 24
    num_in_ch = first_conv_w.shape[1]  # 3
    
    # Find last conv
    max_idx = 0
    for key in params.keys():
        if 'body.' in key and '.weight' in key:
            idx = int(key.split('.')[1])
            max_idx = max(max_idx, idx)
    
    last_conv_w = params[f'body.{max_idx - 1}.weight'].numpy() if f'body.{max_idx - 1}.weight' in params else None
    
    # Find the actual last conv (4D weight)
    for idx in range(max_idx, -1, -1):
        key = f'body.{idx}.weight'
        if key in params and len(params[key].shape) == 4:
            last_conv_w = params[key].numpy()
            last_conv_b = params[f'body.{idx}.bias'].numpy()
            last_conv_idx = idx
            break
    
    out_channels = last_conv_w.shape[0]  # 12 for 3 channels * 2^2, or 3 for x1 models
    
    # Determine upscale
    upscale = 1  # default to x1
    for scale in [2, 3, 4]:
        if out_channels == num_in_ch * scale * scale:
            upscale = scale
            break
    
    num_out_ch = out_channels // (upscale * upscale) if upscale > 1 else out_channels
    
    print(f"Model parameters:")
    print(f"  num_in_ch: {num_in_ch}")
    print(f"  num_out_ch: {num_out_ch}")
    print(f"  num_feat: {num_feat}")
    print(f"  upscale: {upscale}x")
    print(f"  last_conv_idx: {last_conv_idx}")
    print(f"  shader_type: {shader_type}")
    
    # Generate shader
    shader = generate_header(model_name)
    
    if shader_type == "compute":
        # First conv + PReLU
        shader += generate_first_conv(first_conv_w, first_conv_b, first_prelu, "conv0", num_feat, model_name, upscale)
        
        prev_layer = "conv0"
        conv_idx = 1
        
        # Middle convs
        for body_idx in range(2, last_conv_idx, 2):  # Skip by 2 (conv, prelu pairs)
            conv_key = f'body.{body_idx}.weight'
            prelu_key = f'body.{body_idx + 1}.weight'
            
            if conv_key not in params:
                break
            
            conv_w = params[conv_key].numpy()
            conv_b = params[f'body.{body_idx}.bias'].numpy()
            prelu = params[prelu_key].numpy() if prelu_key in params else np.ones(num_feat) * 0.25
            
            layer_name = f"conv{conv_idx}"
            shader += generate_mid_conv(conv_w, conv_b, prelu, layer_name, prev_layer, num_feat, model_name, upscale)
            
            prev_layer = layer_name
            conv_idx += 1
        
        # Last conv (no activation)
        if upscale == 1:
            shader += generate_last_conv_x1(last_conv_w, last_conv_b, f"conv{conv_idx}", prev_layer, num_feat, out_channels, model_name)
            prev_layer = f"conv{conv_idx}"
            shader += generate_output_x1(prev_layer, num_out_ch, model_name)
        else:
            shader += generate_last_conv(last_conv_w, last_conv_b, f"conv{conv_idx}", prev_layer, num_feat, out_channels, model_name)
            prev_layer = f"conv{conv_idx}"
            shader += generate_pixelshuffle(prev_layer, upscale, num_out_ch, model_name)
    
    else:  # fragment shader
        # First conv + PReLU
        shader += generate_first_conv_frag(first_conv_w, first_conv_b, first_prelu, "conv0", num_feat, model_name, upscale)
        
        prev_layer = "conv0"
        conv_idx = 1
        
        # Middle convs
        for body_idx in range(2, last_conv_idx, 2):  # Skip by 2 (conv, prelu pairs)
            conv_key = f'body.{body_idx}.weight'
            prelu_key = f'body.{body_idx + 1}.weight'
            
            if conv_key not in params:
                break
            
            conv_w = params[conv_key].numpy()
            conv_b = params[f'body.{body_idx}.bias'].numpy()
            prelu = params[prelu_key].numpy() if prelu_key in params else np.ones(num_feat) * 0.25
            
            layer_name = f"conv{conv_idx}"
            shader += generate_mid_conv_frag(conv_w, conv_b, prelu, layer_name, prev_layer, num_feat, model_name, upscale)
            
            prev_layer = layer_name
            conv_idx += 1
        
        # Last conv (no activation)
        if upscale == 1:
            shader += generate_last_conv_x1_frag(last_conv_w, last_conv_b, f"conv{conv_idx}", prev_layer, num_feat, out_channels, model_name)
            prev_layer = f"conv{conv_idx}"
            shader += generate_output_x1_frag(prev_layer, num_out_ch, model_name)
        else:
            shader += generate_last_conv_frag(last_conv_w, last_conv_b, f"conv{conv_idx}", prev_layer, num_feat, out_channels, model_name)
            prev_layer = f"conv{conv_idx}"
            shader += generate_pixelshuffle_frag(prev_layer, upscale, num_out_ch, model_name)

    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(shader)
    
    print(f"\nShader saved to: {output_path}")
    print(f"Total conv layers: {conv_idx + 1}")

if __name__ == '__main__':
    import sys
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='Convert SRVGGNetCompact (.pth) to mpv GLSL shader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python ESRGAN_SRVGGNET_convert.py model.pth
  python ESRGAN_SRVGGNET_convert.py model.pth -o output.glsl
  python ESRGAN_SRVGGNET_convert.py model.pth -n "MyModel"
  python ESRGAN_SRVGGNET_convert.py model.pth -t fragment
'''
    )
    parser.add_argument('input', help='Input .pth model file')
    parser.add_argument('-o', '--output', help='Output .glsl file (default: same name as input)')
    parser.add_argument('-n', '--name', help='Model name for DESC (default: derived from filename)')
    parser.add_argument('-t', '--type', choices=['compute', 'fragment'], default='compute',
                        help='Shader type: compute (default) or fragment')
    
    args = parser.parse_args()
    
    model_path = args.input

    if args.output:
        output_path = args.output
    else:
        # Add suffix based on shader type
        base_name = os.path.splitext(model_path)[0]
        suffix = "_frag" if args.type == "fragment" else ""
        output_path = base_name + suffix + '.glsl'

    if args.name:
        model_name = args.name
    else:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    convert_model(model_path, output_path, model_name, args.type)
