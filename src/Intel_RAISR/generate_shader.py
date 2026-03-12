"""https://github.com/OpenVisualCloud/Video-Super-Resolution-Library
├─ Video-Super-Resolution-Library-main/
├─ intel_raisr_glsl/
   ├─ generate_shader.py
   ├─ parse_filterbin.py
   ├─ weights/
      ├─ intel_raisr_1_5x_denoise.py

Intel RAISR GLSL Shader Generator

The RAISR algorithm works in HR space:
1. First upscale LR to HR using cheap method (bilinear)
2. Compute gradients and hash in HR space
3. Apply filters in HR space based on pixelType

Usage:
    python generate_shader.py <weights_file> <output_shader>
"""

import struct
import sys
import os
import math
from pathlib import Path


LICENSE_HEADER = """

"""


def load_weights(weights_file):
    """Load weights from Python file."""
    with open(weights_file, 'r') as f:
        code = f.read()
    local_vars = {}
    exec(code, {}, local_vars)
    return local_vars


def generate_lut_texture(weights, float_format='rgba16f'):
    """Generate LUT texture data in hex format."""
    import numpy as np
    
    model_weights = weights['model_weights']
    quant_angle = weights['quant_angle']
    quant_strength = weights['quant_strength']
    quant_coherence = weights['quant_coherence']
    pixel_types = weights['pixel_types']
    patch_size = weights['patch_size']
    filter_size = patch_size * patch_size
    
    lut_width = (filter_size + 3) // 4
    num_hashkeys = quant_angle * quant_strength * quant_coherence
    lut_height = num_hashkeys * pixel_types
    
    lut_data = []
    for angle in range(quant_angle):
        for strength in range(quant_strength):
            for coherence in range(quant_coherence):
                for pixel_type in range(pixel_types):
                    filter_coeffs = model_weights[angle][strength][coherence][pixel_type]
                    padded = list(filter_coeffs)
                    while len(padded) % 4 != 0:
                        padded.append(0.0)
                    lut_data.extend(padded)
    
    # For mpv GLSL input, //!TEXTURE always expects float32 data (hex) even if the texture format is rgba16f.
    # The driver converts float32 -> float16 on upload.
    # So we always generate float32 data for hex string.
    
    data_array = np.array(lut_data, dtype=np.float32)
    
    if float_format == 'rgba16f':
        format_str = 'rgba16f'
    else:
        format_str = 'rgba32f'
    
    hex_data = data_array.tobytes().hex()
    
    # Format hex data into lines corresponding to texture height
    # Each row has 'lut_width' texels, each texel is 16 bytes (32 hex chars) because input is always float32
    chars_per_row = lut_width * 32  # 16 bytes/pixel * 2 chars/byte
        
    formatted_hex = ""
    for i in range(0, len(hex_data), chars_per_row):
        formatted_hex += hex_data[i:i+chars_per_row] + "\n"
        
    return formatted_hex.strip(), lut_width, lut_height, format_str


def generate_shader(weights, output_path, ratio=2.0):
    """Generate the complete GLSL shader - working in HR space."""
    
    quant_angle = weights['quant_angle']
    quant_strength = weights['quant_strength']
    quant_coherence = weights['quant_coherence']
    pixel_types = weights['pixel_types']
    patch_size = weights['patch_size']
    min_strength = weights['min_strength']
    min_coherence = weights['min_coherence']
    radius = weights['radius']
    filter_size = patch_size * patch_size
    gaussian = weights['gaussian']
    
    print("Generating LUT texture...")
    lut_hex, lut_width, lut_height, lut_format = generate_lut_texture(weights)
    
    # Flatten gaussian to 1D array string
    gaussian_flat = []
    for row in gaussian:
        gaussian_flat.extend(row)
    gaussian_str = ", ".join(f"{v:.10f}" for v in gaussian_flat)
    
    shader = LICENSE_HEADER
    
    # Pass 1: Cheap upscale (bilinear) - use mpv's built-in scaling
    # We just double the resolution first
    # Optimized Single-Pass RAISR
    # We hook the LR image directly and output a 2x HR image.
    # The "Cheap Upscale" (Bilinear) is handled by the GPU hardware sampler on-the-fly.
    # This saves memory bandwidth (no need to write/read intermediate 2x image).
    
    # Determine WHEN threshold based on ratio
    # User requested 1.0 for 1.5x to match "when upscaling" strictly
    if abs(ratio - 1.5) < 0.1:
        when_threshold = 1.0
    else:
        when_threshold = ratio * 0.6  # 1.2 for 2.0x
        
    # Prepare common header (defines, constants, functions) needed by both passes
    common_header = f"""#define PATCH_SIZE {patch_size}
#define RADIUS {radius}
#define LUT_WIDTH {lut_width}
#define LUT_HEIGHT {lut_height}
#define QUANT_ANGLE {quant_angle}
#define QUANT_STR {quant_strength}
#define QUANT_COH {quant_coherence}
#define PIXEL_TYPES {pixel_types}
#define RATIO {ratio}
#define PI 3.141592653589793
#define EPS 1.192092896e-7

const float gaussian[PATCH_SIZE * PATCH_SIZE] = float[]({gaussian_str});
"""

    # Check for 2-Pass Denoise Model
    if 'model_weights_2' in weights:
        print("Detected 2-Pass Denoise Model. Generating Pass 2...")
        
        # Generate LUT for Pass 2
        # Create a temporary weights dict for Pass 2 to reuse generate_lut_texture
        weights2 = weights.copy()
        weights2['model_weights'] = weights['model_weights_2']
        lut2_hex, lut2_width, lut2_height, lut2_fmt = generate_lut_texture(weights2)
        
        # Pass 2 needs its own header updates if dimensions differ, but usually they are same.
        # However, we should be safe and assume same constants unless LUT size changes.
        # We need to redefine definitions for Pass 2 if they differ (like LUT size).
        
        common_header_pass2 = f"""#define PATCH_SIZE {patch_size}
#define RADIUS {radius}
#define LUT_WIDTH {lut2_width}
#define LUT_HEIGHT {lut2_height}
#define QUANT_ANGLE {quant_angle}
#define QUANT_STR {quant_strength}
#define QUANT_COH {quant_coherence}
#define PIXEL_TYPES {pixel_types}
#define RATIO {ratio}
#define PI 3.141592653589793
#define EPS 1.192092896e-7

const float gaussian[PATCH_SIZE * PATCH_SIZE] = float[]({gaussian_str});
"""

        shader = LICENSE_HEADER
        
        # --- PASS 1: Upscale + Denoise ---
        shader += f"""//!HOOK LUMA
//!BIND HOOKED
//!BIND RAISR_LUT_1
//!SAVE RAISR_PRE
//!DESC Intel RAISR Pass 1
//!WIDTH {ratio} HOOKED.w *
//!HEIGHT {ratio} HOOKED.h *
//!WHEN OUTPUT.w HOOKED.w {when_threshold} * > OUTPUT.h HOOKED.h {when_threshold} * > *
//!COMPONENTS 1

{common_header}
float sampleHR(vec2 base, int di, int dj) {{
    vec2 offset = vec2(float(dj), float(di)) * (1.0/RATIO) * HOOKED_pt;
    return HOOKED_tex(base + offset).x;
}}

vec4 hook() {{

    vec2 pos = HOOKED_pos;
    ivec2 ipos = ivec2(HOOKED_pos * HOOKED_size * RATIO);

    // Compute gradient tensor (GTWG) in HR space
    // Standard RAISR uses an 11x11 patch from the upscaled image
    vec3 abd = vec3(0.0);

    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;

            // Calculate gradients using central differences on the virtual HR grid
            float left  = sampleHR(pos, di, dj - 1);
            float right = sampleHR(pos, di, dj + 1);
            float up    = sampleHR(pos, di - 1, dj);
            float down  = sampleHR(pos, di + 1, dj);

            float gx = (right - left) * 0.5;
            float gy = (down - up) * 0.5;
            float w = gaussian[i * PATCH_SIZE + j];
            abd += vec3(gx * gx, gx * gy, gy * gy) * w;
        }}
    }}

    // Eigenvalue decomposition
    float a = abd.x, b = abd.y, d = abd.z;
    float T = a + d;
    float D = a * d - b * b;
    float delta = sqrt(max(T * T / 4.0 - D, 0.0));
    float L1 = T / 2.0 + delta;
    float L2 = T / 2.0 - delta;
    float sqrtL1 = sqrt(L1);
    float sqrtL2 = sqrt(L2);

    // Angle quantization
    float theta = mix(mod(atan(L1 - a, b) + PI, PI), 0.0, abs(b) < EPS);
    float angle = floor(theta * float(QUANT_ANGLE) / PI);
    angle = clamp(angle, 0.0, float(QUANT_ANGLE - 1));

    // Strength quantization
    float lambda = sqrtL1;
    float strength = mix(mix(0.0, 1.0, lambda >= {min_strength[0]}), 2.0, lambda >= {min_strength[1]});

    // Coherence quantization
    float mu = mix((sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2), 0.0, sqrtL1 + sqrtL2 < EPS);
    float coherence = mix(mix(0.0, 1.0, mu >= {min_coherence[0]}), 2.0, mu >= {min_coherence[1]});

    // Compute hash key
    float hashKey = angle * float(QUANT_STR * QUANT_COH) + strength * float(QUANT_COH) + coherence;

    // Compute pixelType based on position in HR image
    int pixelType = 0;
    #if PIXEL_TYPES == 4
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2);
    #elif PIXEL_TYPES > 1
        // Fallback or future support for other patterns
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2); 
    #endif

    // LUT row index
    float lut_row = (hashKey * float(PIXEL_TYPES) + float(pixelType) + 0.5) / float(LUT_HEIGHT);

    // Apply filter
    float result = 0.0;
    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;
            float pixel = sampleHR(pos, di, dj);
            int coef_idx = i * PATCH_SIZE + j;
            int lut_x = coef_idx / 4;
            int lut_c = coef_idx % 4;
            float lut_u = (float(lut_x) + 0.5) / float(LUT_WIDTH);
            vec4 w = texture(RAISR_LUT_1, vec2(lut_u, lut_row));
            float coef;
            if (lut_c == 0) coef = w.x;
            else if (lut_c == 1) coef = w.y;
            else if (lut_c == 2) coef = w.z;
            else coef = w.w;
            result += pixel * coef;
        }}
    }}

    result = clamp(result, 0.0, 1.0);
    return vec4(result, 0.0, 0.0, 0.0);

}}

//!HOOK LUMA
//!BIND RAISR_LUT_2
//!BIND RAISR_PRE
//!DESC Intel RAISR Pass 2 (Refinement)
//!WIDTH {ratio} HOOKED.w *
//!HEIGHT {ratio} HOOKED.h *
//!COMPONENTS 1

{common_header_pass2}

float sampleRef(vec2 base, int di, int dj) {{
    vec2 offset = vec2(float(dj), float(di)) * RAISR_PRE_pt;
    return RAISR_PRE_tex(base + offset).x;
}}

vec4 hook() {{

    vec2 pos = RAISR_PRE_pos;
    ivec2 ipos = ivec2(RAISR_PRE_pos * RAISR_PRE_size);

    // Compute gradient tensor (GTWG) in HR space
    // Standard RAISR uses an 11x11 patch from the upscaled image
    vec3 abd = vec3(0.0);
    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;

            // Calculate gradients using central differences on the virtual HR grid
            float left  = sampleRef(pos, di, dj - 1);
            float right = sampleRef(pos, di, dj + 1);
            float up    = sampleRef(pos, di - 1, dj);
            float down  = sampleRef(pos, di + 1, dj);
            float gx = (right - left) * 0.5;
            float gy = (down - up) * 0.5;
            float w = gaussian[i * PATCH_SIZE + j];
            abd += vec3(gx * gx, gx * gy, gy * gy) * w;
        }}
    }}

    // Eigenvalue decomposition
    float a = abd.x, b = abd.y, d = abd.z;
    float T = a + d;
    float D = a * d - b * b;
    float delta = sqrt(max(T * T / 4.0 - D, 0.0));
    float L1 = T / 2.0 + delta;
    float L2 = T / 2.0 - delta;
    float sqrtL1 = sqrt(L1);
    float sqrtL2 = sqrt(L2);

    // Angle quantization
    float theta = mix(mod(atan(L1 - a, b) + PI, PI), 0.0, abs(b) < EPS);
    float angle = floor(theta * float(QUANT_ANGLE) / PI);
    angle = clamp(angle, 0.0, float(QUANT_ANGLE - 1));

    // Strength quantization
    float lambda = sqrtL1;
    float strength = mix(mix(0.0, 1.0, lambda >= {min_strength[0]}), 2.0, lambda >= {min_strength[1]});

    // Coherence quantization
    float mu = mix((sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2), 0.0, sqrtL1 + sqrtL2 < EPS);
    float coherence = mix(mix(0.0, 1.0, mu >= {min_coherence[0]}), 2.0, mu >= {min_coherence[1]});

    // Compute hash key
    float hashKey = angle * float(QUANT_STR * QUANT_COH) + strength * float(QUANT_COH) + coherence;

    // Compute pixelType based on position in HR image
    int pixelType = 0;
    #if PIXEL_TYPES == 4
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2);
    #elif PIXEL_TYPES > 1
        // Fallback or future support for other patterns
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2); 
    #endif

    // LUT row index
    float lut_row = (hashKey * float(PIXEL_TYPES) + float(pixelType) + 0.5) / float(LUT_HEIGHT);

    // Apply filter
    float result = 0.0;
    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;
            float pixel = sampleRef(pos, di, dj);
            int coef_idx = i * PATCH_SIZE + j;
            int lut_x = coef_idx / 4;
            int lut_c = coef_idx % 4;
            float lut_u = (float(lut_x) + 0.5) / float(LUT_WIDTH);
            vec4 w = texture(RAISR_LUT_2, vec2(lut_u, lut_row));
            float coef;
            if (lut_c == 0) coef = w.x;
            else if (lut_c == 1) coef = w.y;
            else if (lut_c == 2) coef = w.z;
            else coef = w.w;
            result += pixel * coef;
        }}
    }}

    result = clamp(result, 0.0, 1.0);
    return vec4(result, 0.0, 0.0, 0.0);

}}

//!TEXTURE RAISR_LUT_1
//!SIZE {lut_width} {lut_height}
//!FORMAT {lut_format}
//!FILTER NEAREST
{lut_hex}

//!TEXTURE RAISR_LUT_2
//!SIZE {lut2_width} {lut2_height}
//!FORMAT {lut2_fmt}
//!FILTER NEAREST
{lut2_hex}

"""
    else:
        # Standard 1-Pass Logic
        # Prepare common header for single pass
        common_header = f"""#define PATCH_SIZE {patch_size}
#define RADIUS {radius}
#define LUT_WIDTH {lut_width}
#define LUT_HEIGHT {lut_height}
#define QUANT_ANGLE {quant_angle}
#define QUANT_STR {quant_strength}
#define QUANT_COH {quant_coherence}
#define PIXEL_TYPES {pixel_types}
#define RATIO {ratio}
#define PI 3.141592653589793
#define EPS 1.192092896e-7

const float gaussian[PATCH_SIZE * PATCH_SIZE] = float[]({gaussian_str});
"""

        shader += f"""//!HOOK LUMA
//!BIND HOOKED
//!BIND RAISR_LUT
//!DESC Intel RAISR ({ratio}x, LUMA)
//!WIDTH {ratio} HOOKED.w *
//!HEIGHT {ratio} HOOKED.h *
//!WHEN OUTPUT.w HOOKED.w {when_threshold} * > OUTPUT.h HOOKED.h {when_threshold} * > *
//!COMPONENTS 1

{common_header}
float sampleHR(vec2 base, int di, int dj) {{
    vec2 offset = vec2(float(dj), float(di)) * (1.0/RATIO) * HOOKED_pt;
    return HOOKED_tex(base + offset).x;
}}

vec4 hook() {{

    vec2 pos = HOOKED_pos;
    ivec2 ipos = ivec2(HOOKED_pos * HOOKED_size * RATIO);

    // Compute gradient tensor (GTWG) in HR space
    // Standard RAISR uses an 11x11 patch from the upscaled image
    vec3 abd = vec3(0.0);

    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;

            // Calculate gradients using central differences on the virtual HR grid
            float left  = sampleHR(pos, di, dj - 1);
            float right = sampleHR(pos, di, dj + 1);
            float up    = sampleHR(pos, di - 1, dj);
            float down  = sampleHR(pos, di + 1, dj);

            float gx = (right - left) * 0.5;
            float gy = (down - up) * 0.5;
            float w = gaussian[i * PATCH_SIZE + j];
            abd += vec3(gx * gx, gx * gy, gy * gy) * w;
        }}
    }}

    // Eigenvalue decomposition
    float a = abd.x, b = abd.y, d = abd.z;
    float T = a + d;
    float D = a * d - b * b;
    float delta = sqrt(max(T * T / 4.0 - D, 0.0));
    float L1 = T / 2.0 + delta;
    float L2 = T / 2.0 - delta;
    float sqrtL1 = sqrt(L1);
    float sqrtL2 = sqrt(L2);

    // Angle quantization
    float theta = mix(mod(atan(L1 - a, b) + PI, PI), 0.0, abs(b) < EPS);
    float angle = floor(theta * float(QUANT_ANGLE) / PI);
    angle = clamp(angle, 0.0, float(QUANT_ANGLE - 1));

    // Strength quantization
    float lambda = sqrtL1;
    float strength = mix(mix(0.0, 1.0, lambda >= {min_strength[0]}), 2.0, lambda >= {min_strength[1]});

    // Coherence quantization
    float mu = mix((sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2), 0.0, sqrtL1 + sqrtL2 < EPS);
    float coherence = mix(mix(0.0, 1.0, mu >= {min_coherence[0]}), 2.0, mu >= {min_coherence[1]});

    // Compute hash key
    float hashKey = angle * float(QUANT_STR * QUANT_COH) + strength * float(QUANT_COH) + coherence;

    // Compute pixelType based on position in HR image
    int pixelType = 0;
    #if PIXEL_TYPES == 4
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2);
    #elif PIXEL_TYPES > 1
        // Fallback or future support for other patterns
        pixelType = (ipos.y % 2) * 2 + (ipos.x % 2); 
    #endif

    // LUT row index
    float lut_row = (hashKey * float(PIXEL_TYPES) + float(pixelType) + 0.5) / float(LUT_HEIGHT);

    // Apply filter
    float result = 0.0;
    for (int i = 0; i < PATCH_SIZE; i++) {{
        for (int j = 0; j < PATCH_SIZE; j++) {{
            int di = i - RADIUS;
            int dj = j - RADIUS;
            float pixel = sampleHR(pos, di, dj);
            int coef_idx = i * PATCH_SIZE + j;
            int lut_x = coef_idx / 4;
            int lut_c = coef_idx % 4;
            float lut_u = (float(lut_x) + 0.5) / float(LUT_WIDTH);
            vec4 w = texture(RAISR_LUT, vec2(lut_u, lut_row));
            float coef;
            if (lut_c == 0) coef = w.x;
            else if (lut_c == 1) coef = w.y;
            else if (lut_c == 2) coef = w.z;
            else coef = w.w;
            result += pixel * coef;
        }}
    }}

    result = clamp(result, 0.0, 1.0);
    return vec4(result, 0.0, 0.0, 0.0);

}}

//!TEXTURE RAISR_LUT
//!SIZE {lut_width} {lut_height}
//!FORMAT {lut_format}
//!FILTER NEAREST
{lut_hex}

"""

    with open(output_path, 'w', newline='\n') as f:
        f.write(shader)
    
    print(f"\nShader written to: {output_path}")
    print(f"Shader size: {os.path.getsize(output_path) / 1024:.2f} KB")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Intel RAISR GLSL Shader')
    parser.add_argument('weights_file', help='Input Python weights file')
    parser.add_argument('output_file', help='Output GLSL shader file')
    parser.add_argument('--ratio', type=float, default=2.0, help='Upscaling ratio (default: 2.0)')

    args = parser.parse_args()

    if not os.path.exists(args.weights_file):
        print(f"Error: Weights file not found: {args.weights_file}")
        sys.exit(1)

    print(f"Loading weights from: {args.weights_file}")
    print(f"Target ratio: {args.ratio}")
    weights = load_weights(args.weights_file)

    print("\nWeights loaded:")
    print(f"  Radius: {weights['radius']}")
    print(f"  Patch size: {weights['patch_size']}")
    print(f"  Angle bins: {weights['quant_angle']}")
    print(f"  Strength bins: {weights['quant_strength']}")
    print(f"  Coherence bins: {weights['quant_coherence']}")
    print(f"  Pixel types: {weights['pixel_types']}")

    generate_shader(weights, args.output_file, args.ratio)

if __name__ == "__main__":
    main()

