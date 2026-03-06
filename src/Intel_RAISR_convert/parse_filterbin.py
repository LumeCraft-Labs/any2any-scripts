"""
Intel RAISR filterbin Parser

Parses Intel's pre-trained RAISR filter files and extracts:
- Filter coefficients (216 hash keys × 4 pixel types × 121 coefficients)
- Quantization parameters (Qstr, Qcoh)
- Config parameters (angle, strength, coherence bins, patch size)

Usage:
    python parse_filterbin.py <filter_folder> <output_file>
    
Example:
    python parse_filterbin.py ../intel-vsr/filters_2x/filters_lowres intel_raisr_weights.py
"""

import struct
import numpy as np
import os
import sys
from pathlib import Path


def parse_config(config_path):
    """Parse the config file containing algorithm parameters."""
    with open(config_path, 'r') as f:
        values = f.read().strip().split()
    
    return {
        'quant_angle': int(values[0]),      # 24
        'quant_strength': int(values[1]),   # 3
        'quant_coherence': int(values[2]),  # 3
        'patch_size': int(values[3])        # 11
    }


def parse_qfactor(qfactor_path):
    """Parse quantization factor files (Qstr or Qcoh)."""
    with open(qfactor_path, 'r') as f:
        values = [float(line.strip()) for line in f.readlines() if line.strip()]
    return values


def parse_filterbin(filterbin_path):
    """
    Parse Intel's binary filter file.
    
    Format:
    - 4 bytes: type string ('fp32' or 'fp16')
    - 4 bytes: hashkeySize (uint32) - typically 216
    - 4 bytes: pixelTypes (uint32) - typically 4
    - 4 bytes: rows (uint32) - typically 121 (11x11)
    - Rest: filter data as float32 or float16
    
    Returns:
        filters: numpy array of shape (hashkeys, pixel_types, rows)
        metadata: dict with type, hashkeys, pixel_types, rows
    """
    with open(filterbin_path, 'rb') as f:
        # Read header
        type_str = f.read(4).decode('utf-8').strip('\x00')
        header = struct.unpack('III', f.read(12))
        hashkeys, pixel_types, rows = header
        
        # Determine data type
        if type_str == 'fp32':
            dtype = np.float32
        elif type_str == 'fp16':
            dtype = np.float16
        else:
            raise ValueError(f"Unknown filter type: {type_str}")
        
        # Read filter data
        data = np.frombuffer(f.read(), dtype=dtype).copy()
        
        # Reshape: the data is stored as [hashkey][pixelType][row]
        expected_size = hashkeys * pixel_types * rows
        if len(data) != expected_size:
            raise ValueError(f"Data size mismatch: expected {expected_size}, got {len(data)}")
        
        filters = data.reshape(hashkeys, pixel_types, rows)
        
        # Normalize filters to ensure energy conservation (fix brightness shifts)
        print("Normalizing filters...")
        for h in range(hashkeys):
            for p in range(pixel_types):
                kernel = filters[h, p, :]
                ksum = kernel.sum()
                if abs(ksum) > 1e-6:
                    filters[h, p, :] /= ksum
        
        metadata = {
            'type': type_str,
            'hashkeys': hashkeys,
            'pixel_types': pixel_types,
            'rows': rows,
            'patch_size': int(np.sqrt(rows))
        }
        
        return filters, metadata


def analyze_filters(filters, metadata):
    """Analyze filter properties for debugging and optimization."""
    print(f"\n=== Filter Analysis ===")
    print(f"Type: {metadata['type']}")
    print(f"Hash keys: {metadata['hashkeys']}")
    print(f"Pixel types: {metadata['pixel_types']}")
    print(f"Filter size: {metadata['patch_size']}x{metadata['patch_size']} = {metadata['rows']}")
    print(f"Total filters: {metadata['hashkeys'] * metadata['pixel_types']}")
    print(f"Total coefficients: {filters.size}")
    print(f"Data size: {filters.nbytes / 1024:.2f} KB")
    
    # Check for symmetry
    print(f"\n--- Symmetry Analysis ---")
    symmetric_count = 0
    total_filters = metadata['hashkeys'] * metadata['pixel_types']
    
    for h in range(metadata['hashkeys']):
        for p in range(metadata['pixel_types']):
            kernel = filters[h, p, :].reshape(metadata['patch_size'], metadata['patch_size'])
            # Check point symmetry (180 degree rotation)
            rotated = np.rot90(kernel, 2)
            if np.allclose(kernel, rotated, rtol=1e-5):
                symmetric_count += 1
    
    print(f"Point-symmetric filters: {symmetric_count}/{total_filters} ({100*symmetric_count/total_filters:.1f}%)")
    
    # Filter statistics
    print(f"\n--- Value Statistics ---")
    print(f"Min: {filters.min():.6f}")
    print(f"Max: {filters.max():.6f}")
    print(f"Mean: {filters.mean():.6f}")
    print(f"Std: {filters.std():.6f}")
    
    # Check normalization (filter sums)
    sums = []
    for h in range(metadata['hashkeys']):
        for p in range(metadata['pixel_types']):
            sums.append(filters[h, p, :].sum())
    sums = np.array(sums)
    print(f"\n--- Filter Sum Statistics ---")
    print(f"Min sum: {sums.min():.6f}")
    print(f"Max sum: {sums.max():.6f}")
    print(f"Mean sum: {sums.mean():.6f}")


def export_to_python(filters, config, qstr, qcoh, metadata, output_path, filters2=None, metadata2=None):
    """Export filters and parameters to Python format (like bjin-ravu)."""
    
    patch_size = metadata['patch_size']
    radius = patch_size // 2  # 5 for 11x11
    
    with open(output_path, 'w') as f:
        f.write("# Intel RAISR filter weights\n")
        f.write("# Auto-generated from Intel filterbin\n\n")
        
        # Parameters
        f.write(f"radius = {radius}\n")
        f.write(f"gradient_radius = {radius}\n")  # Same as radius for Intel RAISR
        f.write(f"quant_angle = {config['quant_angle']}\n")
        f.write(f"quant_strength = {config['quant_strength']}\n")
        f.write(f"quant_coherence = {config['quant_coherence']}\n")
        f.write(f"pixel_types = {metadata['pixel_types']}\n")
        f.write(f"patch_size = {patch_size}\n")
        
        # Quantization thresholds
        f.write(f"min_strength = {qstr}\n")
        f.write(f"min_coherence = {qcoh}\n")
        
        # Gaussian weights for gradient computation (11x11 gradient window)
        # Using simple Gaussian with sigma = patch_size/6
        sigma = patch_size / 6.0
        gaussian = np.zeros((patch_size, patch_size))
        center = patch_size // 2
        for i in range(patch_size):
            for j in range(patch_size):
                x, y = i - center, j - center
                gaussian[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        gaussian = gaussian / gaussian.sum()
        
        f.write(f"\n# Gaussian weights for gradient computation\n")
        f.write(f"gaussian = [\n")
        for row in gaussian:
            f.write(f"    [{', '.join(f'{v:.16f}' for v in row)}],\n")
        f.write("]\n")
        
        # Helper to write weights
        def write_weights(f_obj, w_array, var_name, md):
            f_obj.write(f"\n# {var_name}: [{config['quant_angle']}][{config['quant_strength']}][{config['quant_coherence']}][{md['pixel_types']}][{md['rows']}]\n")
            f_obj.write(f"{var_name} = ")
            
            reshaped = []
            for angle in range(config['quant_angle']):
                angle_weights = []
                for strength in range(config['quant_strength']):
                    strength_weights = []
                    for coherence in range(config['quant_coherence']):
                        hashkey = angle * (config['quant_strength'] * config['quant_coherence']) + \
                                  strength * config['quant_coherence'] + coherence
                        coherence_weights = []
                        for pixel_type in range(md['pixel_types']):
                            coherence_weights.append(w_array[hashkey, pixel_type, :].tolist())
                        strength_weights.append(coherence_weights)
                    angle_weights.append(strength_weights)
                reshaped.append(angle_weights)
            
            f_obj.write(repr(reshaped))
            f_obj.write("\n")

        # Write primary weights
        write_weights(f, filters, "model_weights", metadata)
        
        # Write secondary weights if present (Pass 2)
        if filters2 is not None:
            write_weights(f, filters2, "model_weights_2", metadata2)
    
    print(f"\nExported to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_filterbin.py <filter_folder> <output_file>")
        print("Example: python parse_filterbin.py ../intel-vsr/filters_2x/filters_lowres weights.py")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_file = sys.argv[2]

    if input_path.is_file():
        filter_folder = input_path.parent
        filterbin_files = [input_path]
    else:
        filter_folder = input_path
        filterbin_files = list(filter_folder.glob("filterbin_*"))

    config_path = filter_folder / "config"
    qstr_files = list(filter_folder.glob("Qfactor_strbin_*"))
    qcoh_files = list(filter_folder.glob("Qfactor_cohbin_*"))
    
    if not config_path.exists():
        print(f"Error: config file not found in {filter_folder}")
        sys.exit(1)
    
    if not filterbin_files:
        print(f"Error: no filterbin files found in {filter_folder}")
        sys.exit(1)

    print(f"Parsing config: {config_path}")
    config = parse_config(config_path)
    print(f"  Angle bins: {config['quant_angle']}")
    print(f"  Strength bins: {config['quant_strength']}")
    print(f"  Coherence bins: {config['quant_coherence']}")
    print(f"  Patch size: {config['patch_size']}")
    
    # Parse quantization factors
    qstr = []
    qcoh = []
    
    if qstr_files:
        print(f"\nParsing Qstr: {qstr_files[0]}")
        qstr = parse_qfactor(qstr_files[0])
        print(f"  Strength thresholds: {qstr}")
    
    if qcoh_files:
        print(f"\nParsing Qcoh: {qcoh_files[0]}")
        qcoh = parse_qfactor(qcoh_files[0])
        print(f"  Coherence thresholds: {qcoh}")
    
    # Parse filterbin
    print(f"\nParsing filterbin: {filterbin_files[0]}")
    filters, metadata = parse_filterbin(filterbin_files[0])
    
    # Check for Pass 2 filterbin (assumes naming convention like filterbin_2_8 and filterbin_2_8_2)
    filters2 = None
    metadata2 = None
    
    # Try multiple naming conventions for pass 2
    # Convention 1: append _2 (e.g., filterbin_2_8 -> filterbin_2_8_2)
    # Convention 2: replace end (e.g., filterbin_2_8 -> filterbin_2_8_2)
    base_name = filterbin_files[0].stem
    
    # Common convention seems to be appending _2
    pass2_candidates = [
        filter_folder / f"{base_name}_2",
        filter_folder / f"{base_name}_2.bin" # binary extension?
    ]
    
    pass2_file = None
    for cand in pass2_candidates:
        if cand.exists():
            pass2_file = cand
            break
            
    if pass2_file:
        print(f"\nFound Pass 2 filterbin: {pass2_file}")
        filters2, metadata2 = parse_filterbin(pass2_file)
        print("Analyzing Pass 2 filters...")
        analyze_filters(filters2, metadata2)

    analyze_filters(filters, metadata)

    export_to_python(filters, config, qstr, qcoh, metadata, output_file, filters2, metadata2)


if __name__ == "__main__":
    main()
