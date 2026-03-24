"""
Gemini Watermark Remover

src: https://github.com/GargantuaX/gemini-watermark-remover

用法:
  python toss_banana_peel.py input.png -o output.png
  python toss_banana_peel.py input_dir/ -o output_dir/
"""

from __future__ import annotations

import argparse
import base64
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1. Alpha Map (Base64 → Float32Array)
# ═══════════════════════════════════════════════════════════════════════════

_ALPHA_MAP_BASE64 = {
	48: None,
	96: None,
}

def _get_embedded_alpha_map(size: int) -> Optional[np.ndarray]:
	b64 = _ALPHA_MAP_BASE64.get(size)
	if b64 is None:
		return None
	raw = base64.b64decode(b64)
	expected = size * size
	arr = np.frombuffer(raw, dtype=np.float32)
	if arr.shape[0] != expected:
		return None
	return arr.copy()


_alpha_cache: Dict[int, np.ndarray] = {}


def get_alpha_map(size: int) -> np.ndarray:
	if size in _alpha_cache:
		return _alpha_cache[size].copy()

	embedded = _get_embedded_alpha_map(size)
	if embedded is not None:
		_alpha_cache[size] = embedded
		return embedded.copy()

	# 非标准尺寸：从 96px 插值
	alpha96 = get_alpha_map(96)
	interpolated = interpolate_alpha_map(alpha96, 96, size)
	_alpha_cache[size] = interpolated
	return interpolated.copy()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Alpha Map 插值
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_alpha_map(source: np.ndarray, source_size: int, target_size: int) -> np.ndarray:
	"""双线性插值缩放 alpha map。"""
	if target_size <= 0:
		return np.zeros(0, dtype=np.float32)
	if source_size == target_size:
		return source.copy()

	src = source.reshape(source_size, source_size)
	scale = (source_size - 1) / max(1, target_size - 1)
	out = np.zeros((target_size, target_size), dtype=np.float32)

	for y in range(target_size):
		sy = y * scale
		y0 = int(sy)
		y1 = min(source_size - 1, y0 + 1)
		fy = sy - y0
		for x in range(target_size):
			sx = x * scale
			x0 = int(sx)
			x1 = min(source_size - 1, x0 + 1)
			fx = sx - x0
			top = src[y0, x0] + (src[y0, x1] - src[y0, x0]) * fx
			bot = src[y1, x0] + (src[y1, x1] - src[y1, x0]) * fx
			out[y, x] = top + (bot - top) * fy

	return out.flatten()


def warp_alpha_map(alpha_map: np.ndarray, size: int,
				   dx: float = 0, dy: float = 0, scale: float = 1) -> np.ndarray:
	"""对 alpha map 做平移 + 缩放变换"""
	if size <= 0 or scale <= 0:
		return np.zeros(0, dtype=np.float32)
	if dx == 0 and dy == 0 and scale == 1:
		return alpha_map.copy()

	src = alpha_map.reshape(size, size)
	out = np.zeros((size, size), dtype=np.float32)
	c = (size - 1) / 2.0

	for y in range(size):
		for x in range(size):
			sx = (x - c) / scale + c + dx
			sy = (y - c) / scale + c + dy
			x0 = int(math.floor(sx))
			y0 = int(math.floor(sy))
			fx = sx - x0
			fy = sy - y0
			ix0 = max(0, min(size - 1, x0))
			iy0 = max(0, min(size - 1, y0))
			ix1 = max(0, min(size - 1, x0 + 1))
			iy1 = max(0, min(size - 1, y0 + 1))
			top = src[iy0, ix0] + (src[iy0, ix1] - src[iy0, ix0]) * fx
			bot = src[iy1, ix0] + (src[iy1, ix1] - src[iy1, ix0]) * fx
			out[y, x] = top + (bot - top) * fy

	return out.flatten()


# ═══════════════════════════════════════════════════════════════════════════
# 3. 核心 — 逆向 Alpha 混合
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_NOISE_FLOOR = 3.0 / 255.0   # 量化噪声阈值
ALPHA_THRESHOLD = 0.002           # 忽略极小 alpha
MAX_ALPHA = 0.99                  # 防止除零
LOGO_VALUE = 255.0                # 白色水印


def remove_watermark(image_data: np.ndarray, width: int,
					 alpha_map: np.ndarray, position: dict,
					 alpha_gain: float = 1.0) -> None:
	"""
	原地对图像数据执行逆向 alpha 混合去除水印。

	image_data: shape (H*W*4,) 的 uint8 数组 (RGBA)
	alpha_map: shape (size*size,) 的 float32 数组
	position: {"x": int, "y": int, "width": int, "height": int}
	"""
	px, py, pw, ph = position["x"], position["y"], position["width"], position["height"]

	for row in range(ph):
		for col in range(pw):
			img_idx = ((py + row) * width + (px + col)) * 4
			alpha_idx = row * pw + col

			raw_alpha = alpha_map[alpha_idx]
			signal_alpha = max(0.0, raw_alpha - ALPHA_NOISE_FLOOR) * alpha_gain

			if signal_alpha < ALPHA_THRESHOLD:
				continue

			alpha = min(raw_alpha * alpha_gain, MAX_ALPHA)
			one_minus_alpha = 1.0 - alpha

			for c in range(3):
				watermarked = float(image_data[img_idx + c])
				original = (watermarked - alpha * LOGO_VALUE) / one_minus_alpha
				image_data[img_idx + c] = max(0, min(255, round(original)))


# ═══════════════════════════════════════════════════════════════════════════
# 4. 检测 — 归一化互相关 + Sobel
# ═══════════════════════════════════════════════════════════════════════════

EPSILON = 1e-8


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
	"""归一化互相关 (Normalized Cross-Correlation)"""
	if a.size != b.size or a.size == 0:
		return 0.0
	a_mean = a.mean()
	b_mean = b.mean()
	a_var = np.mean((a - a_mean) ** 2)
	b_var = np.mean((b - b_mean) ** 2)
	den = math.sqrt(a_var * b_var) * a.size
	if den < EPSILON:
		return 0.0
	num = np.sum((a - a_mean) * (b - b_mean))
	return float(num / den)


def _to_grayscale(data: np.ndarray, width: int, height: int) -> np.ndarray:
	"""RGBA 数据转灰度 (BT.709)，返回 (H, W) float32。"""
	rgba = data.reshape(height, width, 4).astype(np.float32)
	gray = (0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]) / 255.0
	return gray


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
	"""Sobel 梯度，输入 (H, W)，输出同尺寸。"""
	h, w = gray.shape
	grad = np.zeros_like(gray)
	for y in range(1, h - 1):
		for x in range(1, w - 1):
			gx = (-gray[y-1, x-1] - 2*gray[y, x-1] - gray[y+1, x-1]
				  + gray[y-1, x+1] + 2*gray[y, x+1] + gray[y+1, x+1])
			gy = (-gray[y-1, x-1] - 2*gray[y-1, x] - gray[y-1, x+1]
				  + gray[y+1, x-1] + 2*gray[y+1, x] + gray[y+1, x+1])
			grad[y, x] = math.sqrt(gx * gx + gy * gy)
	return grad


def _region_grayscale(data: np.ndarray, width: int, x: int, y: int, size: int) -> np.ndarray:
	"""提取区域灰度 patch，返回 (size*size,) float32。"""
	height = len(data) // (width * 4)
	if x < 0 or y < 0 or x + size > width or y + size > height:
		return np.zeros(0, dtype=np.float32)
	rgba = data.reshape(height, width, 4).astype(np.float32)
	patch = rgba[y:y+size, x:x+size, :]
	gray = (0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]) / 255.0
	return gray.flatten()


def _get_region(data_2d: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
	"""从 2D 数组中提取 size×size 区域并展平。"""
	return data_2d[y:y+size, x:x+size].flatten().copy()


def compute_region_spatial_correlation(data: np.ndarray, width: int,
									   alpha_map: np.ndarray,
									   x: int, y: int, size: int) -> float:
	"""计算区域空间相关性。"""
	patch = _region_grayscale(data, width, x, y, size)
	if patch.size == 0 or patch.size != alpha_map.size:
		return 0.0
	return _ncc(patch, alpha_map)


def compute_region_gradient_correlation(data: np.ndarray, width: int,
										alpha_map: np.ndarray,
										x: int, y: int, size: int) -> float:
	"""计算区域梯度相关性。"""
	patch = _region_grayscale(data, width, x, y, size)
	if patch.size == 0 or patch.size != alpha_map.size:
		return 0.0
	patch_grad = _sobel_magnitude(patch.reshape(size, size)).flatten()
	alpha_grad = _sobel_magnitude(alpha_map.reshape(size, size)).flatten()
	return _ncc(patch_grad, alpha_grad)


def score_region(data: np.ndarray, width: int, alpha_map: np.ndarray, position: dict) -> dict:
	"""计算空间 + 梯度分数。"""
	x, y, sz = position["x"], position["y"], position["width"]
	return {
		"spatialScore": compute_region_spatial_correlation(data, width, alpha_map, x, y, sz),
		"gradientScore": compute_region_gradient_correlation(data, width, alpha_map, x, y, sz),
	}


# ═══════════════════════════════════════════════════════════════════════════
# 5. 质量指标
# ═══════════════════════════════════════════════════════════════════════════

NEAR_BLACK_THRESHOLD = 5


def clone_image_data(data: np.ndarray) -> np.ndarray:
	return data.copy()


def calculate_near_black_ratio(data: np.ndarray, width: int, position: dict) -> float:
	"""计算水印区域近黑像素比率。"""
	px, py, pw, ph = position["x"], position["y"], position["width"], position["height"]
	height = len(data) // (width * 4)
	rgba = data.reshape(height, width, 4)
	region = rgba[py:py+ph, px:px+pw, :3]
	near_black = np.all(region <= NEAR_BLACK_THRESHOLD, axis=2)
	total = pw * ph
	return float(np.sum(near_black) / total) if total > 0 else 0.0


def _region_texture_stats(data: np.ndarray, width: int, region: dict) -> dict:
	"""计算区域纹理统计（均值亮度 + 标准差）。"""
	rx, ry, rw, rh = region["x"], region["y"], region["width"], region["height"]
	height = len(data) // (width * 4)
	rgba = data.reshape(height, width, 4).astype(np.float32)
	patch = rgba[ry:ry+rh, rx:rx+rw, :]
	lum = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
	mean_lum = float(lum.mean())
	std_lum = float(lum.std())
	return {"meanLum": mean_lum, "stdLum": std_lum}


def assess_reference_texture(original_data: np.ndarray, candidate_data: np.ndarray,
							 width: int, position: dict) -> dict:
	"""评估修复质量 vs 参考区域（水印上方）。"""
	ref_y = position["y"] - position["height"]
	if ref_y < 0:
		return {"texturePenalty": 0, "tooDark": False, "tooFlat": False, "hardReject": False}

	ref_region = {"x": position["x"], "y": ref_y,
				  "width": position["width"], "height": position["height"]}
	ref_stats = _region_texture_stats(original_data, width, ref_region)
	cand_stats = _region_texture_stats(candidate_data, width, position)

	darkness_penalty = max(0, ref_stats["meanLum"] - cand_stats["meanLum"] - 1) / max(1, ref_stats["meanLum"])
	flatness_penalty = max(0, ref_stats["stdLum"] * 0.8 - cand_stats["stdLum"]) / max(1, ref_stats["stdLum"])
	too_dark = darkness_penalty > 0
	too_flat = flatness_penalty > 0

	return {
		"texturePenalty": darkness_penalty * 2 + flatness_penalty * 2,
		"tooDark": too_dark,
		"tooFlat": too_flat,
		"hardReject": too_dark and too_flat,
	}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Gemini 尺寸
# ═══════════════════════════════════════════════════════════════════════════

_WATERMARK_CONFIG_BY_TIER = {
	"0.5k": {"logoSize": 48, "marginRight": 32, "marginBottom": 32},
	"1k":   {"logoSize": 96, "marginRight": 64, "marginBottom": 64},
	"2k":   {"logoSize": 96, "marginRight": 64, "marginBottom": 64},
	"4k":   {"logoSize": 96, "marginRight": 64, "marginBottom": 64},
}

_OFFICIAL_GEMINI_SIZES: List[dict] = []

def _add_entries(model: str, tier: str, rows: list):
	for ratio, w, h in rows:
		_OFFICIAL_GEMINI_SIZES.append({
			"modelFamily": model, "resolutionTier": tier,
			"aspectRatio": ratio, "width": w, "height": h
		})

_add_entries("gemini-3.x-image", "0.5k", [
	("1:1",512,512),("1:4",256,1024),("1:8",192,1536),("2:3",424,632),
	("3:2",632,424),("3:4",448,600),("4:1",1024,256),("4:3",600,448),
	("4:5",464,576),("5:4",576,464),("8:1",1536,192),("9:16",384,688),
	("16:9",688,384),("21:9",792,168),
])
_add_entries("gemini-3.x-image", "1k", [
	("1:1",1024,1024),("2:3",848,1264),("3:2",1264,848),("3:4",896,1200),
	("4:3",1200,896),("4:5",928,1152),("5:4",1152,928),("9:16",768,1376),
	("16:9",1376,768),("21:9",1584,672),
])
_add_entries("gemini-3.x-image", "2k", [
	("1:1",2048,2048),("1:4",512,2048),("1:8",384,3072),("2:3",1696,2528),
	("3:2",2528,1696),("3:4",1792,2400),("4:1",2048,512),("4:3",2400,1792),
	("4:5",1856,2304),("5:4",2304,1856),("8:1",3072,384),("9:16",1536,2752),
	("16:9",2752,1536),("21:9",3168,1344),
])
_add_entries("gemini-3.x-image", "4k", [
	("1:1",4096,4096),("1:4",2048,8192),("1:8",1536,12288),("2:3",3392,5056),
	("3:2",5056,3392),("3:4",3584,4800),("4:1",8192,2048),("4:3",4800,3584),
	("4:5",3712,4608),("5:4",4608,3712),("8:1",12288,1536),("9:16",3072,5504),
	("16:9",5504,3072),("21:9",6336,2688),
])
_add_entries("gemini-2.5-flash-image", "1k", [
	("1:1",1024,1024),("2:3",832,1248),("3:2",1248,832),("3:4",864,1184),
	("4:3",1184,864),("4:5",896,1152),("5:4",1152,896),("9:16",768,1344),
	("16:9",1344,768),("21:9",1536,672),
])

_SIZE_INDEX = {f"{e['width']}x{e['height']}": e for e in _OFFICIAL_GEMINI_SIZES}


def _clamp(v, lo, hi):
	return max(lo, min(hi, v))


def match_official_size(w: int, h: int) -> Optional[dict]:
	return _SIZE_INDEX.get(f"{w}x{h}")


def resolve_official_watermark_config(w: int, h: int) -> Optional[dict]:
	entry = match_official_size(w, h)
	if not entry:
		return None
	return _WATERMARK_CONFIG_BY_TIER.get(entry["resolutionTier"])


def detect_watermark_config(w: int, h: int) -> dict:
	cfg = resolve_official_watermark_config(w, h)
	if cfg:
		return dict(cfg)
	if w > 1024 and h > 1024:
		return {"logoSize": 96, "marginRight": 64, "marginBottom": 64}
	return {"logoSize": 48, "marginRight": 32, "marginBottom": 32}


def calculate_watermark_position(w: int, h: int, config: dict) -> dict:
	ls = config["logoSize"]
	return {
		"x": w - config["marginRight"] - ls,
		"y": h - config["marginBottom"] - ls,
		"width": ls, "height": ls,
	}


def resolve_search_configs(w: int, h: int, default_config: Optional[dict] = None) -> List[dict]:
	"""解析 Gemini 尺寸，返回候选配置列表。"""
	target_ar = w / h if h > 0 else 1
	candidates = []
	for entry in _OFFICIAL_GEMINI_SIZES:
		base = _WATERMARK_CONFIG_BY_TIER.get(entry["resolutionTier"])
		if not base:
			continue
		sx = w / entry["width"]
		sy = h / entry["height"]
		scale = (sx + sy) / 2
		ear = entry["width"] / entry["height"]
		ar_delta = abs(target_ar - ear) / ear
		scale_mismatch = abs(sx - sy) / max(sx, sy)
		if ar_delta > 0.02 or scale_mismatch > 0.12:
			continue
		logo = _clamp(round(base["logoSize"] * scale), 24, 192)
		mr = max(8, round(base["marginRight"] * sx))
		mb = max(8, round(base["marginBottom"] * sy))
		if w - mr - logo < 0 or h - mb - logo < 0:
			continue
		score = ar_delta * 100 + scale_mismatch * 20 + abs(math.log2(max(scale, 1e-6)))
		candidates.append((score, logo, mr, mb))

	candidates.sort()
	seen = set()
	result = []
	if default_config:
		key = (default_config["logoSize"], default_config["marginRight"], default_config["marginBottom"])
		seen.add(key)
		result.append(dict(default_config))
	for _, logo, mr, mb in candidates:
		key = (logo, mr, mb)
		if key in seen:
			continue
		seen.add(key)
		result.append({"logoSize": logo, "marginRight": mr, "marginBottom": mb})
		if len(result) >= 4:
			break
	return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. 决策
# ═══════════════════════════════════════════════════════════════════════════

def _has_reliable_standard_signal(spatial: float, gradient: float) -> bool:
	return spatial is not None and gradient is not None and spatial >= 0.3 and gradient >= 0.12


def _has_reliable_adaptive_signal(result: Optional[dict]) -> bool:
	if not result or not result.get("found"):
		return False
	c = result.get("confidence", 0)
	s = result.get("spatialScore", 0)
	g = result.get("gradientScore", 0)
	size = result.get("region", {}).get("size", 0)
	return c >= 0.5 and s >= 0.45 and g >= 0.12 and 40 <= size <= 192


# ═══════════════════════════════════════════════════════════════════════════
# 8. 自适应检测器
# ═══════════════════════════════════════════════════════════════════════════

def _score_candidate(gray: np.ndarray, grad: np.ndarray, alpha: np.ndarray,
					 template_grad: np.ndarray, x: int, y: int, size: int) -> Optional[dict]:
	h, w = gray.shape
	if x < 0 or y < 0 or x + size > w or y + size > h:
		return None
	gray_r = gray[y:y+size, x:x+size].flatten()
	grad_r = grad[y:y+size, x:x+size].flatten()
	spatial = _ncc(gray_r, alpha)
	gradient = _ncc(grad_r, template_grad)

	variance_score = 0.0
	if y > 8:
		ref_y = max(0, y - size)
		ref_h = min(size, y - ref_y)
		if ref_h > 8:
			wm_std = float(gray[y:y+size, x:x+size].std())
			ref_std = float(gray[ref_y:ref_y+ref_h, x:x+size].std())
			if ref_std > EPSILON:
				variance_score = _clamp(1 - wm_std / ref_std, 0, 1)

	confidence = max(0, spatial) * 0.5 + max(0, gradient) * 0.3 + variance_score * 0.2
	return {
		"confidence": _clamp(confidence, 0, 1),
		"spatialScore": spatial,
		"gradientScore": gradient,
		"varianceScore": variance_score,
	}


def detect_adaptive_watermark(data: np.ndarray, width: int, height: int,
							  alpha96: np.ndarray, default_config: dict,
							  threshold: float = 0.35) -> dict:
	"""自适应水印检测：粗搜索 → 精搜索。"""
	gray = _to_grayscale(data, width, height)
	grad = _sobel_magnitude(gray)

	template_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

	def get_template(size: int):
		if size in template_cache:
			return template_cache[size]
		alpha = alpha96 if size == 96 else interpolate_alpha_map(alpha96, 96, size)
		tgrad = _sobel_magnitude(alpha.reshape(size, size)).flatten()
		template_cache[size] = (alpha, tgrad)
		return alpha, tgrad

	# 种子候选
	seed_configs = resolve_search_configs(width, height, default_config)
	seed_candidates = []
	for cfg in seed_configs:
		sz = cfg["logoSize"]
		cx = width - cfg["marginRight"] - sz
		cy = height - cfg["marginBottom"] - sz
		if cx < 0 or cy < 0 or cx + sz > width or cy + sz > height:
			continue
		alpha, tgrad = get_template(sz)
		sc = _score_candidate(gray, grad, alpha, tgrad, cx, cy, sz)
		if sc:
			seed_candidates.append({"x": cx, "y": cy, "size": sz, **sc})

	best_seed = max(seed_candidates, key=lambda c: c["confidence"]) if seed_candidates else None
	if best_seed and best_seed["confidence"] >= threshold + 0.08:
		return {
			"found": True, **{k: best_seed[k] for k in ("confidence", "spatialScore", "gradientScore", "varianceScore")},
			"region": {"x": best_seed["x"], "y": best_seed["y"], "size": best_seed["size"]},
		}

	# 粗搜索
	base_size = default_config["logoSize"]
	min_size = _clamp(round(base_size * 0.65), 24, 144)
	max_size = _clamp(min(round(base_size * 2.8), int(min(width, height) * 0.4)), min_size, 192)
	scale_list = sorted(set(range(min_size, max_size + 1, 8)) | ({48, 96} & set(range(min_size, max_size + 1))))

	mr_range = max(32, round(base_size * 0.75))
	min_mr = _clamp(default_config["marginRight"] - mr_range, 8, width - min_size - 1)
	max_mr = _clamp(default_config["marginRight"] + mr_range, min_mr, width - min_size - 1)
	min_mb = _clamp(default_config["marginBottom"] - mr_range, 8, height - min_size - 1)
	max_mb = _clamp(default_config["marginBottom"] + mr_range, min_mb, height - min_size - 1)

	top_k: List[dict] = []
	def push_top_k(item):
		top_k.append(item)
		top_k.sort(key=lambda c: -c["adjustedScore"])
		while len(top_k) > 5:
			top_k.pop()

	for sc in seed_candidates:
		push_top_k({"size": sc["size"], "x": sc["x"], "y": sc["y"],
						"adjustedScore": sc["confidence"] * min(1, math.sqrt(sc["size"] / 96))})

	for sz in scale_list:
		alpha, tgrad = get_template(sz)
		for mr in range(min_mr, max_mr + 1, 8):
			cx = width - mr - sz
			if cx < 0:
				continue
			for mb in range(min_mb, max_mb + 1, 8):
				cy = height - mb - sz
				if cy < 0:
					continue
				sc = _score_candidate(gray, grad, alpha, tgrad, cx, cy, sz)
				if not sc:
					continue
				adj = sc["confidence"] * min(1, math.sqrt(sz / 96))
				if adj < 0.08:
					continue
				push_top_k({"size": sz, "x": cx, "y": cy, "adjustedScore": adj})

	best = best_seed if best_seed else {
		"x": width - default_config["marginRight"] - default_config["logoSize"],
		"y": height - default_config["marginBottom"] - default_config["logoSize"],
		"size": default_config["logoSize"],
		"confidence": 0, "spatialScore": 0, "gradientScore": 0, "varianceScore": 0,
	}

	# 精搜索
	for coarse in top_k:
		scale_lo = _clamp(coarse["size"] - 10, min_size, max_size)
		scale_hi = _clamp(coarse["size"] + 10, min_size, max_size)
		for sz in range(scale_lo, scale_hi + 1, 2):
			alpha, tgrad = get_template(sz)
			for cx in range(coarse["x"] - 8, coarse["x"] + 9, 2):
				if cx < 0 or cx + sz > width:
					continue
				for cy in range(coarse["y"] - 8, coarse["y"] + 9, 2):
					if cy < 0 or cy + sz > height:
						continue
					sc = _score_candidate(gray, grad, alpha, tgrad, cx, cy, sz)
					if sc and sc["confidence"] > best["confidence"]:
						best = {"x": cx, "y": cy, "size": sz, **sc}

	return {
		"found": best["confidence"] >= threshold,
		"confidence": best["confidence"],
		"spatialScore": best["spatialScore"],
		"gradientScore": best["gradientScore"],
		"varianceScore": best["varianceScore"],
		"region": {"x": best["x"], "y": best["y"], "size": best["size"]},
	}


# ═══════════════════════════════════════════════════════════════════════════
# 9. 多遍去除
# ═══════════════════════════════════════════════════════════════════════════

def remove_repeated_layers(data: np.ndarray, width: int,
						   alpha_map: np.ndarray, position: dict,
						   max_passes: int = 4, alpha_gain: float = 1.0,
						   starting_pass: int = 0) -> dict:
	"""迭代去除水印层。"""
	current = data.copy()
	reference = current.copy()
	base_nbr = calculate_near_black_ratio(current, width, position)
	max_nbr = min(1.0, base_nbr + 0.05)
	passes = []
	stop_reason = "max-passes"
	applied = starting_pass

	for i in range(max_passes):
		before = score_region(current, width, alpha_map, position)
		candidate = current.copy()
		remove_watermark(candidate, width, alpha_map, position, alpha_gain)
		after = score_region(candidate, width, alpha_map, position)
		nbr = calculate_near_black_ratio(candidate, width, position)
		improvement = abs(before["spatialScore"]) - abs(after["spatialScore"])

		texture = assess_reference_texture(reference, candidate, width, position)
		if nbr > max_nbr:
			stop_reason = "safety-near-black"
			break
		if texture["hardReject"]:
			stop_reason = "safety-texture-collapse"
			break

		current = candidate
		applied = starting_pass + i + 1
		passes.append({
			"index": applied,
			"beforeSpatialScore": before["spatialScore"],
			"afterSpatialScore": after["spatialScore"],
			"improvement": improvement,
			"nearBlackRatio": nbr,
		})
		if abs(after["spatialScore"]) <= 0.25:
			stop_reason = "residual-low"
			break

	return {"imageData": current, "passCount": applied, "stopReason": stop_reason, "passes": passes}


# ═══════════════════════════════════════════════════════════════════════════
# 10. 主处理
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_GAIN_CANDIDATES = [1.05, 1.12, 1.2, 1.28, 1.36, 1.45, 1.52, 1.6, 1.7, 1.85, 2.0, 2.2, 2.4, 2.6]


def process_watermark(data: np.ndarray, width: int, height: int,
					  alpha48: np.ndarray, alpha96: np.ndarray,
					  adaptive_mode: str = "auto",
					  max_passes: int = 4) -> dict:
	"""
	主处理函数：检测 + 去除 + 多遍 + 校准。

	data: RGBA uint8 一维数组 (H*W*4)
	返回: {"imageData": np.ndarray, "meta": dict}
	"""
	original = data.copy()

	# 检测配置
	default_config = detect_watermark_config(width, height)

	# 选择 48 vs 96 — 通过互相关比较
	config = dict(default_config)
	for alt_size in (48, 96):
		if alt_size == config["logoSize"]:
			continue
		alt_cfg = {"logoSize": alt_size,
					"marginRight": 32 if alt_size == 48 else 64,
					"marginBottom": 32 if alt_size == 48 else 64}
		alt_pos = calculate_watermark_position(width, height, alt_cfg)
		if alt_pos["x"] < 0 or alt_pos["y"] < 0:
			continue
		if alt_pos["x"] + alt_pos["width"] > width or alt_pos["y"] + alt_pos["height"] > height:
			continue
		alt_alpha = alpha48 if alt_size == 48 else alpha96
		pri_pos = calculate_watermark_position(width, height, config)
		pri_alpha = alpha48 if config["logoSize"] == 48 else alpha96
		if pri_pos["x"] < 0 or pri_pos["y"] < 0:
			config = alt_cfg
			continue
		pri_score = compute_region_spatial_correlation(original, width, pri_alpha, pri_pos["x"], pri_pos["y"], pri_pos["width"])
		alt_score = compute_region_spatial_correlation(original, width, alt_alpha, alt_pos["x"], alt_pos["y"], alt_pos["width"])
		if alt_score >= 0.25 and alt_score > pri_score + 0.08:
			config = alt_cfg

	position = calculate_watermark_position(width, height, config)
	alpha_map = alpha48 if config["logoSize"] == 48 else alpha96

	# 标准检测
	spatial = compute_region_spatial_correlation(original, width, alpha_map, position["x"], position["y"], position["width"])
	gradient = compute_region_gradient_correlation(original, width, alpha_map, position["x"], position["y"], position["width"])
	has_standard = _has_reliable_standard_signal(spatial, gradient)

	# 自适应检测
	adaptive_result = None
	adaptive_confidence = None
	allow_adaptive = adaptive_mode not in ("never", "off")

	if not has_standard and allow_adaptive:
		adaptive_result = detect_adaptive_watermark(original, width, height, alpha96, config)
		adaptive_confidence = adaptive_result.get("confidence")

		if adaptive_result.get("found") and adaptive_confidence and adaptive_confidence >= 0.25:
			ar = adaptive_result["region"]
			sz = ar["size"]
			position = {"x": ar["x"], "y": ar["y"], "width": sz, "height": sz}
			config = {"logoSize": sz,
					  "marginRight": width - ar["x"] - sz,
					  "marginBottom": height - ar["y"] - sz}
			alpha_map = get_alpha_map(sz)
			spatial = adaptive_result["spatialScore"]
			gradient = adaptive_result["gradientScore"]
			has_standard = _has_reliable_adaptive_signal(adaptive_result)

	# 数据不足 → 跳过
	if not has_standard and spatial is not None and spatial < 0.15:
		# 尝试验证: 试做一次去除，看是否有改善
		test = original.copy()
		remove_watermark(test, width, alpha_map, position)
		test_spatial = compute_region_spatial_correlation(test, width, alpha_map, position["x"], position["y"], position["width"])
		improvement = abs(spatial) - abs(test_spatial)
		nbr_increase = calculate_near_black_ratio(test, width, position) - calculate_near_black_ratio(original, width, position)
		texture = assess_reference_texture(original, test, width, position)
		if improvement < 0.08 or nbr_increase > 0.05 or texture["hardReject"]:
			return {
				"imageData": original,
				"meta": {"applied": False, "skipReason": "no-watermark-detected",
						 "detection": {"originalSpatialScore": spatial, "originalGradientScore": gradient}},
			}

	# 第一遍去除
	result = original.copy()
	remove_watermark(result, width, alpha_map, position)

	# 多遍去除
	first_spatial = compute_region_spatial_correlation(result, width, alpha_map, position["x"], position["y"], position["width"])
	total_passes = max(1, max_passes)
	remaining = total_passes - 1
	extra = None
	if remaining > 0 and abs(first_spatial) > 0.25:
		extra = remove_repeated_layers(result, width, alpha_map, position, remaining, 1.0, 1)
		result = extra["imageData"]

	pass_count = extra["passCount"] if extra else 1

	# Alpha 增益重新校准
	processed_spatial = compute_region_spatial_correlation(result, width, alpha_map, position["x"], position["y"], position["width"])
	suppression = spatial - processed_spatial if spatial else 0
	alpha_gain = 1.0

	if spatial and spatial >= 0.6 and processed_spatial >= 0.5 and suppression <= 0.18:
		best_score = processed_spatial
		best_gain = 1.0
		best_data = None
		base_nbr = calculate_near_black_ratio(result, width, position)
		max_nbr = min(1.0, base_nbr + 0.05)

		for gain in ALPHA_GAIN_CANDIDATES:
			cand = result.copy()
			remove_watermark(cand, width, alpha_map, position, gain)
			cand_nbr = calculate_near_black_ratio(cand, width, position)
			if cand_nbr > max_nbr:
				continue
			sc = compute_region_spatial_correlation(cand, width, alpha_map, position["x"], position["y"], position["width"])
			if sc < best_score:
				best_score = sc
				best_gain = gain
				best_data = cand

		if best_data is not None and (processed_spatial - best_score) >= 0.18:
			result = best_data
			alpha_gain = best_gain
			processed_spatial = best_score
			suppression = spatial - processed_spatial

	final_spatial = compute_region_spatial_correlation(result, width, alpha_map, position["x"], position["y"], position["width"])
	final_gradient = compute_region_gradient_correlation(result, width, alpha_map, position["x"], position["y"], position["width"])

	return {
		"imageData": result,
		"meta": {
			"applied": True,
			"position": position,
			"config": config,
			"detection": {
				"originalSpatialScore": spatial,
				"originalGradientScore": gradient,
				"processedSpatialScore": final_spatial,
				"processedGradientScore": final_gradient,
				"suppressionGain": suppression,
				"adaptiveConfidence": adaptive_confidence,
			},
			"alphaGain": alpha_gain,
			"passCount": pass_count,
		},
	}


# ═══════════════════════════════════════════════════════════════════════════
# 11. 高级 API
# ═══════════════════════════════════════════════════════════════════════════

def remove_watermark_from_array(rgba: np.ndarray,
								adaptive_mode: str = "auto",
								max_passes: int = 4) -> dict:
	"""
	从 RGBA numpy 数组去除水印。

	rgba: shape (H, W, 4) 的 uint8 数组
	返回: {"image": np.ndarray (H, W, 4), "meta": dict}
	"""
	h, w = rgba.shape[:2]
	data = rgba.flatten().astype(np.uint8)
	alpha48 = get_alpha_map(48)
	alpha96 = get_alpha_map(96)
	result = process_watermark(data, w, h, alpha48, alpha96, adaptive_mode, max_passes)
	return {
		"image": result["imageData"].reshape(h, w, 4),
		"meta": result["meta"],
	}


def remove_watermark_from_file(input_path: str, output_path: Optional[str] = None,
							   adaptive_mode: str = "auto",
							   max_passes: int = 4) -> dict:
	"""
	从图片文件去除水印。

	input_path: 输入图片路径
	output_path: 输出路径（None 则自动生成 _clean 后缀）
	返回: {"output_path": str, "meta": dict}
	"""
	from PIL import Image

	img = Image.open(input_path).convert("RGBA")
	rgba = np.array(img, dtype=np.uint8)
	result = remove_watermark_from_array(rgba, adaptive_mode, max_passes)

	if output_path is None:
		p = Path(input_path)
		output_path = str(p.with_stem(p.stem + "_clean"))

	out_img = Image.fromarray(result["image"], "RGBA")
	# 根据输出格式决定是否转 RGB
	ext = Path(output_path).suffix.lower()
	if ext in (".jpg", ".jpeg"):
		out_img = out_img.convert("RGB")
	out_img.save(output_path)

	return {"output_path": output_path, "meta": result["meta"]}


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
	parser = argparse.ArgumentParser(
		description="Gemini Watermark Remover — 从 Gemini AI 生成的图片中去除水印",
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("input", nargs="+", help="输入图片路径或目录")
	parser.add_argument("-o", "--output", help="输出路径（单文件）或输出目录（批量）")
	parser.add_argument("--suffix", default="_clean", help="自动输出文件名后缀 (默认: _clean)")
	parser.add_argument("--adaptive", choices=["auto", "always", "never"], default="auto",
						help="自适应检测模式 (默认: auto)")
	parser.add_argument("--max-passes", type=int, default=4, help="最大处理遍数 (默认: 4)")
	parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")
	args = parser.parse_args()

	from PIL import Image

	input_files: List[Path] = []
	for inp in args.input:
		p = Path(inp)
		if p.is_dir():
			for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
				input_files.extend(p.glob(ext))
		elif p.is_file():
			input_files.append(p)
		else:
			# glob pattern
			import glob
			input_files.extend(Path(f) for f in glob.glob(inp))

	if not input_files:
		print("未找到输入文件", file=sys.stderr)
		sys.exit(1)

	output_dir = None
	if args.output and len(input_files) > 1:
		output_dir = Path(args.output)
		output_dir.mkdir(parents=True, exist_ok=True)

	for i, fp in enumerate(input_files):
		if output_dir:
			out_path = str(output_dir / fp.name)
		elif args.output and len(input_files) == 1:
			out_path = args.output
		else:
			out_path = str(fp.with_stem(fp.stem + args.suffix))

		try:
			result = remove_watermark_from_file(str(fp), out_path, args.adaptive, args.max_passes)
			meta = result["meta"]
			if not args.quiet:
				status = "已去除" if meta.get("applied") else "跳过"
				det = meta.get("detection", {})
				score = det.get("originalSpatialScore")
				score_str = f" (score={score:.3f})" if score is not None else ""
				print(f"[{i+1}/{len(input_files)}] {status}{score_str}: {fp.name} → {Path(out_path).name}")
		except Exception as e:
			print(f"[{i+1}/{len(input_files)}] 错误: {fp.name}: {e}", file=sys.stderr)

	if not args.quiet:
		print(f"完成，共处理 {len(input_files)} 个文件")


# ═══════════════════════════════════════════════════════════════════════════
# 13. 嵌入式 Alpha Map 完整 Base64 数据
#     从 src/core/embeddedAlphaMaps.js 提取
# ═══════════════════════════════════════════════════════════════════════════


def _init_alpha_data():
	"""初始化完整的 Alpha Map 数据（从 JS embeddedAlphaMaps.js 原样复制）。"""
	global _ALPHA_MAP_BASE64

	_ALPHA_MAP_BASE64[48] = (
		"gYAAPIGAgDuBgIA7AAAAAAAAAAAAAAAAAAAAAIGAgDsAAAAAAAAAAAAAAAAAAAAAgYCAO4GAgDsA"
		"AAAAAAAAAIGAgDuBgIA7gYCAOwAAAAAAAAAAgYCAOwAAAADj4uI+4eDgPoGAgDuBgIA7gYCAO4GA"
		"gDuBgIA7gYAAPIGAgDuBgIA7gYAAPIGAgDuBgIA7gYAAPMHAQDyBgIA7gYCAO4GAgDuBgIA7gYAA"
		"PIGAgDvBwEA8gYAAPIGAgDuBgIA7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgYCAO4GAgDsAAAAA"
		"AAAAAAAAAAAAAAAAAAAAAIGAgDuBgIA7gYCAOwAAAAAAAAAAAAAAAIGAgDsAAAAAgYCAO4WEBD6B"
		"gAA/gYAAP4GAAD4AAAAAgYAAPAAAAACBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGA"
		"ADyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAADyBgAA8wcBAPIGAgDuBgIA7gYCAOwAAAAAAAAAAAAAA"
		"AAAAAAAAAAAAAAAAAIGAgDsAAAAAgYCAO4GAADyBgAA8gYCAOwAAAAAAAAAAgYAAPIGAgDsAAAAA"
		"AAAAAIGAgDsAAAAAgYCAO5GQkD6BgAA/gYAAP5GQkD4AAAAAgYCAOwAAAACBgIA7gYCAO4GAgDuB"
		"gAA8gYAAPAAAAACBgAA8wcBAPMHAQDyBgIA7gYCAO4GAADyBgAA8gYAAPMHAQDyBgIA7gYCAO4GA"
		"gDuBgIA7gYCAO4GAADwAAAAAAAAAAIGAgDsAAAAAAAAAAIGAgDsAAAAAgYCAO4GAgDuBgIA7gYCA"
		"O4GAgDuBgIA7gYCAO4GAgDuBgAA8gYCAO4GAgDsAAAAAgYCAO+Hg4D6BgAA/gYAAP/Hw8D4AAAAA"
		"gYCAO4GAgDuBgIA7gYAAPIGAgDuBgAA8wcBAPIGAgDuBgIA7gYAAPIGAADyBgIA7gYCAO4GAADyB"
		"gAA8gYAAPIGAgDuBgIA7gYCAO4GAADyBgAA8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIGA"
		"gDsAAAAAAAAAAIGAgDuBgIA7AAAAAAAAAACBgIA7AAAAAAAAAAAAAAAAAAAAAIGAgDsAAAAAgYAA"
		"PoGAAD+BgAA/gYAAP4GAAD+BgAA+AAAAAAAAAACBgIA7gYAAPAAAAACBgAA8gYAAPIGAgDuBgIA7"
		"gYCAOwAAAADBwEA8wcBAPIGAADyBgAA8gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7AAAAAIGAADwA"
		"AAAAAAAAAIGAgDsAAAAAgYCAO4GAgDuBgIA7gYCAOwAAAAAAAAAAgYCAOwAAAAAAAAAAAAAAAAAA"
		"AACBgIA7AAAAAAAAAACBgIA7oaCgPoGAAD+BgAA/gYAAP4GAAD/BwMA+AAAAAAAAAACBgIA7AAAA"
		"AIGAgDuBgAA8gYAAPAAAAACBgIA7gYCAO4GAgDuBgAA8wcBAPMHAQDzBwEA8gYCAO4GAgDsAAAAA"
		"AAAAAIGAgDuBgIA7gYCAOwAAAAAAAAAAAAAAAIGAgDsAAAAAwcBAPIGAgDuBgIA7gYCAOwAAAAAA"
		"AAAAgYCAO4GAgDuBgIA7gYAAPIGAADwAAAAAAAAAAIGAADyJiIg9gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYCAPQAAAACBgIA7AAAAAAAAAAAAAAAAgYCAO4GAADyBgAA8gYCAO4GAgDuBgIA7gYCA"
		"O4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7AAAAAAAAAAAAAAAAAAAAAIGAADwAAAAA"
		"gYCAO4GAADyBgIA7gYCAOwAAAAAAAAAAgYCAO8HAQDyBgIA7gYCAO4GAgDsAAAAAgYCAO4GAgDuh"
		"oKA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/o6KiPoGAgDuBgAA8AAAAAIGAgDuBgIA7gYCAO8HA"
		"QDyBgAA8gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYAA"
		"PAAAAAAAAAAAgYCAO4GAADyBgIA7gYAAPIGAgDuBgIA7gYAAPIGAADyBgIA7gYCAO4GAgDuBgAA8"
		"gYCAO4GAADyBgAA8gYCAO4mIiD2BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD6B"
		"gAA8gYCAO4GAADwAAAAAgYCAO4GAADyBgIA7wcBAPIGAADyBgAA8wcBAPMHAQDzBwEA8gYAAPIGA"
		"ADyBgIA7gYCAO4GAADyBgAA8gYCAOwAAAAAAAAAAgYCAO4GAgDuBgIA7AAAAAIGAADyBgIA7AAAA"
		"AIGAgDuBgIA7AAAAAIGAgDsAAAAAgYCAOwAAAACBgIA7gYCAO+Hg4D6BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP8HAwD6BgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDvB"
		"wEA8wcBAPMHAQDyBgAA8wcBAPIGAADyBgIA7gYCAO4GAADyBgAA8gYAAPIGAgDsAAAAAAAAAAIGA"
		"gDuBgAA8AAAAAIGAgDuBgIA7AAAAAAAAAAAAAAAAgYAAPIGAgDuBgIA7gYAAPAAAAACBgIA7gYCA"
		"PoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/h4GA+AAAAAAAAAACBgAA8"
		"gYAAPMHAQDyBgIA7gYAAPIGAADyBgIA7gYAAPIGAADyBgAA8gYCAO4GAgDuBgAA8gYAAPIGAgDuB"
		"gIA7AAAAAAAAAACBgIA7AAAAAAAAAACBgIA7gYCAO8HAQDwAAAAAgYCAO4GAADwAAAAAgYAAPAAA"
		"AACBgAA8gYCAOwAAAACBgIA9gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD/x8PA+wcDAPYGAgDuBgAA8wcBAPIGAADyBgAA8gYAAPIGAADwAAAAAgYCAO4GAgDuBgIA7"
		"gYCAO4GAADyBgAA8gYAAPIGAgDuBgIA7gYCAOwAAAACBgIA7gYCAOwAAAAAAAAAAAAAAAIGAgDsA"
		"AAAAgYCAO4GAgDuBgIA7AAAAAMHAQDyBgAA8gYCAO4GAgD3h4OA+gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/4eDgPoGAAD2BgIA7gYCAOwAAAACBgIA7gYCA"
		"O4GAgDuBgAA8gYAAPIGAgDuBgIA7gYAAPIGAADyBgAA8gYAAPIGAgDuBgAA8gYCAOwAAAACBgIA7"
		"AAAAAAAAAACBgIA7AAAAAIGAgDsAAAAAgYCAOwAAAACBgIA7gYCAO4GAgDuBgIA7gYCAO9PS0j6B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP8HA"
		"wD6BgAA8gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYAAPIGAADyBgAA8gYAA"
		"PIGAgDuBgAA8gYCAO4GAgDuBgIA7AAAAAAAAAACBgIA7AAAAAAAAAAAAAAAAAAAAAIGAgDsAAAAA"
		"gYAAPIGAgDuBgIA7o6KiPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+hoKA+gYCAOwAAAACBgIA7gYCAO4GAgDvBwEA8gYCAO4GA"
		"gDuBgIA7gYCAO4GAADyBgAA8gYAAPIGAgDuBgIA7AAAAAAAAAAAAAAAAgYCAO4GAgDsAAAAAAAAA"
		"AIGAgDsAAAAAgYCAOwAAAAAAAAAAgYCAO4GAgDuhoKA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/oaCgPgAAAACB"
		"gIA7gYCAO4GAgDvBwEA8gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7AAAAAIGA"
		"ADwAAAAAgYCAO4GAgDsAAAAAAAAAAIGAgDsAAAAAAAAAAAAAAACBgIA7gYAAPcHAwD6BgAA/gYAA"
		"P4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP9HQ0D6BgIA9gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7wcBAPIGAgDzB"
		"wEA8gYAAPAAAAACBgIA7gYCAOwAAAACBgIA7gYCAOwAAAACBgIA7gYCAO4GAgDuBgIA7AAAAAAAA"
		"AADBwMA94eDgPoGAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/h4OA+iYiIPYGAgDyBgAA8"
		"gYCAO4GAgDuBgIA7gYAAPIGAgDuBgAA8gYAAPIGAgDuBgIA7gYCAOwAAAAAAAAAAgYCAO4GAgDsA"
		"AAAAAAAAAIGAgDsAAAAAAAAAAOHgYD7x8PA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4WEhD6BgIA7gYCAO4GAADzBwEA8gYAAPMHAQDzBwEA8gYCAO4GAgDsAAAAA"
		"gYAAPIGAgDsAAAAAgYCAOwAAAAAAAAAAgYAAPIGAgDuBgAA+wcDAPoGAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/h4OA+gYCAPYGAADzBwEA8gYAA"
		"PIGAADyBgAA8gYAAPIGAgDuBgIA7AAAAAIGAgDsAAAAAAAAAAAAAAAAAAAAAgYCAPaOioj6BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP6GgoD6BgIA9gYAAPIGAgDuBgAA8gYAAPIGAgDuBgIA7AAAAAIGAgDsAAAAAgYCA"
		"O4WEBD7BwMA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/oaCgPoGAAD6BgIA7gYAAPIGA"
		"gDuBgIA7AAAAAIGAAD6RkJA+8fDwPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD/h4OA+kZCQPoGAAD6BgIA84eDgPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/j4uI+4eDgPoGAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD/h4OA+gYCAO4GAAD6RkJA+4eDgPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD/x8PA+kZCQPoGAAD6BgIA7gYCAO8HAQDwAAAAAgYCAO4GAAD6hoKA+"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/wcDAPoGAAD6BgAA8gYAAPIGAgDuBgIA7AAAA"
		"AIGAgDuBgIA7AAAAAAAAAACBgAA8gYCAPaOioj6BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP6Oioj6JiIg9gYCAO4GA"
		"gDuBgAA8gYAAPIGAgDuBgIA7gYCAO4GAADyBgIA7gYCAOwAAAAAAAAAAgYCAO4GAADyBgIA94eDg"
		"PoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/B"
		"wMA+gYAAPoGAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAOwAA"
		"AAAAAAAAAAAAAIGAgDsAAAAAgYCAO4GAgD6BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/8/LyPuXkZD6BgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuB"
		"gIA7gYCAO4GAgDsAAAAAgYCAOwAAAACBgIA7AAAAAAAAAAAAAAAAgYAAPAAAAACBgIA94+LiPoGA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD/h4OA+wcDAPYGAgDuBgIA7gYCAO4GAADyBgAA8"
		"gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYCAO4GAgDsAAAAAgYCAOwAAAACBgIA7AAAAAIGAgDsA"
		"AAAAAAAAAAAAAAAAAAAAgYCAPdHQ0D6BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP8HAwD6BgAA9gYCA"
		"O4GAgDuBgIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYCAO4GAgDuBgIA7"
		"gYCAO4GAADyBgIA7gYCAO4GAgDsAAAAAAAAAAAAAAAAAAAAAgYCAO4GAgDuhoKA+gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GA"
		"AD+BgAA/oaCgPsHAQDzBwEA8gYAAPIGAgDuBgAA8gYAAPIGAgDuBgIA7wcBAPMHAQDyBgAA8gYAA"
		"PIGAgDsAAAAAgYCAO4GAgDsAAAAAgYCAO4GAgDuBgIA7gYCAOwAAAAAAAAAAAAAAAAAAAACBgIA7"
		"gYCAO4GAADyBgIA7oaCgPoGAAD+BgAA/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+hoKA+gYCAO4GAgDyBgIA8gYAAPIGAADyBgAA8gYAAPIGA"
		"gDuBgIA7wcBAPMHAQDyBgIA7gYAAPIGAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgAA8wcBA"
		"PMHAQDyBgAA8gYAAPIGAADyBgAA8gYCAO4GAgDuBgIA7gYCAO8PCwj6BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP9HQ0D6BgAA8gYAAPMHAQDzB"
		"wEA8gYCAOwAAAACBgIA7gYAAPIGAADyBgAA8wcBAPMHAQDyBgIA7gYCAO8HAQDzBwEA8gYCAO4GA"
		"gDuBgAA8gYAAPIGAgDuBgIA7gYCAPMHAQDyBgAA8wcBAPIGAADzBwEA8gYCAO4GAgDuBgIA7gYCA"
		"O6GgID3j4uI+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"4eDgPoGAgD2BgAA8wcBAPMHAQDzBwEA8gYCAO4GAgDuBgIA7gYAAPIGAADyBgAA8gYCAPMHAQDwA"
		"AAAAgYCAO8HAQDzBwEA8gYAAPIGAADyBgAA8gYCAO4GAADyBgAA8AAAAAAAAAACBgAA8wcBAPIGA"
		"ADzBwEA8gYAAPIGAADwAAAAAgYCAO4GAADzJyMg98fDwPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYCAPQAAAACBgAA8gYAAPAAAAAAAAAAAgYCAO4GAgDuBgAA8"
		"wcBAPIGAADyBgIA7AAAAAAAAAACBgIA7gYCAO4GAgDuBgIA7gYAAPIGAADyBgAA8gYAAPMHAQDyB"
		"gAA8gYCAO4GAgDuBgAA8gYAAPIGAADyBgAA8gYAAPIGAADyBgIA7gYCAO4GAADyBgAA84eBgPoGA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgIA+gYCAO4GAgDvBwEA8gYAA"
		"PIGAgDsAAAAAgYCAO4GAgDvBwEA8wcBAPIGAgDuBgAA8gYCAOwAAAAAAAAAAAAAAAIGAgDuBgIA7"
		"gYAAPIGAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7AAAAAIGAgDuB"
		"gAA8gYAAPIGAADyBgAA8AAAAAMHAwD6BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP+Hg"
		"4D6BgIA7gYCAO4GAgDuBgAA8gYAAPAAAAACBgAA8gYCAO4GAgDuBgAA8gYCAO4GAgDsAAAAAgYCA"
		"OwAAAAAAAAAAgYCAO4GAgDuBgIA7gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7"
		"gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPMHAQDyBgAA8gYCAO4GAAD6BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAgD2BgIA7gYCAOwAAAACBgAA8gYAAPIGAgDuBgAA8gYCAO4GA"
		"gDuBgIA7gYCAO4GAgDuBgAA8gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCA"
		"O4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPAAAAAAAAAAAgYCAOwAAAACBgAA8gYAAPIGAADyBgIA7"
		"gYCAOwAAAAChoKA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/oaCgPoGAgDuBgIA7gYCAO4GAgDuB"
		"gAA8wcBAPAAAAACBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYAAPIGAADyBgIA7gYCAO4GA"
		"gDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPAAAAACBgIA7gYCA"
		"O4GAADyBgAA8gYAAPIGAgDuBgIA7AAAAAIGAgDuBgIA9gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYCAPYGAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuB"
		"gIA7gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYAAPIGAADyBgAA8gYAAPIGAADyBgAA8gYCAO4GA"
		"gDuBgIA7gYAAPIGAgDuBgIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgIA7gYCAO4GAgDsAAAAAw8LC"
		"PoKBAT+CgQE/gYAAP4GAAD+hoKA+gYCAO4GAADyBgAA8gYAAPIGAgDuBgIA7gYCAO4GAgDuBgIA7"
		"gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYAAPIGAADyB"
		"gAA8gYAAPIGAADyBgAA8gYCAO4GAgDuBgIA7gYCAO4GAgDsAAAAAgYAAPIGAADyBgIA7AAAAAIGA"
		"gDuBgIA7gYAAPMHAQDyBgIA7gYAAPoKBAT+BgAA/gYAAP4GAAD+BgAA+gYCAO4GAADyBgAA8AAAA"
		"AIGAgDuBgIA7gYCAO4GAgDuBgIA7gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8"
		"gYAAPIGAgDuBgIA7gYAAPIGAADyBgAA8gYAAPIGAgDuBgIA7gYAAPIGAgDuBgIA7gYCAO4GAgDuB"
		"gIA7gYAAPIGAADyBgAA8gYAAPAAAAAAAAAAAgYAAPIGAADyBgIA7gYCAO/Py8j6BgAA/gYAAP+Hg"
		"4D6BgIA7gYCAO4GAADzBwEA8gYCAO4GAgDuBgAA8gYAAPAAAAAAAAAAAgYCAO4GAgDuBgIA7gYCA"
		"O4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYAAPIGAADyBgAA8gYAAPIGAgDsAAAAA"
		"gYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7gYAAPIGAADyBgAA8gYAAPIGAgDuBgIA7gYAAPIGAADyB"
		"gIA7gYCAO5OSkj6BgAA/gYAAP5OSkj6BgIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgIA7gYAAPIGA"
		"gDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7gYCA"
		"O4GAgDuBgIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgAA8gYCAO4GAADyBgIA7AAAAAIGAgDvBwEA8"
		"wcBAPIGAgDsAAAAAgYCAO4GAgDuBgAA8gYAAPIGAAD6BgAA/gYAAP4WEBD6BgIA7gYCAO4GAADyB"
		"gAA8gYAAPIGAADwAAAAAgYCAOwAAAACBgIA7gYAAPIGAADyBgIA7gYCAO4GAADyBgAA8gYCAO4GA"
		"gDuBgAA8gYAAPIGAgDuBgIA7gYCAO4GAgDuBgIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgIA7AAAA"
		"AIGAgDuBgIA7gYCAO4GAgDuBgIA7gYAAPIGAgDuBgIA7gYCAOwAAAADBwEA8gYAAPIGAgDvh4OA+"
		"4eDgPoGAgDuBgIA7gYCAO4GAADyBgAA8gYAAPIGAADyBgIA7AAAAAIGAgDsAAAAAgYAAPIGAADyB"
		"gIA7gYCAO4GAADyBgAA8gYCAO4GAgDuBgAA8gYAAPIGAgDuBgIA7"
	)

	_ALPHA_MAP_BASE64[96] = (
		"gYCAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGAgDwAAAAAwcBAPMHAQDyBgIA8wcBAPIGAADyh"
		"oKA8gYAAPIGAADyBgAA8AAAAAIGAADyBgIA7gYCAO4GAgDuBgIA7wcBAPIGAADzBwEA8gYCAPIGA"
		"ADzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA8wcBAPIGAADwAAAAAgYCAO4GAgDuBgIA7gYCA"
		"O4GAgDuBgAA8gYAAPOXkZD7z8vI+4+LiPu3sbD6BgIA7wcBAPAAAAACBgAA8gYCAO4GAADyBgIA8"
		"gYCAPIGAADyBgIA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8gYAAPIGAgDyBgIA8gYCAPIGAADzB"
		"wEA8gYCAO8HAQDyBgAA8gYAAPIGAgDuBgAA8wcBAPMHAQDyBgAA8wcBAPMHAQDyBgIA8wcBAPMHA"
		"QDyBgIA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgIA8oaCgPIGAgDyBgIA8wcBAPIGAgDyBgAA8wcBA"
		"PIGAgDvBwEA8wcBAPIGAgDyBgAA8gYAAPMHAQDzBwEA8gYCAO4GAADyBgIA8wcBAPIGAADyBgAA8"
		"gYCAO4GAgDsAAAAAgYCAOwAAAAAAAAAAgYAAPIGAADzBwEA8wcBAPIGAgDvBwEA8wcBAPIGAADzB"
		"wEA8oaCgPMHAQDzBwEA8gYCAOwAAAAAAAAAAAAAAAIGAgDsAAAAAAAAAAAAAAACBgAA8gYCAPYGA"
		"AD+BgAA/goEBP4KBAT+RkBA9gYAAPAAAAACBgIA7gYAAPIGAADyBgAA8wcBAPMHAQDyBgAA8gYAA"
		"PIGAADzBwEA8gYCAPMHAQDzBwEA8gYCAO4GAgDuBgAA8gYAAPIGAADyBgIA7gYAAPAAAAACBgAA8"
		"AAAAAIGAgDuBgIA7gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYAAPMHAQDyBgIA8wcBAPMHAQDzB"
		"wEA8wcDAPKGgoDyBgIA8oaCgPIGAgDyhoKA8gYCAOwAAAACBgAA8gYCAO4GAADyBgIA8wcBAPIGA"
		"ADwAAAAAgYCAO4GAgDvBwEA8gYCAO4GAgDuBgIA7gYCAO4GAADyBgAA8gYCAOwAAAACBgIA7gYCA"
		"OwAAAACBgIA7gYCAO4GAgDsAAAAAgYCAO4GAADyBgAA8gYAAPMHAQDzBwEA8gYCAPMHAQDzBwEA8"
		"AAAAAIGAgDsAAAAAgYCAOwAAAAAAAAAAgYCAO4GAADyBgAA8paQkPoGAAD+BgAA/goEBP4GAAD/F"
		"xEQ+AAAAAAAAAACBgIA7gYAAPIGAgDsAAAAAAAAAAIGAgDuBgIA7gYAAPKGgoDyBgIA8wcBAPIGA"
		"gDyBgIA8gYAAPAAAAACBgIA7gYAAPIGAADzBwEA8gYCAPIGAgDyBgAA8wcBAPKGgoDzBwEA8gYAA"
		"PIGAADyBgIA8gYCAPMHAQDzBwEA8wcBAPIGAgDyBgIA8gYCAPMHAQDyBgAA8gYCAPMHAQDyBgIA8"
		"gYCAPMHAQDzBwEA8AAAAAIGAgDuBgIA7gYAAPIGAADyhoKA8gYAAPMHAQDwAAAAAgYCAO4GAADyB"
		"gAA8AAAAAAAAAAAAAAAAgYAAPIGAADyBgIA7AAAAAIGAgDsAAAAAgYCAO4GAgDsAAAAAgYCAO4GA"
		"ADyBgIA7gYCAOwAAAACBgIA7gYCAPMHAQDyBgAA8gYCAPIGAgDuBgAA8AAAAAIGAgDsAAAAAAAAA"
		"AAAAAACBgIA7gYAAPIGAADzBwEA8paSkPoGAAD+BgAA/goEBP4KBAT+hoKA+gYCAO4GAgDuBgIA7"
		"gYCAO4GAgDsAAAAAAAAAAIGAgDuBgAA8gYCAPIGAgDzBwEA8gYAAPMHAQDyBgIA8gYCAPIGAADyB"
		"gAA8wcBAPIGAADzBwEA8gYCAPIGAgDyBgIA8gYCAPMHAQDyBgIA8gYAAPMHAQDyBgIA8oaCgPIGA"
		"gDyBgAA8gYCAO8HAQDzBwEA8oaCgPIGAgDzBwEA8gYCAPMHAQDyhoKA8gYCAPMHAQDzBwEA8AAAA"
		"AAAAAACBgIA7wcBAPKGgoDyBgIA8gYCAPMHAQDyBgIA7gYAAPIGAgDuBgIA7AAAAAAAAAADBwEA8"
		"gYCAO8HAQDzBwEA8AAAAAIGAADyBgIA7gYCAOwAAAACBgAA8AAAAAIGAgDsAAAAAAAAAAAAAAACB"
		"gAA8gYCAPMHAQDyBgAA8gYAAPMHAQDyBgAA8gYAAPIGAgDuBgIA8wcBAPAAAAACBgIA7AAAAAIGA"
		"gDvBwEA85eTkPoGAAD+BgAA/gYAAP4GAAD/z8vI+gYAAPMHAQDyBgIA7gYAAPMHAQDyBgIA7gYAA"
		"PMHAQDzBwEA8gYCAPIGAgDyBgIA8wcBAPIGAgDyBgIA8wcBAPIGAADyBgIA7gYCAO8HAQDyBgIA8"
		"oaCgPIGAgDyBgIA8gYCAPIGAgDyBgIA8gYCAPMHAQDzBwEA8wcBAPIGAgDsAAAAAwcBAPIGAADzB"
		"wEA8gYCAPMHAQDzBwEA8gYAAPKGgoDyBgIA8oaCgPIGAgDzBwEA8gYCAOwAAAADBwEA8wcBAPMHA"
		"QDyBgIA8gYCAPIGAgDzBwEA8gYAAPIGAgDyBgIA8AAAAAIGAgDuBgIA7gYCAO4GAADzBwEA8gYAA"
		"PIGAgDsAAAAAgYAAPIGAADyBgIA7gYCAO4GAgDuBgIA7AAAAAIGAgDuBgAA8gYCAPIGAgDyBgIA8"
		"gYCAPIGAgDuBgIA8oaCgPIGAADyBgIA8AAAAAIGAADyBgIA8gYAAPIGAADyFhAQ+goEBP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/iYgIPoGAADyBgAA8gYAAPMHAQDzBwEA8gYAAPIGAgDuBgAA8wcBAPIGA"
		"gDzBwEA8wcBAPMHAQDyBgIA8wcBAPIGAADyBgAA8gYCAO8HAQDzBwEA8wcBAPIGAgDyBgIA8gYCA"
		"PKGgoDzBwEA8gYAAPMHAQDyBgIA8gYCAPIGAADyBgIA7wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8"
		"gYCAPIGAgDzBwEA8gYCAPMHAQDzBwEA8gYCAO4GAADyBgIA7gYCAO4GAgDyBgIA8wcBAPMHAQDyB"
		"gIA8wcBAPAAAAACBgIA7AAAAAIGAADyBgAA8gYCAO4GAgDvBwEA8gYCAO4GAgDuBgAA8wcBAPMHA"
		"QDzBwEA8wcBAPIGAADwAAAAAgYCAO4GAADzBwEA8gYAAPIGAADyBgAA8wcBAPIGAADyBgIA7gYCA"
		"PIGAADyBgIA7gYAAPIGAgDuBgAA8gYAAPIGAADylpKQ+goEBP4GAAD+BgAA/gYAAP4KBAT+BgAA/"
		"k5KSPoGAADyBgAA8gYAAPIGAADyBgIA8wcBAPIGAADyBgIA7gYAAPIGAADzBwEA8wcBAPKGgoDzB"
		"wEA8gYCAPIGAADyBgAA8wcBAPIGAADzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDyBgAA8gYAAPMHA"
		"QDzBwEA8wcBAPIGAADzBwEA8gYAAPMHAQDzBwMA8oaCgPIGAgDyBgIA8wcBAPMHAQDzBwEA8gYCA"
		"PMHAQDzBwEA8gYCAO8HAQDyBgAA8gYAAPMHAQDzBwEA8gYAAPIGAgDyBgIA8gYAAPIGAADyBgIA7"
		"gYAAPMHAQDyBgIA8gYCAO4GAgDsAAAAAgYAAPMHAQDzBwEA8wcBAPIGAgDuBgAA8wcBAPAAAAACB"
		"gAA8gYAAPIGAADyBgAA8gYCAO4GAgDvBwEA8gYCAO4GAgDuBgIA7AAAAAIGAgDuBgIA7gYCAO4GA"
		"gDuBgAA8gYAAPIGAgDvj4uI+goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/8/LyPoGAADyBgAA8gYAA"
		"PIGAADyBgIA8wcBAPMHAQDyBgIA8wcBAPIGAADzBwEA8gYAAPMHAQDzBwEA8gYCAO4GAADyBgAA8"
		"gYAAPIGAADyBgIA7wcBAPIGAgDzBwEA8wcBAPIGAgDyBgAA8gYCAO8HAQDzBwEA8gYAAPMHAQDyB"
		"gIA8wcBAPIGAgDzBwEA8oaCgPKGgoDyBgIA8gYCAPIGAgDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGA"
		"ADyBgAA8gYAAPIGAgDuBgAA8gYCAPMHAQDyBgIA7wcBAPIGAADzBwEA8gYAAPIGAADyBgIA8gYCA"
		"O4GAgDuBgAA8wcBAPIGAgDyBgIA7gYAAPIGAADyBgAA8gYAAPAAAAACBgAA8gYAAPIGAADzBwEA8"
		"gYAAPMHAQDyBgIA8wcBAPIGAADyBgIA7gYCAO4GAgDuBgAA8gYCAO4GAgDzBwEA8wcBAPKmoKD6B"
		"gAA/goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4WEBD7BwEA8gYAAPIGAgDvBwEA8wcBAPIGA"
		"ADzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADzBwEA8gYAAPIGAgDuBgIA7gYCAO4GAgDvBwEA8wcBA"
		"PIGAADyBgIA8gYAAPIGAgDsAAAAAgYCAO4GAgDuBgIA8wcBAPMHAQDyBgIA8wcBAPIGAADyBgIA8"
		"oaCgPIGAgDyBgAA8gYAAPIGAgDyBgIA8oaCgPIGAgDyBgIA8gYCAO8HAQDyBgIA7gYCAOwAAAACB"
		"gIA7wcBAPIGAADyBgIA7wcBAPIGAgDuBgAA8gYAAPMHAQDyBgAA8gYAAPIGAgDuBgAA8wcBAPIGA"
		"gDyBgAA8wcBAPAAAAACBgIA7AAAAAIGAADyBgAA8gYAAPIGAADyBgAA8gYAAPIGAADzBwEA8wcBA"
		"PIGAgDuBgIA7AAAAAIGAgDuBgIA7gYCAO4GAADzBwEA8wcBAPLOysj6CgQE/gYAAP4GAAD+BgAA/"
		"goEBP4KBAT+BgAA/gYAAP8PCwj6BgAA8AAAAAIGAgDvBwEA8gYCAPIGAgDzBwEA8wcBAPMHAQDyB"
		"gAA8gYAAPMHAQDyBgAA8gYCAO4GAgDsAAAAAgYAAPIGAADzBwEA8gYAAPMHAQDyBgAA8gYCAPAAA"
		"AACBgIA7gYAAPIGAADzBwEA8gYCAPIGAgDyBgIA8gYCAPIGAgDyhoKA8gYCAPIGAADyBgIA8wcBA"
		"PMHAQDzBwEA8wcDAPIGAgDyBgIA8wcBAPIGAADyBgIA8wcBAPIGAgDuBgIA7gYCAO4GAgDvBwEA8"
		"wcBAPIGAADyBgAA8gYCAO4GAgDuBgIA7gYAAPMHAQDzBwEA8gYAAPIGAgDyBgIA7gYCAO4GAgDuB"
		"gIA7gYCAO8HAQDyBgAA8gYCAPIGAADyBgAA8gYCAO4GAgDzBwEA8wcBAPIGAgDuBgIA7AAAAAIGA"
		"gDsAAAAAgYCAO4GAADyBgIA8iYiIPYGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAA"
		"P4GAAD+JiIg9gYCAO4GAADyBgAA8gYAAPMHAQDyBgIA7gYAAPIGAADyBgAA8gYCAO4GAgDuBgIA7"
		"gYCAO8HAQDyBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDuBgAA8gYAAPIGAADyBgIA8gYCAPMHAQDzB"
		"wEA8wcBAPMHAQDyBgAA8gYCAPIGAgDzBwEA8gYAAPMHAQDzBwEA8wcBAPIGAgDyBgAA8wcBAPIGA"
		"ADzBwEA8wcBAPIGAADzBwEA8wcBAPIGAgDuBgIA7gYCAO4GAgDuBgAA8wcBAPIGAgDuBgAA8gYAA"
		"PIGAgDuBgIA7gYAAPMHAQDzBwEA8gYAAPIGAADyBgIA7wcBAPIGAgDsAAAAAgYAAPIGAgDzBwEA8"
		"wcBAPIGAADyBgAA8wcBAPMHAQDyBgAA8wcBAPIGAgDuBgAA8gYCAO4GAgDuBgIA7gYCAO4GAADzB"
		"wEA8oaCgPoGAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4KBAT+BgAA/gYAAP4GAAD+FhIQ+gYAAPIGA"
		"ADyBgIA7AAAAAIGAADyBgIA8wcBAPIGAgDyBgAA8gYAAPIGAADyBgAA8gYCAO4GAADyBgIA7gYCA"
		"O8HAQDyBgAA8gYAAPMHAQDwAAAAAgYAAPIGAgDuBgIA8gYCAPIGAgDzBwEA8wcBAPIGAADyBgAA8"
		"wcBAPIGAgDyBgIA7wcBAPIGAgDvBwEA8gYAAPMHAQDzBwEA8wcBAPIGAgDyhoKA8gYCAO4GAgDuB"
		"gIA7gYAAPAAAAACBgIA7gYAAPIGAADyBgAA8AAAAAAAAAACBgIA7gYAAPMHAQDwAAAAAgYAAPIGA"
		"gDuBgAA8gYCAOwAAAADBwEA8wcBAPMHAQDwAAAAAAAAAAIGAgDvBwEA8gYAAPMHAQDzBwEA8wcBA"
		"PIGAADwAAAAAgYCAO4GAADyBgIA7gYAAPAAAAAAAAAAAgYAAPAAAAACJiIg9goEBP4KBAT+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD/z8vI+gYAAPYGAgDuBgIA7gYAAPIGAADyB"
		"gAA8gYAAPIGAADyBgAA8gYCAOwAAAADBwEA8gYAAPIGAgDuBgAA8gYCAO4GAgDzBwEA8wcBAPIGA"
		"ADyBgIA7gYAAPIGAgDuBgIA8wcBAPIGAADyBgAA8gYAAPMHAQDzBwEA8gYAAPIGAgDyBgIA8gYCA"
		"PMHAQDyBgIA8gYCAPMHAwDzBwEA8wcBAPMHAQDzBwEA8gYCAO4GAADwAAAAAgYAAPIGAgDsAAAAA"
		"gYAAPMHAQDyBgAA8gYCAOwAAAACBgAA8wcBAPMHAQDyBgAA8gYAAPIGAADyBgAA8gYCAO4GAADyB"
		"gAA8wcBAPAAAAAAAAAAAgYCAO4GAADzBwEA8gYAAPMHAQDyBgAA8wcBAPMHAQDyBgAA8gYCAO4GA"
		"gDuBgIA7AAAAAIGAADyBgIA7gYAAPMHAQDyDgoI+gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+BgAA/g4KCPoGAADyBgAA8gYAAPIGAgDuBgIA8gYCAPIGAgDyBgAA8"
		"gYCAO4GAgDuBgAA8wcBAPIGAgDuBgIA7gYAAPIGAgDzBwEA8gYCAPMHAQDyBgAA8gYCAO4GAgDuB"
		"gIA7gYAAPIGAADzBwEA8wcBAPMHAQDzBwEA8gYAAPMHAQDzBwEA8wcBAPIGAADyBgIA8gYCAPKGg"
		"oDyBgIA8wcBAPMHAQDzBwEA8gYAAPIGAgDuBgIA7gYCAO8HAQDyBgIA8gYCAO4GAgDuBgIA7gYAA"
		"PIGAgDuBgAA8wcBAPIGAgDyBgAA8gYCAO4GAADyBgAA8gYAAPIGAgDvBwEA8gYCAPAAAAAAAAAAA"
		"gYCAO4GAADzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDwAAAAAgYCAOwAAAACBgIA7AAAAAAAAAACB"
		"gIA8gYCAPJGQkD3j4uI+goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+DggI/goEBP4KB"
		"AT+BgAA/8/LyPpGQED2BgAA8oaCgPIGAgDyBgIA8wcBAPIGAADyBgIA8gYCAPIGAADzBwEA8gYAA"
		"PIGAgDyBgIA7wcBAPIGAgDyBgIA8gYCAPIGAADyBgAA8gYAAPIGAADzBwEA8wcBAPMHAQDyBgIA8"
		"gYCAPMHAQDzBwEA8gYCAPIGAgDyhoKA8gYCAPKGgoDzBwMA8gYCAPIGAADyBgIA8gYCAPMHAQDzB"
		"wEA8gYAAPMHAQDyBgAA8gYAAPMHAQDyhoKA8wcBAPIGAgDuBgAA8gYCAO4GAgDsAAAAAgYAAPIGA"
		"gDuBgAA8gYCAO8HAQDzBwEA8gYAAPIGAADzBwEA8gYAAPIGAADyBgIA7gYAAPIGAADzBwEA8gYCA"
		"O4GAADyBgIA8gYCAPMHAQDyBgAA8gYCAO4GAgDuBgAA8gYAAPIGAADzBwEA8gYCAPKWkpD6CgQE/"
		"goEBP4KBAT+BgAA/gYAAP4KBAT+CgQE/goEBP4OCAj+BgAA/goEBP4KBAT+CgQE/goEBP4WEhD7B"
		"wEA8gYCAO8HAQDyBgIA8gYAAPMHAQDzBwMA8wcBAPIGAADyBgIA7gYCAO4GAADzBwEA8gYCAPMHA"
		"QDyBgIA8gYCAPIGAADyBgAA8gYCAO8HAQDzBwEA8wcBAPIGAgDzBwEA8gYCAPMHAQDyBgIA8oaCg"
		"PMHAwDzBwEA8gYCAPIGAADyhoKA8wcBAPIGAgDzBwEA8oaCgPMHAQDzBwEA8wcBAPMHAQDyBgAA8"
		"wcBAPMHAQDzBwEA8oaCgPMHAQDyBgAA8gYAAPAAAAACBgIA7gYAAPIGAgDuBgIA8gYCAPMHAQDzB"
		"wEA8gYAAPMHAQDyBgIA7gYAAPIGAgDvBwEA8wcBAPIGAADzBwEA8gYAAPIGAgDvBwEA8gYCAPKGg"
		"oDyBgAA8wcBAPIGAADzBwEA8gYCAO4GAADzBwEA8kZCQPYKBAT+CgQE/goEBP4KBAT+CgQE/goEB"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP/Py8j6JiIg9wcBAPMHAQDyBgAA8"
		"wcBAPMHAQDzBwEA8gYCAPMHAQDyBgAA8gYCAO4GAgDyBgAA8wcBAPMHAQDzBwEA8gYAAPIGAADyB"
		"gIA7gYAAPIGAADzBwEA8gYCAPIGAADyBgIA7wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYCAPMHA"
		"QDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDyBgIA7gYAAPMHAQDzBwEA8wcBAPKGgoDyBgIA8wcBA"
		"PIGAgDyBgAA8gYAAPIGAgDuBgIA7gYAAPIGAgDuBgIA8gYCAPIGAgDyBgIA8AAAAAIGAADzBwEA8"
		"wcBAPIGAADzBwEA8gYCAO4GAADyBgIA7gYCAO4GAADzBwEA8gYCAOwAAAACBgAA8gYCAPIGAgDzB"
		"wEA8wcBAPMHAQDyBgAA8paSkPoKBAT+BgAA/goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GA"
		"AD+CgQE/goEBP4GAAD+CgQE/gYAAP4GAAD+zsrI+gYCAPIGAADyBgAA8gYAAPMHAQDzBwEA8gYAA"
		"PIGAADyBgAA8gYCAO8HAQDyBgIA8gYCAOwAAAADBwEA8gYCAO4GAgDuBgAA8wcBAPMHAQDyBgIA8"
		"gYCAPMHAQDyBgAA8wcBAPIGAADzBwEA8gYCAPMHAQDyBgIA8oaCgPMHAQDzBwEA8wcBAPIGAADzB"
		"wEA8gYCAPMHAQDyBgIA8gYAAPMHAQDyBgIA8wcBAPIGAADyBgAA8wcBAPMHAQDyBgIA8gYAAPIGA"
		"gDuBgIA7AAAAAIGAADzBwEA8oaCgPIGAgDzBwEA8gYCAO8HAQDyBgIA7gYCAO4GAADzBwEA8gYCA"
		"PIGAADyBgIA8gYCAPMHAQDyBgAA8gYCAPIGAgDzBwEA8gYCAPKGgoDyBgIA8gYCAPMHAQDylpCQ+"
		"gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4SDAz+CgQE/gYAAP4GAAD+BgAA/goEBP4KBAT+C"
		"gQE/gYAAP4GAAD+CgQE/iYgIPoGAgDuBgIA7gYAAPIGAgDyBgAA8gYCAO4GAADyBgAA8AAAAAIGA"
		"ADyBgAA8gYAAPIGAADyBgAA8wcBAPIGAADzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBA"
		"PIGAgDyBgIA8wcBAPIGAADyBgAA8gYAAPIGAgDzBwEA8gYAAPMHAQDyBgAA8wcBAPIGAgDyBgAA8"
		"gYAAPIGAgDyBgIA8gYAAPIGAgDzBwEA8gYAAPMHAQDyhoKA8oaCgPIGAgDyBgAA8gYCAOwAAAADB"
		"wEA8gYCAPIGAgDzBwEA8gYCAO4GAADyBgAA8gYCAO4GAADzBwEA8gYAAPIGAADyBgIA7gYCAPIGA"
		"ADyBgAA8AAAAAIGAADzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDzj4uI+goEBP4GAAD+CgQE/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+CgQE/g4ICP4GAAD+CgQE/gYAAP4GAAD+CgQE/"
		"4+LiPpGQED3BwEA8wcBAPMHAQDyBgAA8gYAAPIGAgDuBgIA7gYCAO4GAADzBwEA8wcBAPIGAADyB"
		"gAA8wcBAPIGAADzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYCAPIGA"
		"ADyBgAA8wcBAPIGAADzBwEA8oaCgPMHAQDzBwEA8wcBAPIGAADyBgAA8wcBAPIGAgDzBwEA8gYAA"
		"PIGAgDvBwEA8gYCAO4GAADyBgAA8gYAAPMHAQDyBgIA7gYAAPIGAADyBgAA8wcBAPMHAQDyBgAA8"
		"gYAAPMHAQDwAAAAAgYAAPMHAQDyBgIA8wcBAPMHAQDyBgIA8wcBAPIGAADyBgAA8gYCAO4GAgDuB"
		"gIA7gYCAO4GAgDvBwEA8gYCAPIWEhD6BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP5OSkj6BgAA8wcBA"
		"PMHAQDyBgAA8gYCAPIGAgDuBgIA7gYCAO4GAgDvBwEA8wcBAPIGAADzBwEA8gYAAPMHAQDyBgAA8"
		"wcBAPMHAQDzBwEA8gYCAO4GAgDyBgIA8gYCAPIGAADyBgIA8oaCgPIGAgDyBgAA8wcBAPIGAgDuB"
		"gAA8gYAAPIGAADyBgAA8oaCgPIGAADyBgAA8gYCAPIGAgDyBgAA8wcBAPIGAADzBwEA8gYCAO4GA"
		"ADyBgIA7gYAAPIGAADyBgAA8gYAAPIGAADyBgAA8gYAAPMHAQDyBgIA7gYAAPIGAgDuBgIA7gYAA"
		"PMHAQDzBwEA8gYAAPIGAgDzBwEA8gYAAPIGAADyBgAA8gYCAO4GAgDuBgAA8AAAAAMHAQDyBgIA8"
		"2djYPYKBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+C"
		"gQE/goEBP4KBAT+BgAA/gYAAP4GAAD+DggI/goEBP4GAAD+FhAQ+wcBAPMHAQDyBgIA8gYCAPIGA"
		"ADyBgIA7wcBAPMHAQDzBwEA8gYCAPMHAQDyBgIA8gYCAPMHAQDyBgAA8gYCAPMHAQDzBwEA8wcBA"
		"PMHAQDyhoKA8gYCAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYAAPIGAADyBgAA8gYAAPIGAgDyBgIA8"
		"gYCAPIGAgDyBgAA8wcBAPMHAQDyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8gYAAPIGAgDsA"
		"AAAAAAAAAAAAAAChoKA8gYCAPIGAADyBgAA8gYAAPIGAgDvBwEA8wcBAPMHAQDyBgAA8gYCAPMHA"
		"QDyBgIA7gYCAO4GAADzBwEA8oaCgPMHAQDwAAAAAgYCAO4GAADyJiIg95eTkPoKBAT+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+DggI/goEBP4KBAT+BgAA/"
		"goEBP4KBAT+CgQE/goEBP4GAAD/j4uI+sbAwPYGAgDyhoKA8oaCgPMHAQDyBgAA8AAAAAIGAADzB"
		"wEA8wcBAPMHAQDzBwEA8gYCAPIGAgDzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8oaCgPIGA"
		"ADzBwEA8gYAAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAgDyBgAA8wcBAPIGAADyBgIA7wcBA"
		"PMHAQDwAAAAAAAAAAAAAAACBgIA7AAAAAAAAAAAAAAAAgYCAOwAAAAAAAAAAAAAAAIGAgDvBwEA8"
		"gYCAPIGAADyBgIA7gYAAPIGAADyBgAA8wcBAPIGAADzBwEA8gYCAPMHAQDzBwEA8gYAAPIGAADzB"
		"wEA8gYAAPMHAQDyBgAA8gYAAPIGAADyjoqI+gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4KB"
		"AT+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+BgAA/gYAAP4KBAT+BgAA/gYAAP4KBAT+CgQE/goEB"
		"P4KBAT+CgQE/s7KyPsHAQDzBwEA8gYCAPMHAQDzBwEA8gYAAPIGAADzBwEA8gYCAPIGAADyBgIA8"
		"wcBAPKGgoDyBgIA8wcBAPIGAgDyhoKA8oaCgPMHAQDyBgAA8gYCAPMHAQDzBwEA8gYAAPMHAQDyB"
		"gAA8gYAAPIGAgDyBgIA8wcBAPIGAADyBgIA7gYAAPMHAQDyBgAA8wcBAPMHAQDwAAAAAAAAAAAAA"
		"AACBgIA7wcBAPMHAQDyBgIA7gYAAPIGAgDuBgIA7AAAAAAAAAACBgIA7gYAAPAAAAACBgAA8wcBA"
		"PIGAADyBgAA8gYCAO4GAADzBwEA8wcBAPMHAQDyBgIA7gYAAPIGAADyBgAA8wcBAPIGAgDuBgIA8"
		"wcBAPIWEhD6CgQE/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+DggI/goEBP4KBAT+C"
		"gQE/goEBP4GAAD+CgQE/gYAAP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP+Xk"
		"ZD6BgAA8gYCAO4GAgDyBgIA8gYCAO8HAQDyBgIA8gYAAPMHAQDyBgIA7gYCAPIGAgDyBgIA8wcBA"
		"PMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyhoKA8gYAAPMHAQDzBwEA8gYAAPMHAQDzBwEA8"
		"wcBAPIGAgDyBgAA8wcBAPIGAADwAAAAAgYAAPIGAgDuBgIA7AAAAAAAAAACBgIA7gYAAPIGAADzB"
		"wEA8gYCAOwAAAAAAAAAAgYCAO4GAgDsAAAAAgYCAO4GAgDsAAAAAwcBAPMHAQDyBgAA8gYAAPIGA"
		"ADzBwEA8gYCAPIGAgDyBgIA7gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8jYwMPoKBAT+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4OCAj+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/"
		"goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4OCAj+CgQE/gYAAP4GAAD+BgAA+gYCAO4GAgDyB"
		"gIA8gYAAPIGAADyBgIA8gYCAO4GAADyhoKA8wcBAPIGAgDyBgIA8gYCAO4GAgDyBgIA8gYCAPIGA"
		"gDyBgAA8oaCgPIGAgDzBwEA8gYCAPKGgoDzBwEA8gYCAPIGAgDyBgIA8wcBAPMHAQDyBgIA7wcBA"
		"PIGAgDuBgAA8gYCAO4GAADyBgIA7gYCAO4GAADyBgAA8gYAAPMHAQDyBgIA8wcBAPIGAgDsAAAAA"
		"gYCAO4GAgDuBgAA8gYCAO4GAADzBwEA8gYCAO4GAgDuBgIA7gYCAO4GAgDuBgAA8wcBAPIGAADzB"
		"wEA8gYAAPIGAgDsAAAAAwcBAPIGAADyJiIg98/LyPoKBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/gYAAP4KBAT+CgQE/gYAA"
		"P4GAAD+CgQE/gYAAP4KBAT+BgAA/gYAAP4GAAD/x8PA+kZCQPYGAADyBgIA8gYAAPIGAADyBgAA8"
		"gYAAPIGAgDuBgAA8gYCAPIGAgDyBgAA8gYAAPIGAgDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzB"
		"wEA8gYAAPIGAADzBwEA8gYCAO8HAQDzBwEA8gYAAPIGAgDzBwEA8wcBAPKGgoDzBwEA8gYAAPIGA"
		"ADwAAAAAgYCAO4GAADyBgAA8wcBAPMHAQDyBgAA8gYAAPIGAgDuBgAA8gYAAPIGAgDuBgIA7gYCA"
		"O4GAADzBwEA8gYAAPIGAADyBgIA7gYAAPIGAgDvBwEA8wcBAPMHAQDwAAAAAgYCAO4GAgDuBgIA7"
		"gYCAO4GAgD3h4OA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+C"
		"gQE/goEBP4KBAT+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/goEBP4GA"
		"AD+CgQE/gYAAP4GAAD+BgAA/4+LiPomIiD3BwEA8gYCAO4GAgDuBgIA8wcBAPIGAADyBgAA8gYCA"
		"O8HAQDyBgIA7gYAAPMHAQDyBgIA8oaCgPMHAQDzBwEA8gYCAPIGAADyBgIA8wcBAPMHAQDzBwEA8"
		"gYCAPMHAQDyBgAA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgAA8wcBAPIGAADyBgIA7gYCAO4GAADyB"
		"gIA8gYAAPMHAQDyBgAA8gYCAPIGAADyBgAA8gYCAO4GAgDuBgIA7gYAAPIGAgDzBwEA8wcBAPIGA"
		"ADwAAAAAgYCAO8HAQDyBgIA7wcBAPIGAgDsAAAAAgYCAO4GAADzBwEA8kZCQPePi4j6BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+B"
		"gAA/goEBP+Pi4j6BgIA9gYCAO4GAgDuhoKA8wcBAPIGAADyBgAA8gYAAPMHAQDyBgAA8gYAAPMHA"
		"QDyhoKA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPIGAADyBgAA8wcBAPIGAgDyhoKA8wcBA"
		"PMHAQDzBwEA8wcBAPIGAADyBgAA8wcBAPIGAADyBgAA8gYAAPMHAQDyhoKA8wcBAPMHAQDzBwEA8"
		"wcBAPIGAgDuBgIA7gYCAO4GAADyBgAA8gYAAPIGAADyBgAA8wcBAPMHAQDwAAAAAgYAAPIGAADyB"
		"gAA8gYCAO4GAADyBgIA7gYAAPIGAADyxsDA94+LiPoKBAT+BgAA/goEBP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+DggI/gYAAP4GAAD+CgQE/gYAA"
		"P4GAAD+CgQE/gYAAP4KBAT+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT/T0tI+"
		"gYAAPcHAQDyBgAA8gYCAPIGAgDsAAAAAgYCAO8HAQDyBgAA8gYAAPIGAADyBgIA8wcBAPMHAQDzB"
		"wEA8gYAAPMHAQDzBwEA8wcBAPIGAADyBgAA8wcBAPIGAgDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGA"
		"ADyBgAA8wcBAPIGAgDuBgIA7gYAAPMHAQDyBgAA8gYCAPMHAQDyBgAA8gYCAO8HAQDwAAAAAgYCA"
		"O4GAADyBgAA8gYAAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYCAOwAAAACBgAA8gYCAO4GAgDuBgAA8"
		"gYAAPJmYmD3V1NQ+gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+C"
		"gQE/goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/gYAAP4GA"
		"AD+CgQE/g4ICP4GAAD+BgAA/gYAAP4GAAD+CgQE/g4ICP4KBAT+CgQE/4+LiPqGgoD2BgIA8wcDA"
		"PIGAADyBgAA8gYCAO4GAADzBwEA8gYCAPMHAQDyBgIA7gYCAPIGAgDyBgAA8gYAAPIGAADzBwEA8"
		"wcBAPIGAgDzBwEA8gYCAPIGAgDyhoKA8oaCgPMHAQDzBwEA8gYCAPIGAgDuBgAA8wcBAPMHAQDyB"
		"gAA8gYCAO8HAQDzBwEA8gYAAPIGAgDuBgIA7wcBAPIGAADyBgAA8gYAAPMHAQDyBgAA8gYCAO8HA"
		"QDzBwEA8gYCAO8HAQDyBgIA7wcBAPIGAADzBwEA8gYAAPIGAADyBgIA7kZCQPeXk5D6CgQE/goEB"
		"P4GAAD+BgAA/goEBP4GAAD+BgAA/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/"
		"goEBP4GAAD+CgQE/gYAAP4KBAT+CgQE/goEBP4OCAj+DggI/gYAAP4KBAT+CgQE/goEBP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4OCAj+BgAA/gYAAP+Pi4j6JiIg9gYAAPAAAAAAAAAAAwcBAPIGA"
		"ADzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA8gYAAPIGAADyBgAA8wcBAPIGAgDyBgIA8gYCA"
		"PMHAQDyBgIA8gYCAPIGAgDzBwEA8wcBAPIGAADzBwEA8gYCAPMHAQDwAAAAAgYAAPMHAQDzBwEA8"
		"gYCAO4GAgDuBgIA7wcBAPIGAADyBgIA7gYAAPIGAgDzBwEA8gYAAPIGAgDuBgIA7gYCAO4GAADzB"
		"wEA8gYCAPIGAADyBgAA8gYAAPIGAADyRkJA95eTkPoKBAT+CgQE/g4ICP4GAAD+BgAA/goEBP4GA"
		"AD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEB"
		"P4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/"
		"goEBP4KBAT+BgAA/gYAAP4GAAD/j4uI+gYCAPYGAgDuBgIA7gYCAO4GAADzBwEA8oaCgPMHAQDzB"
		"wEA8wcBAPKGgoDyBgAA8gYAAPMHAQDyBgAA8wcBAPIGAgDyBgAA8wcBAPIGAgDyBgIA8wcBAPIGA"
		"gDyBgIA8gYCAPIGAADyBgAA8gYCAPIGAgDyBgIA7AAAAAMHAQDzBwEA8gYCAO4GAgDuBgIA7gYCA"
		"PMHAQDyBgAA8wcBAPMHAQDyBgIA8wcBAPIGAgDuBgIA7AAAAAIGAADyBgAA8oaCgPMHAQDyBgIA7"
		"gYCAO4mICD7z8vI+gYAAP4KBAT+CgQE/g4ICP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/goEBP4KB"
		"AT+CgQE/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEB"
		"P4GAAD+BgAA/8fDwPoWEBD6BgIA7gYCAO4GAgDuBgAA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgIA7"
		"gYAAPIGAADzBwEA8gYCAPIGAgDyBgAA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8gYAAPIGAADzB"
		"wEA8wcDAPKGgoDwAAAAAgYAAPMHAQDzBwEA8gYCAO4GAgDuBgAA8gYAAPIGAADyBgAA8wcBAPMHA"
		"QDzBwEA8wcBAPIGAgDuBgIA7gYCAO4GAADyBgIA8wcBAPMHAQDyBgIA75eRkPoKBAT+CgQE/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+C"
		"gQE/goEBP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GA"
		"AD+BgIA+gYAAPIGAADzBwEA8wcBAPIGAgDuBgIA8wcBAPIGAADyBgAA8gYCAPIGAgDyBgIA8gYCA"
		"PIGAgDyBgIA8gYCAPIGAgDyBgIA8wcBAPMHAQDyBgAA8wcBAPMHAwDyhoKA8gYCAPMHAQDwAAAAA"
		"gYCAO4GAgDyBgIA8gYAAPIGAADyBgIA7gYCAPIGAADyBgIA7wcBAPMHAQDyBgAA8wcBAPIGAADyB"
		"gIA7gYCAO4GAADyBgAA8wcBAPIGAAD2xsLA+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/goEBP4KBAT+DggI/goEBP4GAAD+BgAA/"
		"gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/o6KiPpGQkD3B"
		"wEA8wcBAPMHAQDzBwEA8gYCAPIGAgDyBgIA8gYCAPIGAgDyBgIA8gYCAPIGAgDzBwEA8wcBAPIGA"
		"gDyhoKA8wcBAPMHAQDzBwEA8gYCAPIGAgDyhoKA8gYAAPMHAQDyBgIA7gYCAO4GAgDuBgAA8gYAA"
		"PMHAQDyBgIA7gYCAO8HAQDyBgAA8gYAAPMHAQDzBwEA8wcBAPMHAQDyBgAA8AAAAAIGAADzBwEA8"
		"iYgIPuHg4D6BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4KB"
		"AT+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP+Pi4j7R0NA9wcBAPIGAgDuBgIA8"
		"gYAAPIGAADyBgIA8wcBAPMHAQDzBwEA8gYAAPIGAADzBwEA8wcBAPIGAgDyBgIA8wcBAPMHAQDzB"
		"wEA8gYCAPIGAADyBgAA8gYCAO4GAADyBgIA7gYCAOwAAAACBgAA8wcBAPMHAQDyBgIA7gYAAPIGA"
		"gDyhoKA8gYCAPMHAQDzBwEA8gYCAPIGAADyBgAA8gYAAPIGAAD2TkpI+gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/"
		"gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/goEBP4KBAT+DggI/gYAAP4GAAD+BgAA/goEBP4KBAT+C"
		"gQE/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/hYSEPsHAQDyBgAA8gYCAO4GAADyBgIA8wcBA"
		"PMHAQDzBwEA8gYAAPIGAADyBgIA8wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgIA7"
		"gYCAPKGgoDzBwEA8gYCAOwAAAADBwEA8gYAAPMHAQDyBgAA8wcBAPMHAQDyBgIA8wcBAPIGAgDzB"
		"wEA8gYCAPIGAgDzBwEA8iYgIPuPi4j6BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4KB"
		"AT+CgQE/gYAAP4KBAT+CgQE/gYAAP4GAAD+CgQE/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP+Xk5D6pqCg+gYAAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDyBgIA8wcBAPIGAgDzBwEA8wcBAPIGAgDyBgAA8gYAAPIGAADyBgAA8wcBAPIGAgDyBgAA8gYAA"
		"PIGAgDuBgAA8gYCAO8HAQDzBwEA8wcBAPMHAQDyhoKA8gYAAPIGAADyBgIA8wcBAPImIiD2zsrI+"
		"gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+B"
		"gAA/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/g4ICP4KB"
		"AT+DggI/g4ICP4KBAT+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAA"
		"P4KBAT+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/"
		"gYAAP4KBAT+CgQE/paSkPomIiD3BwEA8wcBAPIGAgDzBwEA8gYCAPIGAgDzBwEA8wcBAPIGAgDzB"
		"wEA8wcBAPMHAQDyBgAA8wcBAPIGAADzBwEA8wcDAPIGAgDzBwEA8gYCAPIGAgDuBgAA8gYAAPMHA"
		"QDyBgAA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA9g4KCPvPy8j6CgQE/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/goEBP4KBAT+BgAA/goEBP4GAAD+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+CgQE/"
		"goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4OCAj+CgQE/goEB"
		"P4KBAT+lpKQ+kZCQPYGAADyhoKA8gYCAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8"
		"wcBAPIGAgDyBgIA8wcBAPIGAgDzBwEA8gYCAPIGAADyBgAA8gYCAPMHAQDyBgAA8wcBAPIGAADyB"
		"gAA8kZAQPYWEhD7z8vI+gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KB"
		"AT+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4KBAT+DggI/goEBP4KBAT+CgQE/4+LiPoWE"
		"hD6ZmJg9gYCAPKGgoDzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAgDyhoKA8wcBA"
		"PMHAQDyBgIA7AAAAAIGAADzBwEA8gYAAPIGAADyBgIA7gYAAPIGAgD2FhIQ+8/LyPoGAAD+BgAA/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4KBAT+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+BgAA/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4KBAT+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/o6KiPomIiD2B"
		"gAA8gYCAPIGAADyBgIA8gYAAPIGAADyBgAA8gYAAPIGAADzBwEA8gYAAPIGAADyBgIA7wcBAPMHA"
		"QDyBgIA8gYAAPIGAADyFhAQ+w8LCPoGAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+C"
		"gQE/goEBP4SDAz+BgAA/g4ICP4KBAT+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEB"
		"P4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+xsLA+paQkPsHAQDzBwEA8"
		"gYCAO4GAgDyBgAA8wcBAPMHAQDzBwEA8gYAAPMHAQDyBgAA8gYAAPIGAgDuBgAA+kZCQPvPy8j6B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GA"
		"AD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4KBAT+CgQE/goEBP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+C"
		"gQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+BgAA/goEBP4GAAD+CgQE/goEBP4KB"
		"AT+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/g4ICP+Pi4j6joqI+kZAQPoGAgDyBgIA8wcBA"
		"PMHAQDyBgIA8wcBAPKGgID3FxEQ+o6KiPvPy8j6BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4OCAj+CgQE/goEBP4KBAT+B"
		"gAA/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/g4ICP4OC"
		"Aj+CgQE/gYAAP4GAAD+CgQE/g4ICP4OCAj+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+C"
		"gQE/gYAAP4GAAD+CgQE/goEBP4KBAT+DggI/goEBP+Xk5D6lpKQ+rawsPpGQkD2BgAA86ehoPoKB"
		"AT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/g4ICP4KBAT+CgQE/goEB"
		"P4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+DggI/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/"
		"goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+B"
		"gAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GA"
		"AD+CgQE/goEBP4KBAT+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEB"
		"P4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/"
		"goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/goEBP4GAAD/p6Gg+4+LiPoKBAT+CgQE/goEBP4OCAj+C"
		"gQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4OCAj+CgQE/goEBP4GAAD+CgQE/g4ICP4GA"
		"AD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4KBAT+DggI/gYAA"
		"P4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/"
		"gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+B"
		"gAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4KBAT+DggI/goEBP4KBAT+BgAA/gYAAP4GA"
		"AD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/goEB"
		"P4KBAT+BgAA/gYAAP4KBAT/z8vI+8/LyPoKBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/"
		"goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+B"
		"gAA/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4KB"
		"AT+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/"
		"gYAAP4KBAT+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+C"
		"gQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KB"
		"AT/l5OQ+6ehoPoGAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/gYAA"
		"P4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/g4ICP4KBAT+CgQE/goEBP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GA"
		"AD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/g4IC"
		"P4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT/t7Gw+wcBAPKmoqD2h"
		"oCA+paSkPuPi4j6BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4GA"
		"AD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+C"
		"gQE/goEBP4KBAT+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4KB"
		"AT+CgQE/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEB"
		"P4GAAD+CgQE/goEBP/X09D6joqI+xcREPpGQED2BgAA8wcBAPIGAgDzBwEA8wcBAPIGAADyFhAQ+"
		"oaCgPuPi4j6BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+CgQE/gYAAP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+BgAA/goEBP4KBAT+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAA"
		"P4KBAT+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/"
		"gYAAP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+B"
		"gAA/goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/gYAAP4KBAT+CgQE/goEBP/Py8j6VlJQ+iYgIPoGA"
		"gDzBwEA8gYAAPIGAgDuBgAA8wcBAPIGAgDyBgAA8gYCAO4GAADyBgIA7gYAAPIGAADylpCQ+tbS0"
		"PoGAAD+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/"
		"gYAAP4KBAT+CgQE/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+C"
		"gQE/goEBP4GAAD+DggI/gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4GAAD+BgAA/goEBP4KBAT/DwsI+iYgIPoGAgDyhoKA8wcBAPMHAQDyBgIA7gYAAPIGAADyB"
		"gIA7gYCAPIGAgDzBwEA8wcBAPIGAADyBgAA8gYAAPIGAADyBgIA8wcBAPIGAgD2joqI+gYAAP4GA"
		"AD+BgAA/g4ICP4OCAj+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+CgQE/gYAAP4GAAD+CgQE/gYAA"
		"P4KBAT+BgAA/gYAAP4KBAT+CgQE/gYAAP4OCAj+CgQE/goEBP4KBAT+CgQE/g4ICP4OCAj+CgQE/"
		"gYAAP4GAAD+BgAA/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+B"
		"gAA/gYAAP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+CgQE/goEBP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/goEBP4GAAD/z8vI+hYSE"
		"PpmYmD3BwEA8wcBAPMHAwDzBwMA8wcBAPMHAQDyBgAA8gYCAPIGAADyBgAA8gYCAPMHAQDzBwEA8"
		"wcBAPKGgoDyBgIA8gYCAPMHAQDyBgIA8gYCAPIGAADyBgAA8kZCQPYWEhD7j4uI+goEBP4KBAT+C"
		"gQE/gYAAP4KBAT+DggI/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KB"
		"AT+CgQE/gYAAP4GAAD+CgQE/gYAAP4KBAT+CgQE/g4ICP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4GAAD+CgQE/"
		"goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+B"
		"gAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/8/LyPoOCgj6RkBA9gYAAPIGAADyBgAA8gYCAPIGA"
		"gDyhoKA8wcBAPIGAADyBgIA8wcBAPIGAgDuBgAA8gYCAPIGAADzBwEA8gYCAPKGgoDyBgIA8gYAA"
		"PMHAQDyhoKA8oaCgPIGAADzBwEA8gYAAPIGAgDyBgIA9o6KiPoKBAT+BgAA/gYAAP4KBAT+DggI/"
		"g4ICP4KBAT+CgQE/goEBP4OCAj+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+C"
		"gQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KB"
		"AT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4GAAD+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/"
		"goEBP/X09D6FhIQ+oaAgPYGAgDyBgAA8gYAAPIGAADyBgAA8gYAAPIGAgDyhoKA8gYCAPMHAQDyB"
		"gIA8wcBAPIGAADyBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDyBgAA8gYCAPIGAgDzBwEA8oaCgPIGA"
		"gDyBgAA8gYAAPIGAADwAAAAAgYAAPImIiD2npqY+gYAAP4KBAT+DggI/goEBP4GAAD+CgQE/gYAA"
		"P4GAAD+CgQE/gYAAP4OCAj+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/"
		"gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+C"
		"gQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KB"
		"AT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/s7KyPpGQkD2BgIA8wcBA"
		"PMHAQDyBgAA8gYCAPIGAgDyBgIA8wcBAPIGAgDyBgIA8gYCAPMHAQDzBwEA8gYAAPMHAQDyBgIA8"
		"wcBAPMHAQDyBgAA8wcBAPIGAADyBgAA8wcBAPIGAgDzBwEA8gYCAPIGAgDuBgIA7gYCAO4GAADwA"
		"AAAAgYCAO8HAQDzBwEA8oaAgPuHg4D6BgAA/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4KB"
		"AT+DggI/goEBP4KBAT+DggI/goEBP4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4GAAD+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/"
		"goEBP4GAAD+BgAA/gYAAP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+C"
		"gQE/gYAAP4KBAT+CgQE/goEBP+Xk5D6JiAg+gYCAPMHAQDzBwEA8wcBAPIGAgDzBwEA8wcBAPIGA"
		"ADzBwEA8wcBAPIGAgDyBgIA8wcBAPMHAQDyBgIA7gYAAPMHAQDzBwEA8wcBAPIGAADzBwEA8gYAA"
		"PMHAQDzBwEA8wcBAPIGAgDzBwEA8gYCAPKGgoDyBgIA8wcBAPIGAADyBgIA7gYAAPIGAADyBgAA8"
		"AAAAAIGAADyDgoI+gYAAP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+C"
		"gQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/goEBP4GA"
		"AD+BgAA/goEBP4KBAT+BgAA/gYAAP4KBAT+BgAA/gYAAP4KBAT+DggI/goEBP4GAAD+BgAA/gYAA"
		"P4GAAD+BgAA/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/"
		"lZSUPrGwMD3BwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPKGgoDzB"
		"wEA8gYCAO8HAQDyBgIA7gYAAPIGAgDuBgAA8gYCAPIGAgDzBwEA8gYCAPMHAQDyBgIA8wcBAPMHA"
		"QDzBwEA8wcBAPMHAQDyBgIA8wcBAPIGAgDwAAAAAgYAAPAAAAACBgAA8gYCAO4GAgDsAAAAAwcDA"
		"PePi4j6CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4GAAD+CgQE/"
		"goEBP4KBAT+BgAA/goEBP4GAAD+CgQE/gYAAP4GAAD+CgQE/goEBP4GAAD+CgQE/g4ICP4KBAT+B"
		"gAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GA"
		"AD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP+Pi4j6JiAg+wcBAPMHAQDzBwEA8wcBA"
		"PMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYCAPKGgoDyBgIA8wcBAPMHAQDyBgIA7"
		"gYCAO4GAgDuBgIA7wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYCAPIGAADyB"
		"gAA8wcBAPIGAADwAAAAAgYCAO8HAQDyBgAA8AAAAAAAAAADBwEA8gYCAO4mIiD2joqI+goEBP4GA"
		"AD+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4GAAD+CgQE/gYAAP4GAAD+BgAA/goEBP4GAAD+DggI/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/"
		"goEBP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+C"
		"gQE/goEBP4KBAT+CgQE/s7KyPqGgID3BwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDzBwEA8wcBAPMHAQDzBwEA8wcBAPKGgoDyBgIA8wcBAPMHAQDyBgAA8wcBAPIGAADyBgAA8wcBA"
		"PIGAgDyBgIA8gYCAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDyBgIA7"
		"gYCAO4GAgDuBgIA7gYAAPIGAgDuBgIA8gYAAPIGAgDuBgIA8hYSEPoKBAT+CgQE/goEBP4KBAT+D"
		"ggI/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4OCAj+CgQE/goEBP4GA"
		"AD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+BgAA/goEB"
		"P4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT/l5GQ+"
		"wcBAPMHAQDzBwEA8wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDzB"
		"wEA8wcBAPIGAgDyhoKA8wcBAPIGAADyBgIA7gYAAPIGAgDuBgAA8wcBAPMHAQDzBwEA8gYAAPMHA"
		"QDzBwEA8oaCgPIGAgDzBwEA8gYCAPIGAgDzBwEA8gYAAPMHAQDyBgAA8gYAAPIGAgDzBwEA8gYAA"
		"PMHAQDyBgIA7wcBAPIGAgDuBgAA8gYAAPImICD7z8vI+goEBP4GAAD+CgQE/goEBP4OCAj+CgQE/"
		"gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/goEBP4GAAD+B"
		"gAA/gYAAP4KBAT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/goEBP4GA"
		"AD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/8fDwPomICD7BwEA8gYAAPMHAQDzBwEA8wcBA"
		"PIGAgDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDzBwEA8wcBAPKGgoDyBgIA8"
		"gYCAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgAA8wcBAPMHAQDzBwEA8gYCAPIGAgDzB"
		"wEA8oaCgPIGAgDyBgIA8gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYAAPIGAADyBgIA7wcBAPIGA"
		"ADzBwEA8gYCAPMHAQDyZmJg94+LiPoGAAD+CgQE/goEBP4GAAD+CgQE/gYAAP4KBAT+CgQE/gYAA"
		"P4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/"
		"goEBP4GAAD+BgAA/goEBP4KBAT+BgAA/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+C"
		"gQE/gYAAP4GAAD/j4uI+iYiIPcHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDzBwEA8oaCgPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgIA8oaCgPMHAQDzBwEA8wcBA"
		"PIGAgDzBwEA8gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgIA8gYCAPMHAQDyBgIA8"
		"wcBAPMHAQDzBwEA8gYCAO8HAQDzBwEA8wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYCAPIGAgDyB"
		"gIA8kZCQPePi4j6CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/goEBP4KB"
		"AT+BgAA/goEBP4KBAT+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/gYAA"
		"P4KBAT+BgAA/gYAAP4KBAT+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP+Pi4j6RkJA9"
		"gYAAPIGAADzBwEA8gYAAPIGAADzBwEA8wcBAPMHAQDyBgIA8oaCgPMHAQDyBgIA8oaCgPIGAgDyB"
		"gIA8oaCgPMHAQDzBwEA8wcBAPIGAgDyBgIA8wcBAPIGAADyBgIA8gYAAPIGAADyBgIA7gYCAPIGA"
		"gDyBgAA8wcBAPMHAQDyBgIA8wcBAPIGAgDyhoKA8oaCgPMHAQDyBgIA8wcBAPIGAADzBwEA8gYAA"
		"PIGAgDuBgIA8gYCAPIGAgDyBgIA8wcBAPIGAgDyBgIA8gYAAPMHAQDzBwEA8wcBAPJGQkD3j4uI+"
		"goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/g4ICP4GAAD+BgAA/gYAAP4KBAT+CgQE/goEBP4OCAj+C"
		"gQE/goEBP4OCAj+CgQE/g4ICP4OCAj+CgQE/goEBP4GAAD+CgQE/gYAAP4GAAD+BgAA/goEBP4KB"
		"AT+CgQE/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/09LSPpGQkD3BwEA8wcBAPMHAQDyBgIA8wcBA"
		"PMHAQDzBwEA8wcBAPMHAQDyBgIA8gYCAPMHAQDyBgIA8gYCAPIGAgDyBgIA8gYCAPMHAQDzBwEA8"
		"gYAAPIGAgDyBgIA8gYAAPIGAADyBgAA8gYAAPIGAADyBgAA8gYCAPIGAgDzBwEA8wcBAPMHAQDyB"
		"gIA8wcBAPIGAgDyBgIA8oaCgPMHAQDyBgIA8wcBAPIGAgDuBgAA8gYAAPMHAQDyBgIA8gYCAPMHA"
		"QDzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDyBgIA8gYCAPIGAADyhoCA909LSPoKBAT+CgQE/goEB"
		"P4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4OCAj+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+C"
		"gQE/gYAAP4KBAT/j4uI+sbAwPYGAgDyBgIA7wcBAPMHAQDyBgAA8gYAAPMHAQDwAAAAAgYAAPIGA"
		"gDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgIA8wcBAPIGAADzBwEA8wcBAPIGAgDyhoKA8gYAA"
		"PMHAQDyBgAA8gYAAPIGAADyBgAA8gYCAPIGAgDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8"
		"gYCAPMHAQDyBgIA8gYAAPIGAgDyBgAA8gYAAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyB"
		"gIA8wcBAPIGAgDuBgAA8wcBAPMHAQDzBwEA8mZiYPeXk5D6CgQE/goEBP4KBAT+CgQE/goEBP4KB"
		"AT+BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/gYAA"
		"P4GAAD+CgQE/gYAAP4KBAT+DggI/g4ICP4GAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP+Pi4j6RkJA9"
		"gYCAPIGAADzBwEA8gYCAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGAADzBwEA8gYCAPMHAQDyB"
		"gAA8gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYCAPIGAgDyBgIA8wcBAPIGAADzBwEA8gYAAPIGA"
		"ADyBgAA8wcBAPIGAgDzBwEA8wcBAPIGAADzBwEA8gYAAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYAA"
		"PIGAADyBgAA8gYCAO8HAQDzBwEA8wcBAPMHAQDyBgAA8gYAAPMHAQDzBwEA8wcBAPIGAgDyBgIA7"
		"wcBAPMHAQDzBwEA8wcBAPJGQkD3j4uI+g4ICP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4KBAT+C"
		"gQE/goEBP4GAAD+BgAA/gYAAP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4KBAT+BgAA/goEBP4KB"
		"AT+DggI/goEBP4KBAT+CgQE/gYAAP4GAAD+DggI/4+LiPpGQkD2BgAA8gYAAPIGAgDzBwEA8gYCA"
		"PMHAQDzBwEA8wcBAPIGAgDyBgAA8wcBAPIGAADyhoKA8gYCAPMHAQDyBgIA8wcBAPIGAADzBwEA8"
		"wcBAPMHAQDzBwEA8wcBAPIGAgDyBgIA8wcBAPMHAQDyBgIA8wcBAPIGAADyBgAA8wcBAPIGAgDyB"
		"gIA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgAA8gYCAPIGAADyBgIA7gYCAO4GAgDuBgAA8gYAAPIGA"
		"ADzBwEA8gYAAPMHAQDzBwEA8wcBAPMHAQDzBwEA8oaCgPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYCA"
		"PMHAQDyJiIg98/LyPoKBAT+CgQE/goEBP4KBAT+DggI/goEBP4KBAT+CgQE/goEBP4GAAD+BgAA/"
		"gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4OCAj+C"
		"gQE/goEBP4KBAT/19PQ+kZCQPYGAgDzBwEA8wcBAPMHAQDyBgAA8wcBAPIGAADyBgIA8wcBAPIGA"
		"gDzBwEA8gYCAPMHAQDyBgIA8gYAAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPKGgoDyhoKA8oaCg"
		"PMHAQDyBgIA8wcBAPIGAgDzBwEA8wcBAPIGAADyBgAA8wcBAPIGAgDzBwEA8gYCAPIGAADyBgIA8"
		"wcBAPIGAADyBgAA8gYAAPMHAQDyBgAA8gYAAPIGAADyBgIA7gYCAO4GAADzBwEA8wcBAPMHAQDzB"
		"wEA8gYCAPIGAgDyhoKA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYAAPMHAQDyBgIA7hYQEPoKB"
		"AT+BgAA/gYAAP4GAAD+CgQE/goEBP4GAAD+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/goEB"
		"P4KBAT+BgAA/gYAAP4GAAD+CgQE/goEBP4KBAT+DggI/goEBP4KBAT+CgQE/gYAAP4KBAT+NjAw+"
		"oaCgPKGgoDzBwEA8wcBAPMHAQDzBwEA8gYCAO4GAgDyBgAA8wcBAPKGgoDyBgAA8wcBAPIGAgDyB"
		"gIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPMHA"
		"QDzBwEA8wcBAPIGAADyBgAA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYAA"
		"PMHAQDyBgIA7gYAAPIGAgDuBgAA8gYAAPIGAADzBwEA8gYAAPMHAQDzBwEA8wcBAPIGAgDyhoKA8"
		"oaCgPMHAQDyBgIA8gYCAPIGAgDyBgAA8wcBAPIGAADyBgAA8gYAAPOXkZD6BgAA/gYAAP4GAAD+C"
		"gQE/gYAAP4GAAD+CgQE/goEBP4GAAD+BgAA/gYAAP4KBAT+BgAA/goEBP4KBAT+CgQE/goEBP4KB"
		"AT+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/goEBP4OCgj6BgIA8oaCgPIGAgDzBwEA8wcBA"
		"PMHAQDzBwEA8gYCAPIGAADyBgAA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8"
		"gYCAPMHAQDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8gYCAPIGAADyB"
		"gAA8gYAAPKGgoDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAwDyBgAA8wcBAPIGAADzBwEA8gYAAPIGA"
		"ADyBgAA8gYAAPMHAQDyBgAA8wcBAPMHAQDyhoKA8oaCgPKGgoDyBgIA8gYCAPIGAADzBwEA8wcBA"
		"PMHAQDyBgAA8AAAAAIGAADyBgAA8gYCAO8HAQDyzsrI+gYAAP4GAAD+BgAA/gYAAP4GAAD+BgAA/"
		"goEBP4KBAT+BgAA/goEBP4GAAD+BgAA/gYAAP4KBAT+CgQE/g4ICP4KBAT+CgQE/goEBP4KBAT+B"
		"gAA/gYAAP4GAAD+BgAA/oaCgPoGAgDuBgAA8gYCAO4GAgDzBwEA8wcBAPMHAQDyBgAA8gYCAO4GA"
		"ADyBgAA8gYCAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYCAPMHAQDzBwEA8gYCAPMHAQDyBgAA8gYCA"
		"PIGAADyBgAA8gYCAPIGAgDyBgIA8wcBAPIGAADzBwEA8wcBAPIGAgDuhoKA8wcBAPMHAQDzBwEA8"
		"wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8gYCAPIGAADyBgAA8gYAAPMHAQDwAAAAAwcBAPMHAQDyB"
		"gAA8gYCAPIGAgDyhoKA8oaCgPKGgoDyBgIA8wcBAPIGAgDyBgIA8wcBAPMHAQDyBgIA8gYAAPMHA"
		"QDyBgIA7gYAAPMHAQDyBgAA94eDgPoGAAD+BgAA/gYAAP4GAAD+CgQE/g4ICP4KBAT+CgQE/goEB"
		"P4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4GAAD+BgAA/gYAAP4GAAD/j4uI+"
		"kZCQPYGAADyBgIA7wcBAPMHAQDyBgIA8oaCgPMHAQDyBgIA7wcBAPIGAgDzBwEA8gYAAPIGAgDzB"
		"wEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYAAPMHAQDyBgIA8wcBAPIGAgDyBgAA8wcBAPMHA"
		"QDyBgIA8wcBAPIGAgDvBwEA8gYAAPIGAADzBwEA8gYCAO4GAADyBgIA7gYCAPMHAQDyBgIA8gYAA"
		"PMHAQDzBwEA8wcBAPMHAQDyBgIA7wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDyBgIA8"
		"gYCAPIGAADzBwEA8gYCAPIGAgDzBwEA8wcBAPMHAQDyBgAA8gYCAPIGAgDzBwEA8wcBAPMHAQDyB"
		"gAA8gYAAPoGAAD+BgAA/gYAAP4KBAT+CgQE/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4KB"
		"AT+BgAA/goEBP4KBAT+BgAA/goEBP4GAAD+BgAA/goEBP4KBAT/R0NA9gYCAPIGAgDzBwEA8gYCA"
		"PIGAgDzBwEA8wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYAAPIGAgDyhoKA8gYCAPIGAADzBwEA8"
		"gYAAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgAA8gYCAPMHAQDyBgAA8gYAAPMHAQDyB"
		"gAA8wcBAPIGAgDyBgAA8gYCAO4GAgDuBgAA8gYCAO4GAADzBwEA8wcBAPIGAgDzBwEA8wcBAPIGA"
		"ADyBgAA8gYAAPIGAADyBgAA8gYAAPIGAgDzBwEA8wcBAPMHAQDyBgAA8gYAAPIGAgDyBgIA8wcBA"
		"PMHAQDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYAAPJOSkj6BgAA/"
		"gYAAP4KBAT+DggI/gYAAP4KBAT+CgQE/goEBP4GAAD+BgAA/goEBP4KBAT+CgQE/goEBP4GAAD+B"
		"gAA/gYAAP4GAAD+BgAA/goEBP4WEhD6BgAA8gYCAPIGAgDzBwEA8gYAAPMHAQDyBgIA8wcBAPIGA"
		"gDyBgIA8gYCAPIGAgDyBgIA8gYAAPIGAgDyBgIA7gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8wcBA"
		"PMHAQDzBwEA8wcBAPIGAgDyBgIA8gYAAPIGAADzBwEA8wcBAPIGAADzBwEA8gYAAPIGAgDuBgAA8"
		"gYAAPMHAQDyBgIA7wcBAPIGAADzBwEA8wcBAPMHAQDyBgAA8wcBAPIGAADyBgIA8gYAAPMHAQDyB"
		"gAA8gYAAPIGAgDyBgAA8wcBAPMHAQDyBgIA8gYCAPMHAQDyBgAA8oaCgPIGAADyBgIA7wcBAPMHA"
		"QDzBwEA8gYAAPMHAQDzBwEA8wcBAPMHAQDyBgIA8gYCAO5GQED3j4uI+gYAAP4OCAj+CgQE/goEB"
		"P4OCAj+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+BgAA/gYAAP4GAAD+CgQE/"
		"5eTkPoGAgDzBwEA8oaCgPIGAgDzBwEA8wcBAPMHAQDyhoKA8gYCAPIGAADzBwEA8oaCgPMHAQDzB"
		"wEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYAAPIGAgDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDyBgAA8gYAAPIGAADzBwEA8wcBAPIGAgDyhoKA8wcBAPMHAQDyBgIA8gYAAPIGAgDyBgAA8oaCg"
		"PIGAADzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDyBgIA8wcBAPIGAgDzBwEA8gYAAPIGAgDzBwEA8"
		"wcBAPMHAQDyBgIA8gYCAPIGAgDzBwEA8gYCAPIGAgDyBgAA8gYAAPIGAADyBgIA8wcBAPMHAQDzB"
		"wEA8wcBAPIGAgDzBwEA8gYAAPIGAADyJiAg+goEBP4KBAT+DggI/goEBP4OCAj+CgQE/g4ICP4GA"
		"AD+CgQE/goEBP4KBAT+CgQE/goEBP4KBAT+CgQE/gYAAP4KBAT+CgQE/qagoPoGAADyBgAA8gYAA"
		"PIGAADyBgIA8gYCAPIGAADyBgIA8gYCAO8HAQDyBgAA8wcBAPIGAADyBgAA8gYCAPIGAADyBgIA7"
		"gYAAPIGAADyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYAAPMHAQDyBgAA8wcBAPIGAADyB"
		"gAA8gYAAPMHAQDyBgIA8wcBAPIGAgDyBgIA7wcBAPKGgoDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDyBgIA8gYCAPIGAgDvBwEA8gYAAPIGAgDvBwEA8gYCAPKGgoDyBgIA8gYCAPMHAQDzBwEA8gYCA"
		"PMHAQDzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDyBgIA8gYAAPIGAADzBwEA8wcBAPAAAAACBgAA8"
		"gYCAO4GAADyBgIA7s7KyPoGAAD+BgAA/gYAAP4GAAD+CgQE/g4ICP4GAAD+CgQE/goEBP4KBAT+C"
		"gQE/goEBP4KBAT+CgQE/gYAAP4KBAT+lpKQ+gYCAPMHAQDzBwEA8wcBAPKGgoDyBgIA8gYCAPMHA"
		"QDzBwEA8gYCAPMHAQDzBwEA8gYCAPMHAQDzBwEA8gYAAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBA"
		"PIGAgDyBgIA8gYCAPIGAADyBgAA8gYAAPIGAgDvBwEA8gYCAPMHAQDzBwEA8gYCAPIGAgDzBwEA8"
		"gYAAPIGAADyBgAA8wcBAPKGgoDzBwEA8wcBAPIGAADyBgAA8wcBAPIGAgDyBgIA8gYCAPIGAADyB"
		"gAA8gYCAO4GAADyBgIA8wcBAPIGAgDyhoKA8gYCAPIGAADzBwEA8gYCAPMHAQDyBgIA8wcBAPIGA"
		"ADzBwEA8oaCgPIGAgDyBgIA8gYAAPIGAADzBwEA8wcBAPIGAgDzBwEA8gYAAPIGAADyBgAA8iYiI"
		"PfHw8D6BgAA/gYAAP4GAAD+BgAA/goEBP4KBAT+CgQE/g4ICP4OCAj+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4KBAT+JiIg9wcBAPMHAQDzBwEA8gYCAO4GAgDyBgAA8wcBAPMHAQDyhoKA8wcBAPMHAQDyB"
		"gAA8gYCAPMHAQDyhoKA8wcBAPIGAADzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDyBgIA8gYCAPIGA"
		"gDyBgAA8gYCAO4GAgDvBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYAA"
		"PIGAgDyBgIA8gYCAPIGAADzBwEA8gYCAPIGAgDzBwEA8oaCgPMHAQDzBwEA8gYCAPIGAgDzBwEA8"
		"gYAAPMHAQDyBgAA8gYAAPIGAgDyBgAA8wcBAPIGAADyBgAA8gYCAO8HAQDzBwEA8wcBAPIGAADzB"
		"wEA8wcBAPMHAQDyBgIA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIOCgj6BgAA/gYAAP4GA"
		"AD+BgAA/gYAAP4KBAT+DggI/goEBP4KBAT+CgQE/goEBP4OCAj+CgQE/goEBP6WkpD6BgIA8gYCA"
		"PIGAgDzBwEA8wcBAPKGgoDzBwEA8wcBAPIGAADzBwEA8gYAAPMHAQDyBgAA8wcBAPIGAgDuBgAA8"
		"gYAAPIGAADzBwEA8wcBAPIGAADzBwEA8oaCgPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzB"
		"wEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPIGAgDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDzBwEA8gYCAPMHAQDyBgIA8gYCAPIGAADyBgIA8wcBAPIGAgDyBgAA8gYCAPIGAADyBgAA8AAAA"
		"AIGAADyBgAA8wcBAPIGAADyBgIA8gYAAPIGAADzBwEA8wcBAPIGAADyBgIA8wcBAPMHAQDzBwEA8"
		"wcBAPKGgoDyBgIA8wcBAPIGAgDyBgIA8gYCAPJGQED3z8vI+gYAAP4GAAD+DggI/goEBP4GAAD+C"
		"gQE/g4ICP4KBAT+CgQE/goEBP4KBAT+CgQE/4+LiPpmYmD3BwEA8wcBAPKGgoDyBgIA8gYCAPMHA"
		"QDzBwEA8wcBAPIGAADzBwEA8wcBAPIGAgDyBgAA8gYAAPKGgoDzBwEA8gYCAPKGgoDzBwEA8wcBA"
		"PMHAQDzBwEA8wcBAPIGAADzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHAQDzBwEA8"
		"wcBAPMHAQDyBgAA8wcBAPIGAgDyBgIA8wcBAPMHAQDzBwEA8gYCAPIGAADzBwEA8wcBAPIGAgDzB"
		"wEA8gYAAPMHAQDzBwEA8gYAAPMHAQDyBgIA7gYAAPIGAADyBgAA8AAAAAMHAQDzBwEA8wcBAPAAA"
		"AACBgAA8gYAAPIGAADyBgAA8gYAAPIGAgDuBgAA8gYAAPMHAQDzBwEA8gYCAPIGAgDyBgIA8gYCA"
		"PIGAgDzBwEA8wcBAPIGAADyDgoI+goEBP4KBAT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/"
		"goEBP4KBAT+CgQE/g4KCPsHAQDyhoKA8oaCgPIGAgDzBwEA8gYCAPMHAQDyBgAA8gYAAPMHAQDzB"
		"wEA8gYAAPKGgoDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHA"
		"QDyBgIA8gYCAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAgDyBgIA8wcBAPMHAQDyBgIA8wcBA"
		"PIGAgDuBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHAQDzBwEA8oaCgPIGAADyBgAA8"
		"gYAAPIGAgDuBgIA7gYCAO8HAQDyBgIA7gYCAO4GAADyBgAA8wcBAPMHAQDyBgIA7wcBAPMHAQDyB"
		"gIA8gYAAPIGAgDuBgAA8gYAAPMHAQDzBwEA8gYCAPIGAgDyBgIA8oaCgPIGAgDzBwEA8gYCAPIGA"
		"ADyhoCA98/LyPoKBAT+CgQE/goEBP4GAAD+CgQE/goEBP4KBAT+BgAA/gYAAP4KBAT+CgQE/mZiY"
		"PYGAADyBgIA8oaCgPIGAgDzBwEA8wcBAPMHAQDzBwEA8gYAAPIGAADzBwEA8gYCAPIGAgDzBwEA8"
		"oaCgPIGAgDvBwEA8gYCAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgIA8gYCAPIGAgDzB"
		"wEA8gYCAPIGAADzBwEA8wcBAPMHAQDyBgIA8gYAAPMHAQDzBwEA8gYCAPIGAADyBgAA8wcBAPMHA"
		"QDzBwEA8gYAAPIGAgDzBwEA8wcBAPMHAQDyBgIA8gYAAPIGAADyBgIA7gYAAPIGAADyBgAA8gYAA"
		"PIGAADyBgIA8gYAAPIGAgDuBgIA7wcBAPMHAQDyBgAA8gYAAPMHAQDzBwEA8gYCAPIGAADyBgAA8"
		"gYAAPIGAADzBwEA8wcBAPIGAgDyBgIA8gYCAPIGAgDzBwEA8wcBAPIGAADyBgAA8hYSEPoKBAT+B"
		"gAA/goEBP4GAAD+CgQE/goEBP4GAAD+CgQE/gYAAP4KBAT+lpKQ+gYCAPIGAgDzBwEA8gYCAPMHA"
		"QDzBwEA8gYCAPIGAgDyBgIA8wcBAPIGAADzBwEA8gYAAPIGAgDuBgIA7wcBAPIGAgDuBgAA8gYCA"
		"PIGAADyBgAA8gYCAPMHAQDyBgIA8wcBAPIGAgDyBgIA8wcBAPIGAgDyBgIA8wcBAPMHAQDzBwEA8"
		"wcBAPMHAQDyhoKA8wcBAPMHAQDzBwEA8wcBAPIGAgDuBgAA8gYCAPMHAQDzBwEA8wcBAPIGAADzB"
		"wEA8wcBAPIGAgDzBwMA8wcBAPMHAQDzBwEA8AAAAAIGAADzBwEA8AAAAAIGAADyBgIA8gYAAPIGA"
		"gDvBwEA8oaCgPIGAgDyBgAA8wcBAPIGAADwAAAAAgYAAPMHAQDyBgAA8gYCAO4GAADzBwEA8wcBA"
		"PIGAgDyBgIA8gYCAPIGAgDzBwEA8gYAAPIGAADyBgAA8kZCQPYKBAT+CgQE/goEBP4KBAT+CgQE/"
		"goEBP4KBAT+CgQE/goEBP4KBAT+RkJA9wcBAPIGAgDyBgIA8gYCAPKGgoDzBwEA8wcBAPKGgoDzB"
		"wEA8wcBAPMHAQDyBgIA7gYCAO4GAADyBgIA8gYAAPIGAADyBgIA7gYCAO4GAgDyBgAA8gYCAPIGA"
		"gDzBwEA8wcBAPMHAQDyBgIA8wcBAPIGAgDyBgIA8gYCAPMHAQDyBgIA8wcBAPMHAQDyBgIA8wcBA"
		"PMHAQDzBwEA8wcBAPIGAADyBgAA8wcBAPMHAQDyBgAA8wcBAPMHAQDyBgAA8gYCAO8HAQDzBwEA8"
		"wcBAPMHAQDzBwEA8gYCAO4GAgDvBwEA8gYAAPIGAADyBgAA8gYCAO4GAADzBwEA8wcBAPIGAADyB"
		"gAA8wcBAPMHAQDyBgIA7gYCAO8HAQDyBgAA8AAAAAIGAADyBgAA8wcBAPMHAQDzBwEA8wcBAPMHA"
		"QDzBwEA8wcBAPMHAQDzBwEA8gYAAPMPCwj6BgAA/gYAAP4KBAT+CgQE/goEBP4KBAT+BgAA/goEB"
		"P7Oysj6BgIA8wcBAPIGAgDyBgIA8gYCAPIGAgDyBgIA8oaCgPKGgoDyBgIA8gYCAPIGAgDvBwEA8"
		"gYAAPMHAQDyBgAA8gYAAPIGAgDvBwEA8gYAAPMHAQDzBwEA8gYAAPMHAQDzBwEA8gYCAPIGAgDyB"
		"gAA8wcBAPIGAgDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPIGA"
		"gDuBgAA8wcBAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAgDyBgIA8gYAA"
		"PIGAADyBgAA8wcBAPIGAgDuBgAA8gYAAPMHAQDyBgIA8wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA8"
		"gYAAPIGAgDuBgAA8gYCAO4GAADzBwEA8gYAAPIGAADzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAgDyB"
		"gIA8gYAAPIGAAD6BgAA/gYAAP4KBAT+CgQE/goEBP4OCAj+BgAA/goEBP6WkJD7BwEA8wcBAPIGA"
		"ADyBgAA8gYCAPKGgoDyBgIA7wcBAPMHAQDzBwEA8wcBAPKGgoDyBgIA8gYCAO4GAADwAAAAAgYCA"
		"O4GAADzBwEA8gYAAPIGAgDwAAAAAgYAAPMHAQDyBgIA8wcBAPMHAQDzBwEA8gYAAPMHAQDzBwEA8"
		"wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDyB"
		"gAA8gYAAPIGAADyBgIA8oaCgPMHAQDyBgAA8gYAAPMHAQDzBwEA8gYAAPIGAgDuBgAA8wcBAPIGA"
		"gDuBgAA8gYAAPIGAADyBgAA8gYAAPIGAADzBwEA8gYAAPMHAQDyBgAA8gYAAPIGAADyBgAA8gYAA"
		"PIGAADyBgIA8wcBAPIGAgDyBgIA8gYCAPIGAgDzBwEA8wcBAPIGAADyhoKA8wcBAPMHAQDzz8vI+"
		"goEBP4KBAT+CgQE/goEBP4OCAj+CgQE/4+LiPoGAgDzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDzB"
		"wEA8wcBAPMHAQDzBwEA8wcBAPKGgoDyBgIA8gYCAO4GAgDuBgAA8gYAAPIGAgDuBgAA8gYCAO4GA"
		"ADzBwEA8wcBAPIGAADyBgAA8wcBAPIGAADzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAADyBgAA8wcBA"
		"PIGAADyBgAA8gYAAPIGAgDvBwEA8gYAAPIGAADyBgAA8wcBAPKGgoDyBgAA8gYCAPIGAgDyhoKA8"
		"oaCgPIGAgDzBwEA8gYCAPMHAQDzBwEA8gYAAPIGAADyBgAA8wcBAPAAAAADBwEA8wcBAPIGAgDuB"
		"gIA7gYAAPIGAgDyBgAA8wcBAPIGAADyBgIA7gYAAPMHAQDyBgIA7gYCAO4GAgDvBwEA8wcBAPMHA"
		"QDyBgIA8gYCAPIGAgDzBwEA8oaCgPMHAQDzBwEA8wcBAPIGAADyVlJQ+goEBP4KBAT+CgQE/goEB"
		"P4KBAT+CgQE/paSkPoGAgDyBgIA8wcBAPMHAQDyhoKA8gYCAPIGAgDzBwEA8wcBAPMHAQDzBwEA8"
		"gYAAPIGAgDyBgIA8gYAAPIGAADyBgAA8gYCAPIGAADzBwEA8gYAAPIGAgDvBwEA8oaCgPIGAgDyB"
		"gIA8gYAAPIGAADzBwEA8gYCAPIGAgDzBwEA8gYCAPIGAgDyBgAA8oaCgPIGAADzBwEA8wcBAPMHA"
		"QDyBgAA8gYAAPIGAADyBgAA8gYCAPIGAgDzBwEA8gYCAPMHAQDzBwEA8gYCAPIGAgDyBgIA8gYCA"
		"PMHAQDzBwEA8gYAAPMHAQDyBgIA7gYAAPIGAgDuBgAA8gYAAPIGAADzBwEA8gYAAPIGAADyBgIA8"
		"wcBAPIGAADwAAAAAgYAAPIGAADyBgAA8gYAAPIGAADyBgIA7wcBAPIGAgDyBgIA8gYAAPIGAADzB"
		"wEA8gYAAPMHAQDzBwEA8wcBAPMHAQDyJiAg+goEBP4KBAT+CgQE/gYAAP4KBAT+BgAA/iYgIPoGA"
		"ADyBgIA8gYAAPMHAQDyBgIA8wcBAPIGAgDyBgIA8gYCAPIGAgDzBwEA8gYAAPIGAADzBwEA8gYCA"
		"O4GAgDuBgIA7gYAAPIGAADyBgAA8gYAAPMHAQDyBgAA8wcBAPIGAADyBgAA8gYCAO4GAADyhoKA8"
		"gYCAO4GAgDsAAAAAgYCAO4GAgDuBgIA7gYAAPAAAAACBgAA8gYCAO4GAgDvBwEA8gYCAPIGAADyB"
		"gAA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgIA8gYCAPIGAgDyBgIA8gYCAPMHAQDyhoKA8wcBAPIGA"
		"gDyBgAA8wcBAPMHAQDyBgAA8AAAAAIGAgDuBgIA8wcBAPAAAAACBgAA8wcBAPIGAgDuBgAA8gYAA"
		"PIGAADzBwEA8gYAAPIGAADyBgAA8gYCAO4GAADyBgIA8gYCAO8HAQDyBgAA8wcBAPMHAQDyBgAA8"
		"wcBAPMHAQDyBgAA88/LyPoGAAD+CgQE/goEBP4KBAT/j4uI+gYAAPMHAQDzBwEA8wcBAPMHAQDyB"
		"gIA8gYCAPKGgoDyBgIA8gYCAPIGAgDyBgIA7gYCAO8HAQDyBgAA8gYAAPIGAADyBgIA7gYAAPIGA"
		"gDsAAAAAAAAAAMHAQDyBgAA8gYAAPIGAgDuBgAA8gYAAPIGAADyBgAA8wcBAPIGAgDuBgIA7gYAA"
		"PMHAQDwAAAAAgYAAPIGAgDuBgIA7gYAAPMHAQDzBwEA8gYAAPMHAQDyBgAA8gYCAPIGAgDzBwEA8"
		"wcBAPMHAQDzBwEA8wcBAPMHAQDyBgAA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHAQDyB"
		"gAA8gYAAPKGgoDyBgAA8wcBAPIGAADyBgAA8gYAAPMHAQDyBgAA8gYAAPIGAADyBgIA7wcBAPIGA"
		"gDzBwEA8gYAAPAAAAACBgAA8gYAAPMHAQDzBwEA8wcBAPMHAQDyBgIA8wcBAPMHAQDyBgAA8paSk"
		"PoKBAT+CgQE/goEBP4KBAT+hoKA+gYAAPMHAQDyhoKA8gYCAPIGAADyBgIA8gYCAPMHAQDyBgIA8"
		"oaCgPIGAADyBgAA8gYAAPMHAQDzBwEA8oaCgPIGAgDyBgIA7gYCAPIGAADyBgAA8wcBAPIGAgDyh"
		"oKA8wcBAPIGAADyhoKA8gYAAPIGAADyBgIA7gYAAPIGAADyBgAA8gYAAPIGAADyBgAA8gYAAPIGA"
		"ADzBwEA8gYAAPMHAQDyBgAA8gYAAPIGAADzBwEA8wcBAPIGAgDyBgAA8wcBAPIGAADzBwEA8gYAA"
		"PIGAADzBwEA8gYAAPMHAQDyBgIA8gYAAPIGAgDyBgIA8wcBAPMHAQDwAAAAAgYCAO8HAQDyBgAA8"
		"gYCAOwAAAACBgAA8gYAAPIGAADyBgAA8gYAAPIGAgDvBwEA8wcBAPMHAQDzBwEA8gYAAPIGAADyB"
		"gIA7wcBAPMHAQDzBwEA8gYAAPMHAQDyBgIA8wcBAPMHAQDzBwEA8xcREPoKBAT+CgQE/goEBP4KB"
		"AT+pqCg+wcBAPIGAgDyBgIA8wcBAPIGAADzBwEA8gYCAPIGAgDyBgIA8wcBAPMHAQDyBgIA8gYCA"
		"PIGAADyBgAA8gYCAPMHAQDyBgIA7wcBAPIGAADyBgAA8wcBAPIGAADzBwEA8wcBAPIGAgDuBgAA8"
		"gYCAO4GAADyBgAA8gYAAPIGAgDuBgAA8gYAAPIGAgDuBgAA8gYAAPIGAADyBgAA8gYCAO4GAgDuB"
		"gAA8gYAAPIGAgDuBgAA8gYAAPMHAQDzBwEA8gYAAPIGAADyBgIA8wcBAPMHAQDzBwEA8wcBAPIGA"
		"gDzBwEA8gYCAPMHAQDzBwEA8wcBAPIGAgDzBwEA8wcBAPIGAADzBwEA8gYCAO4GAADzBwEA8gYAA"
		"PIGAADzBwEA8wcBAPMHAQDzBwEA8wcBAPMHAQDzBwEA8gYCAPMHAQDzBwEA8wcBAPMHAQDyBgIA8"
		"gYCAPMHAQDzBwEA8wcBAPIGAADyBgAA8kZAQPYGAAD+CgQE/goEBP4KBAT+JiIg9gYCAPIGAgDyB"
		"gIA8wcBAPIGAgDyBgAA8gYCAPMHAQDyBgAA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgAA8gYAAPIGA"
		"ADyBgIA7gYAAPMHAQDyBgIA7gYCAO4GAgDsAAAAAgYCAO4GAADyBgAA8wcBAPIGAADyBgAA8gYCA"
		"O4GAADzBwEA8wcBAPMHAQDzBwEA8wcBAPIGAADyBgIA7gYAAPIGAgDyBgIA7gYAAPAAAAACBgAA8"
		"wcBAPMHAQDzBwEA8wcBAPIGAADzBwEA8wcBAPIGAgDyBgAA8wcBAPIGAgDzBwEA8wcBAPKGgoDyB"
		"gAA8wcBAPIGAADzBwEA8AAAAAIGAADyBgIA7gYAAPMHAQDyBgAA8gYAAPIGAADyBgAA8wcBAPIGA"
		"ADzBwEA8gYCAPIGAADyBgIA7wcBAPIGAgDyBgIA8gYAAPMHAQDyBgIA8gYCAPMHAQDyBgIA8wcBA"
		"PMHAQDyBgAA8wcBAPOXkZD7h4OA+8/LyPunoaD7BwEA8gYCAPIGAgDzBwEA8gYAAPIGAgDvBwEA8"
		"wcBAPIGAADyBgIA8wcBAPIGAADyBgIA8wcBAPMHAQDyBgAA8gYAAPIGAADyBgAA8gYAAPIGAgDvB"
		"wEA8wcBAPIGAgDuBgAA8gYAAPIGAgDuBgAA8wcBAPIGAgDyBgIA7gYCAO4GAADzBwEA8wcBAPMHA"
		"QDzBwEA8wcBAPIGAgDuBgIA7gYCAPMHAQDyBgIA7gYAAPIGAADyBgAA8"
	# 完整数据在 JS 源文件 src/core/embeddedAlphaMaps.js 中
	)

_init_alpha_data()


if __name__ == "__main__":
	main()

