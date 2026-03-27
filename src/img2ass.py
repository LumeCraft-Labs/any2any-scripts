
r"""
img2ass — 图片/SVG 转 ASS 绘图指令转换器

转换流水线: PNG/JPG/BMP/WEBP --(vtracer 矢量化)--> SVG --(路径解析)--> ASS 绘图指令
也可直接输入 SVG 文件或原始 SVG 路径数据。

SVG 路径转换移植自 MontageSubs/svg-to-ass (MIT License)
https://github.com/MontageSubs/svg-to-ass

用法:
    python img2ass.py logo.png                       # 图片 -> ASS (默认 8x)
    python img2ass.py logo.png -s 16                 # 16x 极致精度
    python img2ass.py logo.png --svg                 # 图片 -> SVG (输出到 stdout)
    python img2ass.py logo.png --save-svg out.svg    # 保存 SVG + 输出 ASS
    python img2ass.py input.svg                      # SVG -> ASS
    python img2ass.py -d "M0,0 L100,0 ..." -s 1      # 原始路径数据
    python img2ass.py photo.jpg --colormode binary   # 强制黑白描摹
    python img2ass.py icon.png -m                    # 每条路径独立输出一行
"""

from __future__ import annotations

import math
import os
import re
import sys
import argparse

# Windows 终端中文输出支持
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from typing import Optional


# ── 数据类型 ─────────────────────────────────────────────────────────────────

@dataclass
class MoveCmd:
    type: str = "M"
    x: float = 0.0
    y: float = 0.0


@dataclass
class LineCmd:
    type: str = "L"
    x: float = 0.0
    y: float = 0.0


@dataclass
class CubicCmd:
    type: str = "C"
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    x: float = 0.0
    y: float = 0.0


DrawCmd = MoveCmd | LineCmd | CubicCmd


# ── 弧线转贝塞尔 ──────────────────────────────────────────────────────────────

def arc_to_bezier(
    px: float, py: float, cx: float, cy: float,
    rx: float, ry: float,
    x_axis_rotation: float,
    large_arc_flag: bool, sweep_flag: bool,
) -> Optional[list[dict]]:
    """将 SVG 弧线转换为一组三次贝塞尔曲线。"""
    if rx == 0 or ry == 0:
        return None

    phi = math.radians(x_axis_rotation)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    pxp = cos_phi * (px - cx) / 2 + sin_phi * (py - cy) / 2
    pyp = -sin_phi * (px - cx) / 2 + cos_phi * (py - cy) / 2

    if pxp == 0 and pyp == 0:
        return []

    rx = abs(rx)
    ry = abs(ry)

    lam = (pxp * pxp) / (rx * rx) + (pyp * pyp) / (ry * ry)
    if lam > 1:
        rx *= math.sqrt(lam)
        ry *= math.sqrt(lam)

    radicant = (rx * rx * ry * ry) - (rx * rx * pyp * pyp) - (ry * ry * pxp * pxp)
    if radicant < 0:
        radicant = 0
    radicant /= (rx * rx * pyp * pyp) + (ry * ry * pxp * pxp)
    radicant = math.sqrt(radicant) * (-1 if large_arc_flag == sweep_flag else 1)

    center_xp = radicant * rx / ry * pyp
    center_yp = radicant * -ry / rx * pxp

    center_x = cos_phi * center_xp - sin_phi * center_yp + (px + cx) / 2
    center_y = sin_phi * center_xp + cos_phi * center_yp + (py + cy) / 2

    ang1 = math.atan2((pyp - center_yp) / ry, (pxp - center_xp) / rx)
    ang2 = math.atan2((-pyp - center_yp) / ry, (-pxp - center_xp) / rx)

    ang_diff = ang2 - ang1
    if not sweep_flag and ang_diff > 0:
        ang_diff -= 2 * math.pi
    elif sweep_flag and ang_diff < 0:
        ang_diff += 2 * math.pi

    segments = max(1, math.ceil(abs(ang_diff) / (math.pi / 2)))
    segment_angle = ang_diff / segments
    k = (4 / 3) * math.tan(segment_angle / 4)

    curves = []
    current_ang = ang1
    for _ in range(segments):
        next_ang = current_ang + segment_angle
        cos1, sin1 = math.cos(current_ang), math.sin(current_ang)
        cos2, sin2 = math.cos(next_ang), math.sin(next_ang)

        p1x = rx * (cos1 - k * sin1)
        p1y = ry * (sin1 + k * cos1)
        p2x = rx * (cos2 + k * sin2)
        p2y = ry * (sin2 - k * cos2)
        p3x = rx * cos2
        p3y = ry * sin2

        curves.append({
            "x1": center_x + cos_phi * p1x - sin_phi * p1y,
            "y1": center_y + sin_phi * p1x + cos_phi * p1y,
            "x2": center_x + cos_phi * p2x - sin_phi * p2y,
            "y2": center_y + sin_phi * p2x + cos_phi * p2y,
            "x":  center_x + cos_phi * p3x - sin_phi * p3y,
            "y":  center_y + sin_phi * p3x + cos_phi * p3y,
        })
        current_ang = next_ang

    return curves


# ── SVG 路径解析器 ────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"([a-zA-Z])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


class SVGParser:
    """解析 SVG path 的 d 属性，输出绝对坐标的 M/L/C 命令序列。"""

    def __init__(self, path_str: str):
        self.tokens = [m.group() for m in _TOKEN_RE.finditer(path_str)]
        self.index = 0
        self.pen_x = 0.0
        self.pen_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.last_control_x = 0.0
        self.last_control_y = 0.0
        self.last_command = ""

    def _next_num(self) -> float:
        if self.index >= len(self.tokens):
            return float("nan")
        val = self.tokens[self.index]
        self.index += 1
        try:
            return float(val)
        except ValueError:
            return float("nan")

    @staticmethod
    def _is_command(t: str) -> bool:
        return len(t) == 1 and t.isalpha()

    def parse(self) -> list[DrawCmd]:
        commands: list[DrawCmd] = []
        self.index = 0
        current_command = ""

        while self.index < len(self.tokens):
            prev_index = self.index
            token = self.tokens[self.index]

            if self._is_command(token):
                current_command = token
                self.index += 1
            else:
                if current_command == "M":
                    current_command = "L"
                if current_command == "m":
                    current_command = "l"

            cmds = self._process_command(current_command)
            if cmds:
                commands.extend(cmds)
            self.last_command = current_command

            if self.index == prev_index:
                self.index += 1

        return commands

    def _process_command(self, cmd: str) -> list[DrawCmd]:
        result: list[DrawCmd] = []
        cx, cy = self.pen_x, self.pen_y
        is_rel = cmd == cmd.lower()

        def req() -> float:
            v = self._next_num()
            if math.isnan(v):
                raise ValueError(f"命令 '{cmd}' 的数据不完整")
            return v

        try:
            if cmd in ("M", "m"):
                x, y = req(), req()
                if is_rel:
                    x += self.pen_x
                    y += self.pen_y
                self.pen_x, self.pen_y = x, y
                self.start_x, self.start_y = x, y
                result.append(MoveCmd(x=x, y=y))
                cx, cy = x, y

            elif cmd in ("L", "l", "H", "h", "V", "v"):
                x, y = self.pen_x, self.pen_y
                lower = cmd.lower()
                if lower == "h":
                    x = req()
                    if is_rel:
                        x += self.pen_x
                elif lower == "v":
                    y = req()
                    if is_rel:
                        y += self.pen_y
                else:
                    x, y = req(), req()
                    if is_rel:
                        x += self.pen_x
                        y += self.pen_y
                self.pen_x, self.pen_y = x, y
                result.append(LineCmd(x=x, y=y))
                cx, cy = x, y

            elif cmd in ("C", "c"):
                x1, y1 = req(), req()
                x2, y2 = req(), req()
                x, y = req(), req()
                if is_rel:
                    x1 += self.pen_x; y1 += self.pen_y
                    x2 += self.pen_x; y2 += self.pen_y
                    x += self.pen_x;  y += self.pen_y
                result.append(CubicCmd(x1=x1, y1=y1, x2=x2, y2=y2, x=x, y=y))
                self.pen_x, self.pen_y = x, y
                cx, cy = x2, y2

            elif cmd in ("S", "s"):
                x2, y2 = req(), req()
                x, y = req(), req()
                if is_rel:
                    x2 += self.pen_x; y2 += self.pen_y
                    x += self.pen_x;  y += self.pen_y
                if self.last_command in ("C", "c", "S", "s"):
                    x1 = 2 * self.pen_x - self.last_control_x
                    y1 = 2 * self.pen_y - self.last_control_y
                else:
                    x1, y1 = self.pen_x, self.pen_y
                result.append(CubicCmd(x1=x1, y1=y1, x2=x2, y2=y2, x=x, y=y))
                self.pen_x, self.pen_y = x, y
                cx, cy = x2, y2

            elif cmd in ("Q", "q"):
                x1, y1 = req(), req()
                x, y = req(), req()
                if is_rel:
                    x1 += self.pen_x; y1 += self.pen_y
                    x += self.pen_x;  y += self.pen_y
                cp1x = self.pen_x + (2 / 3) * (x1 - self.pen_x)
                cp1y = self.pen_y + (2 / 3) * (y1 - self.pen_y)
                cp2x = x + (2 / 3) * (x1 - x)
                cp2y = y + (2 / 3) * (y1 - y)
                result.append(CubicCmd(x1=cp1x, y1=cp1y, x2=cp2x, y2=cp2y, x=x, y=y))
                self.pen_x, self.pen_y = x, y
                cx, cy = x1, y1

            elif cmd in ("T", "t"):
                x, y = req(), req()
                if is_rel:
                    x += self.pen_x
                    y += self.pen_y
                if self.last_command in ("Q", "q", "T", "t"):
                    x1 = 2 * self.pen_x - self.last_control_x
                    y1 = 2 * self.pen_y - self.last_control_y
                else:
                    x1, y1 = self.pen_x, self.pen_y
                cp1x = self.pen_x + (2 / 3) * (x1 - self.pen_x)
                cp1y = self.pen_y + (2 / 3) * (y1 - self.pen_y)
                cp2x = x + (2 / 3) * (x1 - x)
                cp2y = y + (2 / 3) * (y1 - y)
                result.append(CubicCmd(x1=cp1x, y1=cp1y, x2=cp2x, y2=cp2y, x=x, y=y))
                self.pen_x, self.pen_y = x, y
                cx, cy = x1, y1

            elif cmd in ("A", "a"):
                rx, ry = req(), req()
                rot = req()
                laf, sf = req(), req()
                x, y = req(), req()
                if is_rel:
                    x += self.pen_x
                    y += self.pen_y
                curves = arc_to_bezier(
                    self.pen_x, self.pen_y, x, y, rx, ry, rot,
                    laf == 1, sf == 1,
                )
                if curves is None:
                    result.append(LineCmd(x=x, y=y))
                else:
                    for c in curves:
                        result.append(CubicCmd(
                            x1=c["x1"], y1=c["y1"],
                            x2=c["x2"], y2=c["y2"],
                            x=c["x"], y=c["y"],
                        ))
                self.pen_x, self.pen_y = x, y
                cx, cy = x, y

            elif cmd in ("Z", "z"):
                self.pen_x, self.pen_y = self.start_x, self.start_y
                result.append(LineCmd(x=self.start_x, y=self.start_y))
                cx, cy = self.start_x, self.start_y

        except ValueError:
            pass

        self.last_control_x = cx
        self.last_control_y = cy
        return result


# ── SVG 路径提取 ──────────────────────────────────────────────────────────────

_PATH_D_RE = re.compile(r'\bd\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
_BASIC_SHAPE_RE = re.compile(
    r"<(circle|rect|line|polygon|polyline|ellipse)", re.IGNORECASE
)


def extract_path_data(svg_str: str) -> str:
    """从 SVG 标记中提取 path 的 d 属性；若输入不含标签则视为原始路径数据。"""
    if "<" not in svg_str and ">" not in svg_str and "=" not in svg_str:
        return svg_str

    has_basic_shapes = bool(_BASIC_SHAPE_RE.search(svg_str))
    matches = _PATH_D_RE.findall(svg_str)

    if matches:
        if has_basic_shapes:
            print(
                "警告: 基础形状 (circle/rect 等) 已被忽略，仅转换 <path> 元素。",
                file=sys.stderr,
            )
        if len(matches) > 1:
            print(f"已智能合并 {len(matches)} 条路径。", file=sys.stderr)
        return " ".join(matches)

    if has_basic_shapes:
        raise ValueError(
            "检测到 <circle>/<rect>/<line> 等基础形状，本工具仅支持 <path> 路径。\n"
            "解决方法:\n"
            "  - AI 生成: 提示词中要求「将所有图形转换为 <path> 路径输出」\n"
            "  - Illustrator: 全选 → 对象 → 扩展 / 路径 → 轮廓化描边\n"
            "  - Inkscape: 全选 → 路径 → 对象转路径 (Object to Path)\n"
            "  - Figma: 全选 → 右键 → Flatten Selection (拼合所选内容)\n"
            "  - 或直接用 img2ass 输入原始图片，自动矢量化为 <path>"
        )

    raise ValueError("未检测到有效的 SVG 路径数据，请检查输入。")


# ── SVG → ASS 转换 ───────────────────────────────────────────────────────────

SCALE_MAP = {1: r"\p1", 8: r"\p4", 16: r"\p5"}


def svg_to_ass(
    svg_input: str,
    scale: int = 8,
    extra_tags: str = "",
) -> str:
    r"""
    将 SVG 路径数据（或完整 SVG 标记）转换为 ASS 绘图指令字符串。

    参数:
        svg_input: 原始 SVG path d 数据，或完整的 SVG/HTML 标记。
        scale: 精度倍率 -- 1, 8 (\p4), 或 16 (\p5)。
        extra_tags: 附加 ASS 标签，如 "\\pos(960,540)"。

    返回:
        完整的 ASS 绘图字符串，如 "{\\p4}m 0 0 l 800 0 ...{\\p0}"
    """
    path_data = extract_path_data(svg_input)

    # 超大输入保护 (与原版一致，阈值 500KB)
    if len(path_data) > 500_000:
        print(
            f"警告: 输入过大 ({len(path_data) // 1024}KB)，转换可能较慢。",
            file=sys.stderr,
        )

    parser = SVGParser(path_data)
    abs_commands = parser.parse()

    if not abs_commands:
        raise ValueError("无效的路径数据 -- 未解析到任何命令。")

    f = lambda n: round(n * scale)  # noqa: E731

    parts: list[str] = []
    for c in abs_commands:
        if isinstance(c, MoveCmd):
            parts.append(f"m {f(c.x)} {f(c.y)}")
        elif isinstance(c, LineCmd):
            parts.append(f"l {f(c.x)} {f(c.y)}")
        elif isinstance(c, CubicCmd):
            parts.append(
                f"b {f(c.x1)} {f(c.y1)} {f(c.x2)} {f(c.y2)} {f(c.x)} {f(c.y)}"
            )

    p_tag = SCALE_MAP.get(scale, r"\p1")
    head = "{" + extra_tags + p_tag + "}"
    return head + " ".join(parts) + r"{\p0}"


def svg_to_ass_multi(
    svg_input: str,
    scale: int = 8,
    extra_tags: str = "",
) -> list[str]:
    """
    转换含多个 <path> 的 SVG，每条路径返回一个独立的 ASS 绘图字符串。

    若输入为单路径或原始 d 数据，返回只含一个元素的列表。
    """
    if "<" not in svg_input and ">" not in svg_input and "=" not in svg_input:
        return [svg_to_ass(svg_input, scale, extra_tags)]

    matches = _PATH_D_RE.findall(svg_input)
    if not matches:
        raise ValueError("未检测到有效的 SVG 路径数据，请检查输入。")

    results = []
    for d in matches:
        results.append(svg_to_ass(d, scale, extra_tags))
    return results


# ── 图片 → SVG (vtracer) ─────────────────────────────────────────────────────

_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif", ".gif"}


def is_raster_image(path: str) -> bool:
    """根据文件扩展名判断是否为光栅图片。"""
    return os.path.splitext(path)[1].lower() in _RASTER_EXTS


def image_to_svg(
    image_path: str,
    *,
    colormode: str = "color",
    hierarchical: str = "stacked",
    mode: str = "spline",
    filter_speckle: int = 4,
    color_precision: int = 6,
    layer_difference: int = 16,
    corner_threshold: int = 60,
    length_threshold: float = 4.0,
    max_iterations: int = 10,
    splice_threshold: int = 45,
    path_precision: int = 3,
) -> str:
    r"""
    使用 vtracer 将光栅图片矢量化为 SVG 字符串。

    参数:
        image_path: PNG/JPG/BMP/WEBP 文件路径。
        colormode: "color" 彩色（默认）或 "binary" 黑白。
        hierarchical: "stacked" 堆叠（紧凑，默认）或 "cutout" 镂空（带孔洞）。
        mode: "spline" 贝塞尔曲线（默认）或 "polygon" 直线段。
        filter_speckle: 过滤小于 N 像素的噪点斑块。
        color_precision: 颜色有效位数 (1-8)。
        layer_difference: 层间颜色差异阈值。
        corner_threshold: 拐角检测角度（度）。
        length_threshold: 最小线段长度。
        max_iterations: 曲线拟合最大迭代次数。
        splice_threshold: 样条拼接角度阈值（度）。
        path_precision: SVG 路径坐标小数位数。

    返回:
        SVG 标记字符串。

    异常:
        ImportError: 未安装 vtracer 时抛出。
        FileNotFoundError: 图片文件不存在时抛出。
    """
    try:
        import vtracer
    except ImportError:
        raise ImportError(
            "转换光栅图片需要 vtracer。\n"
            "请执行: pip install vtracer"
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    with open(image_path, "rb") as fh:
        img_bytes = fh.read()

    svg_str = vtracer.convert_raw_image_to_svg(
        img_bytes,
        img_format=os.path.splitext(image_path)[1].lstrip(".").lower(),
        colormode=colormode,
        hierarchical=hierarchical,
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        layer_difference=layer_difference,
        corner_threshold=corner_threshold,
        length_threshold=length_threshold,
        max_iterations=max_iterations,
        splice_threshold=splice_threshold,
        path_precision=path_precision,
    )
    return svg_str


def image_to_ass(
    image_path: str,
    *,
    scale: int = 8,
    extra_tags: str = "",
    multi: bool = False,
    # vtracer options
    colormode: str = "color",
    hierarchical: str = "stacked",
    mode: str = "spline",
    filter_speckle: int = 4,
    color_precision: int = 6,
    layer_difference: int = 16,
    corner_threshold: int = 60,
    length_threshold: float = 4.0,
    max_iterations: int = 10,
    splice_threshold: int = 45,
    path_precision: int = 3,
) -> str | list[str]:
    r"""
    一条龙: 光栅图片 -> ASS 绘图指令。

    参数:
        image_path: PNG/JPG/BMP/WEBP 文件路径。
        scale: ASS 精度 -- 1, 8 (\p4), 或 16 (\p5)。
        extra_tags: 附加 ASS 标签，如 "\\pos(960,540)"。
        multi: 为 True 时，每条路径返回独立的 ASS 字符串。
        其余参数: 传递给 image_to_svg() 用于 vtracer 调参。

    返回:
        ASS 绘图字符串；multi=True 时返回字符串列表。
    """
    svg_str = image_to_svg(
        image_path,
        colormode=colormode,
        hierarchical=hierarchical,
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        layer_difference=layer_difference,
        corner_threshold=corner_threshold,
        length_threshold=length_threshold,
        max_iterations=max_iterations,
        splice_threshold=splice_threshold,
        path_precision=path_precision,
    )
    if multi:
        return svg_to_ass_multi(svg_str, scale, extra_tags)
    return svg_to_ass(svg_str, scale, extra_tags)


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="img2ass",
        description=(
            "将图片 (PNG/JPG/BMP/WEBP) 或 SVG 文件转换为 ASS 绘图指令。\n"
            "光栅图片通过 vtracer 自动矢量化。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s logo.png                       # 图片 -> ASS (8x)\n"
            "  %(prog)s logo.png -s 16                 # 图片 -> ASS (16x)\n"
            "  %(prog)s logo.png --svg                 # 图片 -> SVG (stdout)\n"
            "  %(prog)s logo.png --save-svg out.svg    # 保存 SVG + 输出 ASS\n"
            "  %(prog)s logo.png --colormode binary    # 黑白描摹\n"
            "  %(prog)s input.svg                      # SVG -> ASS\n"
            '  %(prog)s -d "M0,0 L100,0 100,100 Z"     # 原始路径数据\n'
            "  %(prog)s icon.png -m                    # 每条路径独立一行\n"
            "  %(prog)s photo.jpg --filter-speckle 8   # 过滤更多噪点\n"
        ),
    )

    # ── 输入源 ────────────────────────────────────────────────────────────────
    ap.add_argument(
        "input",
        nargs="?",
        help="图片或 SVG 文件路径，'-' 表示从 stdin 读取 (SVG/路径数据)",
    )
    ap.add_argument(
        "-d", "--data",
        help='原始 SVG path d 属性，如 "M0,0 L100,0 100,100 Z"',
    )

    # ── ASS 输出选项 ──────────────────────────────────────────────────────────
    ass_grp = ap.add_argument_group("ASS 输出")
    ass_grp.add_argument(
        "-s", "--scale",
        type=int,
        default=8,
        choices=[1, 8, 16],
        help=r"精度: 1 (\p1), 8 (\p4), 16 (\p5)。默认: 8",
    )
    ass_grp.add_argument(
        "-t", "--tags",
        default="",
        help=r'附加 ASS 标签，如 "\pos(960,540)"',
    )
    ass_grp.add_argument(
        "-m", "--multi",
        action="store_true",
        help="每条 <path> 独立输出一行 ASS (而非合并)",
    )

    # ── vtracer 选项 (仅光栅图片) ─────────────────────────────────────────────
    vt_grp = ap.add_argument_group("vtracer 选项 (仅光栅图片)")
    vt_grp.add_argument(
        "--colormode",
        choices=["color", "binary"],
        default="color",
        help="颜色模式: 'color' 彩色 (默认) 或 'binary' 黑白",
    )
    vt_grp.add_argument(
        "--hierarchical",
        choices=["stacked", "cutout"],
        default="stacked",
        help="形状堆叠: 'stacked' 堆叠紧凑 (默认) 或 'cutout' 镂空 (带孔洞)",
    )
    vt_grp.add_argument(
        "--mode",
        choices=["spline", "polygon"],
        default="spline",
        help="曲线模式: 'spline' 贝塞尔 (默认) 或 'polygon' 直线段",
    )
    vt_grp.add_argument(
        "--filter-speckle",
        type=int,
        default=4,
        metavar="N",
        help="过滤小于 N 像素的噪点斑块 (默认: 4)",
    )
    vt_grp.add_argument(
        "--color-precision",
        type=int,
        default=6,
        choices=range(1, 9),
        metavar="N",
        help="颜色精度位数 1-8 (默认: 6)",
    )
    vt_grp.add_argument(
        "--layer-difference",
        type=int,
        default=16,
        metavar="N",
        help="层间颜色差异阈值 (默认: 16)",
    )
    vt_grp.add_argument(
        "--corner-threshold",
        type=int,
        default=60,
        metavar="DEG",
        help="拐角检测角度，单位度 (默认: 60)",
    )
    vt_grp.add_argument(
        "--length-threshold",
        type=float,
        default=4.0,
        metavar="N",
        help="最小线段长度 (默认: 4.0)",
    )
    vt_grp.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        metavar="N",
        help="曲线拟合最大迭代次数 (默认: 10)",
    )
    vt_grp.add_argument(
        "--splice-threshold",
        type=int,
        default=45,
        metavar="DEG",
        help="样条拼接角度阈值，单位度 (默认: 45)",
    )
    vt_grp.add_argument(
        "--path-precision",
        type=int,
        default=3,
        metavar="N",
        help="SVG 坐标小数位数 (默认: 3)",
    )

    # ── SVG 中间产物输出 ──────────────────────────────────────────────────────
    out_grp = ap.add_argument_group("SVG 输出")
    out_grp.add_argument(
        "--svg",
        action="store_true",
        help="将 SVG 输出到 stdout，不转换为 ASS (仅光栅图片输入)",
    )
    out_grp.add_argument(
        "--save-svg",
        metavar="PATH",
        help="将中间 SVG 保存到文件 (光栅输入: 矢量化结果; SVG 输入: 原样保存)",
    )

    args = ap.parse_args()

    # ── 辅助函数: 转换 SVG 为 ASS 并输出 ──────────────────────────────────────
    def output_ass(svg_str: str):
        if args.multi:
            for line in svg_to_ass_multi(svg_str, args.scale, args.tags):
                print(line)
        else:
            print(svg_to_ass(svg_str, args.scale, args.tags))

    def save_svg_file(svg_str: str):
        if args.save_svg:
            with open(args.save_svg, "w", encoding="utf-8") as fh:
                fh.write(svg_str)
            print(f"SVG 已保存至: {args.save_svg}", file=sys.stderr)

    # ── 判断输入来源 ──────────────────────────────────────────────────────────
    if args.data:
        if args.svg:
            print("错误: --svg 需要光栅图片输入，不能与 -d/--data 同时使用",
                  file=sys.stderr)
            sys.exit(1)
        try:
            output_ass(args.data)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.input:
        ap.print_help()
        sys.exit(1)

    # stdin -> 视为 SVG/路径数据
    if args.input == "-":
        if args.svg:
            print("错误: --svg 需要光栅图片文件，不支持 stdin",
                  file=sys.stderr)
            sys.exit(1)
        svg_input = sys.stdin.read()
        try:
            output_ass(svg_input)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 文件输入 -- 根据扩展名检测类型
    if not os.path.isfile(args.input):
        print(f"错误: 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    if is_raster_image(args.input):
        # ── 光栅图片流水线 ────────────────────────────────────────────────────
        try:
            vt_kwargs = dict(
                colormode=args.colormode,
                hierarchical=args.hierarchical,
                mode=args.mode,
                filter_speckle=args.filter_speckle,
                color_precision=args.color_precision,
                layer_difference=args.layer_difference,
                corner_threshold=args.corner_threshold,
                length_threshold=args.length_threshold,
                max_iterations=args.max_iterations,
                splice_threshold=args.splice_threshold,
                path_precision=args.path_precision,
            )
            svg_str = image_to_svg(args.input, **vt_kwargs)
            save_svg_file(svg_str)

            if args.svg:
                # 输出 SVG 到 stdout，跳过 ASS 转换
                print(svg_str)
            else:
                output_ass(svg_str)

        except ImportError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        except (ValueError, FileNotFoundError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # ── SVG 文件 ──────────────────────────────────────────────────────────
        with open(args.input, encoding="utf-8") as fh:
            svg_input = fh.read()

        if args.svg:
            # 已经是 SVG，直接输出
            save_svg_file(svg_input)
            print(svg_input)
            return

        save_svg_file(svg_input)
        try:
            output_ass(svg_input)
        except ValueError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

