"""
转换hqx纹理
"""

from PIL import Image
import sys

IMAGE_FILENAME = "hq2x.png" 
OUTPUT_FILENAME = "hq2x.txt"
TEXTURE_NAME = "HQX2"

def convert_image_to_texture_block():

	try:
		with Image.open(IMAGE_FILENAME) as img:
			img = img.convert("RGBA")
			width, height = img.size
			pixels = list(img.getdata())
			print(f"成功读取图像 '{IMAGE_FILENAME}'")
			print(f"尺寸: {width}x{height}")
			print(f"总像素数: {len(pixels)}")

	except FileNotFoundError:
		print(f"错误: 无法找到文件 '{IMAGE_FILENAME}'。请确保文件与脚本位于同一目录。")
		sys.exit(1)
	except Exception as e:
		print(f"处理图像时发生错误: {e}")
		sys.exit(1)

	output_lines = []
	output_lines.append(f"//!TEXTURE {TEXTURE_NAME}")
	output_lines.append(f"//!SIZE {width} {height}")
	output_lines.append("//!FILTER NEAREST")
	output_lines.append("//!BORDER CLAMP")
	output_lines.append("//!FORMAT rgba8")

	hex_data = ""
	for r, g, b, a in pixels:
		# 将每个通道的 8 位整数转换为 2 位的十六进制字符串
		hex_data += f"{r:02x}{g:02x}{b:02x}{a:02x}"

	line_length = 128 # 每行128个字符 (对应16个像素)
	formatted_hex_data = '\n'.join(
		hex_data[i:i+line_length] for i in range(0, len(hex_data), line_length)
	)
	output_lines.append(formatted_hex_data)

	try:
		with open(OUTPUT_FILENAME, "w") as f:
			f.write("\n".join(output_lines))
		print(f"\n成功！纹理块已保存到 '{OUTPUT_FILENAME}'")
	except Exception as e:
		print(f"写入文件时发生错误: {e}")

if __name__ == "__main__":
	convert_image_to_texture_block()
