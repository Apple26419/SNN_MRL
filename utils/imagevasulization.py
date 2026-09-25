import os
from PIL import Image, ImageOps
import torchvision

# 设置保存图片的目录
output_dir = "cifar_with_border"
os.makedirs(output_dir, exist_ok=True)

# 下载 CIFAR-10 测试集（返回的是 PIL Image）
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# 设置黑色描边的宽度（单位：像素）
border_size = 15
border_color = 'red'

# 需要保存的图片数量 N
N = 5  # 可根据需要修改

# # 遍历数据集中的前 N 张图片
# for idx in range(min(N, len(testset))):
#     image, label = testset[idx]
#     # 为图片添加黑色描边
#     image_with_border = ImageOps.expand(image, border=border_size, fill='black')
#     filename = os.path.join(output_dir, f"cifar_{idx}.png")
#     image_with_border.save(filename)
#
#     # 每处理10张图片打印一次进度
#     if (idx + 1) % 10 == 0:
#         print(f"已保存 {idx + 1} 张图片")
#
# print(f"前 {N} 张图片均已保存到目录：{output_dir}")
#
#
image_path = './cifar_with_border/snake.jpg'
image = Image.open(image_path)
image_with_border = ImageOps.expand(image, border=border_size, fill=border_color)
filename = os.path.join(output_dir, f"snake_red.jpg")
image_with_border.save(filename)
