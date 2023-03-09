from PIL import Image
import os

folder1 = "combine_data/image/"
folder2 = "combine_data/spot/"
folder3 = "combine_data/output/"
# 获取文件夹1中的图片列表
files1 = os.listdir(folder1)
images1 = [Image.open(folder1 + f) for f in files1]

# 获取文件夹2中的图片列表
files2 = os.listdir(folder2)
images2 = [Image.open(folder2 + f) for f in files2]

for i in range(len(images1)):
    for j in range(len(images2)):
        if files1[i] == files2[j]:
            # 获取两张图片的像素值
            pixels1 = images1[i].load()
            pixels2 = images2[j].load()

            # 获取图片的尺寸
            width, height = images1[i].size

            # 像素叠加
            for x in range(width):
                for y in range(height):
                    r1, g1, b1 = pixels1[x, y]
                    r2, g2, b2 = pixels2[x, y]
                    r = min(r1 + r2, 255)
                    g = min(g1 + g2, 255)
                    b = min(b1 + b2, 255)
                    pixels1[x, y] = (r, g, b)

            # 保存图片
            images1[i].save(folder3 + files1[i])
