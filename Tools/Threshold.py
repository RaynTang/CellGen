from PIL import Image
import os

# 定义二值化函数
def binarize(image_path, threshold):
    # 打开图像并转换为灰度图像
    image = Image.open(image_path).convert('L')
    # 对图像进行二值化处理
    image = image.point(lambda x: 0 if x < threshold else 255)
    return image

# 定义输入输出文件夹
input_folder = r'D:\Project\stylegan3-main\test_results\2023.03\4609'
output_folder = r'D:\Project\stylegan3-main\test_results\Threshold\4609'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义二值化阈值（可以根据需要进行调整）
threshold = 240

# 循环遍历输入文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图片文件
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # 获取输入文件的完整路径
        input_path = os.path.join(input_folder, filename)
        # 获取输出文件的完整路径
        output_path = os.path.join(output_folder, filename)
        # 进行二值化处理
        image = binarize(input_path, threshold)
        # 保存二值化后的图像
        image.save(output_path)
