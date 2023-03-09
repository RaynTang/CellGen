import cv2
import os
import numpy as np
from tqdm import tqdm

# 定义数据增强函数
def data_augmentation(image, mask):
    # 随机水平翻转
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    # 随机垂直翻转
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    # 随机旋转
    angle = np.random.randint(-10, 10)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    mask = cv2.warpAffine(mask, M, (cols, rows))

    # 随机亮度调整
    alpha = 1 + np.random.uniform(-0.1, 0.1)
    image = cv2.convertScaleAbs(image, alpha=alpha)

    # 随机色彩调整
    beta = np.random.randint(-20, 20)
    image = cv2.add(image, beta)

    return image, mask

# 设置输入和输出文件夹
input_folder = "datasets/spots/train_B/"
mask_folder = "datasets/spots/train_A/"
output_folder = "datasets/spots_aug/train_B/"
output_mask = "datasets/spots_aug/train_A/"
# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in tqdm(os.listdir(input_folder)):
    # 如果文件不是图像文件，则跳过
    if not filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    # 加载图像和掩码
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)
    mask_path = os.path.join(mask_folder, filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 数据增强
    for i in range(10):  # 生成10个增强后的图像
        augmented_image, augmented_mask = data_augmentation(image, mask)
        output_filename = os.path.join(output_folder, "{}_{}.png".format(filename[:-4], i))
        cv2.imwrite(output_filename, augmented_image)
        output_mask_filename = os.path.join(output_mask, "{}_{}.png".format(filename[:-4], i))
        cv2.imwrite(output_mask_filename, augmented_mask)
