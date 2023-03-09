import cv2
import numpy as np
import random
import os

# 指定文件夹路径
folder_path = "situation1/input/image"
save_path = "situation1/output"

# 循环处理文件夹中的图片
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):

        n_samples = 200
        # 定义两个指定像素点
        x1, y1 = 550, 10
        x2, y2 = 500, 410
        x3, y3 = 10, 900
        x4, y4 = 1020, 350
        x5, y5 = 1020, 1

        # 创建黑色画布
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)

        # generate random sample, two components
        np.random.seed(0)

        # generate spherical data
        shifted_gaussian = np.random.randn(n_samples, 2)

        # generate zero centered stretched Gaussian data
        C = np.array([[-5, 2], [5, 0]])
        stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

        # 读取图片
        img = cv2.imread(os.path.join(folder_path, filename))

        # 生成100个随机点（第一个指定点）
        points1 = []
        for i in range(50):
            distance = random.uniform(0, 50)
            x_new = int(x1 + 1.3*distance * shifted_gaussian[i][0])
            y_new = int(y1 + 1.3*distance * shifted_gaussian[i][1])
            points1.append([x_new, y_new])

        # 生成100个随机点（第二个指定点）
        points2 = []
        for i in range(50):
            distance = random.uniform(0, 50)
            x_new = int(x2 + 0.8*distance * stretched_gaussian[i][0])
            y_new = int(y2 + 0.8*distance * stretched_gaussian[i][1])
            points2.append([x_new, y_new])

        # 生成100个随机点（第三个指定点）
        points3 = []
        for i in range(40):
            distance = random.uniform(0, 50)
            x_new = int(x3 + 0.9*distance * shifted_gaussian[i][0])
            y_new = int(y3 + 0.9*distance * shifted_gaussian[i][1])
            points3.append([x_new, y_new])

        # 生成100个随机点（第4个指定点）
        points4 = []
        for i in range(40):
            distance = random.uniform(0, 50)
            x_new = int(x4 + 0.9*distance * shifted_gaussian[i][0])
            y_new = int(y4 + 0.9*distance * shifted_gaussian[i][1])
            points4.append([x_new, y_new])

        # 生成100个随机点（第5个指定点）
        points5 = []
        for i in range(30):
            distance = random.uniform(0, 50)
            x_new = int(x5 + distance * shifted_gaussian[i][0])
            y_new = int(y5 + distance * shifted_gaussian[i][1])
            points5.append([x_new, y_new])

        # 在图片上绘制随机点，同时分配颜色
        for point in points1:
            distance = np.sqrt((x1 - point[0]) ** 2 + (y1 - point[1]) ** 2)
            color = [0, 200 + int(55 * (1 - distance / 50)), 0]
            cv2.circle(img, (point[0], point[1]), 1, color, -1)
            mask1 = cv2.circle(canvas, (point[0], point[1]), 3, color, -1)

        for point in points2:
            distance = np.sqrt((x2 - point[0]) ** 2 + (y2 - point[1]) ** 2)
            color = [0, 200 + int(55 * (1 - distance / 50)), 0]
            images = cv2.circle(img, (point[0], point[1]), 1, color, -1)
            mask2 = cv2.circle(canvas, (point[0], point[1]), 3, color, -1)

        for point in points3:
            distance = np.sqrt((x3 - point[0]) ** 2 + (y3 - point[1]) ** 2)
            color = [0, 200 + int(55 * (1 - distance / 50)), 0]
            cv2.circle(img, (point[0], point[1]), 1, color, -1)
            mask3 = cv2.circle(canvas, (point[0], point[1]), 3, color, -1)

        for point in points4:
            distance = np.sqrt((x4 - point[0]) ** 2 + (y4 - point[1]) ** 2)
            color = [0, 200 + int(55 * (1 - distance / 50)), 0]
            cv2.circle(img, (point[0], point[1]), 1, color, -1)
            mask4 = cv2.circle(canvas, (point[0], point[1]), 3, color, -1)

        for point in points5:
            distance = np.sqrt((x5 - point[0]) ** 2 + (y5 - point[1]) ** 2)
            color = [0, 200 + int(55 * (1 - distance / 50)), 0]
            cv2.circle(img, (point[0], point[1]), 1, color, -1)
            mask5 = cv2.circle(canvas, (point[0], point[1]), 3, color, -1)
        # 把点额外保存在黑色的画布
        temp = mask1 + mask2 + mask5

        masks = temp + mask3 +mask4

        # 转换为灰度图像
        gray = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY)

        # 设置阈值，大于阈值的像素点为255，否则为0
        threshold_value = 20
        binary = ret, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # 保存图片
        cv2.imwrite(os.path.join(save_path, filename), images)
        cv2.imwrite(os.path.join(save_path, "mask_"+filename), binary)
