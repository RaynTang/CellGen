import os

import cv2
import cv2 as cv
import numpy as np

# 图像填充
def fill_color_demo(src):
    for a, b, c in os.walk(root):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            # print(file_i_path)
            src = cv2.imread(file_i_path)
            img_copy = src.copy()
            h, w, ch = src.shape
            # 声明一个矩形形状，注意高度和宽度都增加 2 个像素
            # np.zeros 返回一个给定形状和类型的用 0 填充的数组
            mask = np.zeros([h+2, w+2], np.uint8)
            # 参数1，待使用泛洪填充的图像
            # 参数2，掩膜，使用掩膜可以规定是在哪个区域使用该算法，如果是对于完整图像都要使用，则掩膜大小为原图行数 + 2，列数 + 2
            # 掩膜，是一个二维的0矩阵，因为只有掩膜上对应为 0 的位置才能泛洪
            # 参数3，泛洪填充的种子点，基于该点的像素判断和它相近颜色的像素点，是否被泛洪处理
            # 参数4，泛洪区域的新颜色（BGR格式）
            # 参数5，种子点像素可以向下的像素值
            # 参数6，种子点像素可以向上的像素值
            # 参数7，泛洪算法的处理模式
            cv.floodFill(img_copy, mask, (500, 1000), (255, 255, 255),
                 (50, 50, 50), (100, 100, 100), cv.FLOODFILL_FIXED_RANGE)
            # cv.imshow("color_demo", img_copy)
            new_mask = 255 - img_copy
            # cv.imshow("color_demo", new_mask)
            new_mask[np.where((new_mask == [255, 255, 255]).all(axis=2))] = [76, 76, 76];
            # cv.imshow("color_demo", new_mask)
            new_img = new_mask + src

            cv2.imwrite(os.path.join(save_path, file_i[:-4] + ".png"), new_img)

if __name__ == "__main__":
    root = './dataset/10.16NEW'
    save_path = './dataset/10.17'
    for a, b, c in os.walk(root):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            print(file_i_path)
            src = cv2.imread(file_i_path)


    fill_color_demo(src)
    cv.waitKey()
    cv.destroyAllWindows()
