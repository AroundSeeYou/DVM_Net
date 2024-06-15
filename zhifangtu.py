import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

root_path = r'/home/knog/data/lzh/lzh/images/WHU_CD/test/A'  # 要修改的图像所在的文件夹路径
save_path = r'/home/knog/data/lzh/lzh/images/WHU_CD/test/A1'
filelist = os.listdir(root_path)  # 遍历文件夹
i = 0
for item in filelist:

    if item.endswith('.png'):
        src = os.path.join(os.path.abspath(root_path), item)  # 原本的名称

        src1 = src.split('/')[10]

        img = cv2.imread(src)
        # 彩色图像均衡化 需要分解通道 对每一个通道均衡化
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)

        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))




        # plt.figure('original')
        # plt.hist(img.ravel(), 256)
        # plt.figure('after')
        # plt.hist(img_e.ravel(), 256)
        # plt.show()
        #
        # cv2.imshow('img', img)
        # cv2.imshow('e_img', img_e)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        src1 = src1.replace( '.png', '.tif')
        print(src1)
        cv2.imwrite(save_path + '/' + src1, result)

print('ending...')







