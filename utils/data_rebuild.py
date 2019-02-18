# coding:utf8

import cv2
import os
import numpy as np


def rebuild(path, target):
    nums = os.listdir(path)
    for i in nums:
        for j in os.listdir(os.path.join(path, i)):

            img = cv2.imread(os.path.join(path, i, j), cv2.IMREAD_GRAYSCALE)

            kernel = np.ones((3, 3), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
            img3 = cv2.GaussianBlur(img, (3, 3), 1)
            img5 = cv2.GaussianBlur(img, (5, 5), 1)

            if not os.path.exists(os.path.join(target, i)):
                os.mkdir(os.path.join(target, i))

            cv2.imwrite(os.path.join(target, i, '3' + j), img3)
            cv2.imwrite(os.path.join(target, i, '5' + j), img5)
        print(i, 'done')
    print('all done')


if __name__ == '__main__':
    if not os.path.exists(r'F:\number_ok1'):
        os.mkdir(r'F:\number_ok1')
    rebuild(r'F:\number_ok', r'F:\number_ok1')
