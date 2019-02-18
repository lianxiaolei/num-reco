# coding: utf-8

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


def get_new_data(base_path):
    """

    :param base_path:
    :return:
    """
    nums = os.listdir(base_path)
    train_data = []
    train_label = []
    lbl = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    for num in nums:
        calc = 0
        jpgs = os.listdir(os.path.join(base_path, num))
        print('-' * 30, 'now load %s' % num, '-' * 30)
        for jpg in jpgs:
            # calc += 1
            # if calc > 5000:
            #     print('the %s data is more than 5000' % num)
            # break

            fname = os.path.join(base_path, num, jpg)
            pic = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            pic = pic / 255.
            pic = pic.reshape(-1)
            train_data.append(pic)
            # train_label.append(lbl[int(num)])
            train_label.append(int(num))

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    # print(train_data.shape, train_label.shape)
    # print(train_data)
    # print(np.argmax(train_label, axis=1))
    return train_data, train_label


data, label = get_new_data(r'F:\num_ocr')
train_x, test_x, train_y, test_y = train_test_split(data, label)

# 从仍然需要对训练和测试的特征数据进行标准化。
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)

lsvc = LinearSVC(verbose=1)
lsvc.fit(train_x, train_y)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中。
y_predict = lsvc.predict(test_x)

from sklearn.metrics import classification_report

print(classification_report(test_y, y_predict))
joblib.dump(lsvc, "ocr_svm_1.0.1.m")
print('save done')
