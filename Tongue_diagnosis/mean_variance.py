import numpy as np
import cv2
import os
# img_h, img_w = 32, 32
img_h, img_w = 224, 224   #根据⾃⼰数据集适当调整，影响不⼤
means, stdevs = [], []
img_list = []
imgs_path = './data/tongue'
imgs_path_list = os.listdir(imgs_path)
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)
imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成⼀⾏
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
# BGR --> RGB ， CV读取的需要转换，PIL读取的不⽤转换
means.reverse()
stdevs.reverse()
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
