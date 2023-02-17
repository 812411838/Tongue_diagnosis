from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='./msyhbd.ttc')
img = Image.open("./data/test/199.png")
print("原图大小：",img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)


# plt.subplot(2,2,1),plt.imshow(img),plt.title("原图",fontproperties=font)
# plt.subplot(2,2,2),plt.imshow(data1),plt.title("转换后的图1",fontproperties=font)
# plt.subplot(2,2,3),plt.imshow(data2),plt.title("转换后的图2",fontproperties=font)
# plt.subplot(2,2,4),plt.imshow(data3),plt.title("转换后的图3",fontproperties=font)
# plt.show()

img1 = transforms.RandomHorizontalFlip()(data2)
img2 = transforms.RandomHorizontalFlip()(data2)
img3 = transforms.RandomHorizontalFlip()(data2)

plt.subplot(2,2,1),plt.imshow(data2),plt.title("原图",fontproperties=font)
plt.subplot(2,2,2), plt.imshow(img1), plt.title("变换后的图1",fontproperties=font)
# plt.subplot(2,2,3), plt.imshow(img2), plt.title("变换后的图2",fontproperties=font)
# plt.subplot(2,2,4), plt.imshow(img3), plt.title("变换后的图3",fontproperties=font)
plt.show()