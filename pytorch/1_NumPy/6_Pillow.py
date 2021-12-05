from PIL import Image
import numpy as np

im = Image.open("../../images/pytorch_2.png")

# print(im.size) # (977, 838)

im_pillow = np.asarray(im)
# print(im_pillow.shape) # (838, 977, 3)

im_pillow_c1 = im_pillow[:, :, 0]
im_pillow_c2 = im_pillow[:, :, 1]
im_pillow_c3 = im_pillow[:, :, 2]

zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 2))

# print(im_pillow_c1.shape)
# print(zeros.shape)

# 使用 np.newaxis 让数组增加一个维度
im_pillow_c1 = im_pillow_c1[:, :, np.newaxis]
print(im_pillow_c1.shape)

im_pillow_c1_3ch = np.concatenate((im_pillow_c1, zeros), axis=2)

print(im_pillow_c1_3ch.shape)