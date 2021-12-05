from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

im = Image.open("../../images/pytorch_2.png")
im_pillow = np.asarray(im)

im_pillow_c1 = im_pillow[:, :, 0]
im_pillow_c2 = im_pillow[:, :, 1]
im_pillow_c3 = im_pillow[:, :, 2]

zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 2))
im_pillow_c1 = im_pillow_c1[:, :, np.newaxis]
im_pillow_c2_3ch = np.zeros(im_pillow.shape)
im_pillow_c2_3ch[:, :, 1] = im_pillow_c2

im_pillow_c1_3ch = np.concatenate((im_pillow_c1, zeros), axis=2)

im_pillow_c3_3ch = np.zeros(im_pillow.shape)
im_pillow_c3_3ch[:, :, 2] = im_pillow_c3

plt.subplot(2, 2, 1)
plt.title("Origin Image")
plt.imshow(im_pillow)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.title("Red Channel")
plt.imshow(im_pillow_c1_3ch.astype(np.uint8))
plt.axis("off")
plt.subplot(2, 2, 3)
plt.title("Green Channel")
plt.imshow(im_pillow_c2_3ch.astype(np.uint8))
plt.axis("off")
plt.subplot(2, 2, 4)
plt.title("Blue Channel")
plt.imshow(im_pillow_c3_3ch.astype(np.uint8))
plt.axis("off")
plt.savefig("./rgb_pillow.png", dpi=150)
