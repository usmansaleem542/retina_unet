import matplotlib.pyplot as plt
from skimage.draw import random_shapes
from skimage.util import random_noise
from skimage import io
import numpy as np
import os

def show_imgae(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def gen_image(imgType, noise=True):
    if imgType == 'grayscale':
        img, _ = random_shapes((512, 512), max_shapes=30, intensity_range=(0, 255),
                                  min_size=50, multichannel=False,
                                  max_size=np.random.randint(50, 150), allow_overlap=False)
    else:
        img, _ = random_shapes((512, 512), max_shapes=30, intensity_range=(0, 255),
                                  min_size=50, multichannel=False,
                                  max_size=np.random.randint(50, 150), allow_overlap=False)
    img[img == 255] = 0
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[:, :] = img
    mask[mask > 0] = 255

    if noise:
        nsImg = random_noise(img)
    return nsImg, mask

def cook_data(savePath, count = 50, imgType='grayscale', noise = False, test_split=0.3):
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+'/train/image', exist_ok=True)
    os.makedirs(savePath+'/train/label', exist_ok=True)
    os.makedirs(savePath+'/test', exist_ok=True)
    for i in range(count):

        img, mask = gen_image(imgType)
        imgPath = savePath+'/train/'
        io.imsave(imgPath+'image/'+str(i)+'.png', img)
        io.imsave(imgPath+'label/'+str(i)+'.png', mask)
        print("Saving Image" +str(i)+'.png'+"in Train Folder")
        # show_imgae(img)

    for i in range(int(test_split*count)):
        img, mask = gen_image(imgType)
        imgPath = savePath + '/test/'
        io.imsave(imgPath + str(i) + '.png', img)
        print("Saving Image" +str(i)+'.png'+"in Test Folder")


cook_data('data/shapes')