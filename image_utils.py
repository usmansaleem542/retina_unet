import os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import util
import numpy as np

def merge_mask_and_image(color='red'):
    img = io.imread('test/01.png')
    msk = io.imread('test/01_test_pred.png')
    msk = resize(msk, (584,565))*256
    msk = np.uint8(msk)
    img[:,:,0]=msk
    io.imsave('test/01_resized.png', img)


def convert_image(dst, imageFile):
    path, img = os.path.split(imageFile)
    name, ext = img.split('.')
    print("processing ", img)
    original = io.imread(imageFile)
    grayscale = rgb2gray(original)
    grayscale = resize(grayscale, (512, 512))
    savePath = path +'/' +name + '.'+dst
    io.imsave(savePath, grayscale)

def convert_images(src, dst, path):
    path = 'train/image/'
    for img in os.listdir(path):
        name, ext = img.split('.')
        if ext == src:
            print("processing ", img)
            imgPath = path+img
            original = io.imread(imgPath)
            grayscale = rgb2gray(original)
            grayscale = resize(grayscale,(512,512))
            savePath = path+name+'.'+dst
            io.imsave(savePath,grayscale)


def invert_images(path, newPath=''):
    for img in os.listdir(path):
        imgPath = path + img
        original = io.imread(imgPath)
        invImg = util.invert(original)
        if newPath != '':
            imgPath = newPath+img
        io.imsave(imgPath, invImg)

# merge_mask_and_image()
# convert_image('png', '/home/usman/Desktop/temp3.jpg')
# invert_images('train/label/', 'train/label2/')