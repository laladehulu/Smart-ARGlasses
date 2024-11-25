import numpy as np
import skimage as sk
import skimage.io as skio
from scipy import signal
import cv2
from scipy import ndimage

def sharpen(im, sigma, alpha):
    
    height, width = im.shape[0], im.shape[1]
    R = np.dsplit(im, im.shape[2])[0].reshape((height, width))
    G = np.dsplit(im, im.shape[2])[1].reshape((height, width))
    B = np.dsplit(im, im.shape[2])[2].reshape((height, width))
    channels = [R, G, B]

    for i in range(len(channels)):
        blurred = ndimage.gaussian_filter(channels[i], sigma=sigma)
        channels[i] = channels[i]*(1+alpha) - blurred*alpha 

    channels = np.clip(channels, 0, 1)
    im_sharpen = np.dstack([channels[0], channels[1], channels[2]])
    return im_sharpen

def median(im, size):
    
    height, width = im.shape[0], im.shape[1]
    R = np.dsplit(im, im.shape[2])[0].reshape((height, width))
    G = np.dsplit(im, im.shape[2])[1].reshape((height, width))
    B = np.dsplit(im, im.shape[2])[2].reshape((height, width))
    channels = [R, G, B]

    for i in range(len(channels)):
        channels[i] = ndimage.median_filter(channels[i], size=size)

    channels = np.clip(channels, 0, 1)
    im_sharpen = np.dstack([channels[0], channels[1], channels[2]])
    return im_sharpen

im = skio.imread('hao-01.jpg')
im = sk.img_as_float(im)
#im_median = median(im, size=20)
#im_median = (im_median * 255).astype(np.uint8)

im_sharpen = sharpen(im, sigma=4, alpha=8)
im_sharpen = (im_sharpen * 255).astype(np.uint8) 
# skio.imsave('', im_sharpen)

skio.imshow(im_sharpen)
skio.show()
    

