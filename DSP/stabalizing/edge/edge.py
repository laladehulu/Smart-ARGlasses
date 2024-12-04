import numpy as np
import skimage as sk
import skimage.io as skio
from scipy import signal
import cv2 



im = skio.imread('hao-06.jpg')
im = sk.color.rgb2gray(im)
skio.imsave('im_gray.jpg', (im * 255).astype(np.uint8))
im = sk.img_as_float(im)

# define Gaussian
G_1d = cv2.getGaussianKernel(37, 6)
G_2d = np.matmul(G_1d, np.transpose(G_1d))

im = signal.convolve2d(im, G_2d, boundary='symm', mode='same')
blurred = np.dstack([im, im, im])
blurred  = (blurred * 255).astype(np.uint8)
skio.imsave('blurred.jpg', blurred)
        
Dx = np.array([[1, -1]])
Dy = np.array([[1], [-1]])
dim_dx = signal.convolve2d(im, Dx, boundary='symm', mode='same')
dim_dy = signal.convolve2d(im, Dy, boundary='symm', mode='same')
    
# gradient magnitude
gradient_mag = np.sqrt(np.square(dim_dx) + np.square(dim_dy))
gradient_mag = gradient_mag / np.max(gradient_mag)
gradient_mag_out = np.dstack([gradient_mag, gradient_mag, gradient_mag])
gradient_mag_out = (gradient_mag_out * 255).astype(np.uint8)
skio.imsave('gradient_mag.jpg', gradient_mag_out)     

# edge image
thresholds = np.arange(0.004, 0.041, 0.004)
for threshold in thresholds:
    edge = np.heaviside(gradient_mag - threshold, 1)
    edge_out = np.dstack([edge, edge, edge])
    edge_out = (edge_out * 255).astype(np.uint8)
    skio.imsave('edge '+"{:.3f}".format(threshold)+'.jpg', edge_out)



