# https://www.geeksforgeeks.org/im-enhancement-techniques-using-opencv-python/

import cv2
import numpy as np
import skimage as sk
import skimage.io as skio

class image():

    def __init__(self, name):
        self.name = name
        self.im = cv2.imread(name)

    def brightneww_contrast(self, brightness=0, contrast=1.0):  # https://www.geeksforgeeks.org/addition-blending-ims-using-opencv-python/
        self.im = cv2.addWeighted(self.im, contrast, np.zeros(self.im.shape, self.im.dtype), 0, brightness)
        return self

    def sharpen(self, kernal=np.array([[0, -1.5, 0], [-1.5, 7.5, -1.5], [0, -1.5, 0]])):
        self.im = cv2.filter2D(self.im, -1, kernel)
        return self
        
    def median(self, sizez=21):
        self.im = cv2.medianBlur(self.im, size)
        return self

    def Gaussian(self, kernal=(7,7)):
        self.im = cv2.GaussianBlur(self.im, kernal, 0)
        return self

    def enhancement(self):
        self.im = cv2.cvtColor(self.im, cv2.COLOR_RGB2HSV) 
        self.im[:, :, 0] = self.im[:, :, 0] * 0.7
        self.im[:, :, 1] = self.im[:, :, 1] * 1.5
        self.im[:, :, 2] = self.im[:, :, 2] * 0.5
        self.im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return self
        
    def inverse(self):
        self.im = 255 - self.im
        return self

    def histo(self):  # https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-im
        self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2YCrCb)
        self.im[:, :, 0] = cv2.equalizeHist(self.im[:, :, 0])
        self.im = cv2.cvtColor(self.im, cv2.COLOR_YCrCb2BGR)
        return self

    def show(self):
        skio.imshow(self.im)
        skio.show()

im = image('hao-06.jpg').brightneww_contrast(brightness=0, contrast=2.0).show()


    
