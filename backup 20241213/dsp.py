import cv2
import numpy as np

def bright_contrast(im, brightness=0, contrast=1.0):
	im = np.array(im)
	return cv2.addWeighted(im, contrast, np.zeros(im.shape, im.dtype), 0, brightness)
