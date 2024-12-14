import cv2
import sys
from screeninfo import get_monitors


count = sys.argv[1]
img = cv2.imread("temp.jpg")

'''
screen = get_monitors()[0]
cv2.namedWindow(count, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(count, screen.x-1, screen.y-1)
cv2.setWindowProperty(count, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
'''

cv2.imshow(count, img)
cv2.waitKey(5000)
cv2.destroyWindow(count)

