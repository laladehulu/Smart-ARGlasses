from picamzero import Camera
from time import sleep

cam = Camera()
cam.start_preview()
cam.capture_sequence('hao.jpg', num_images=10, interval=5)
cam.stop_preview()

