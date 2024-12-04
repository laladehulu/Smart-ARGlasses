# Load the uploaded image to check its content
image_path = "hao-06.jpg"

import cv2
import numpy as np

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def estimate_psf(image, kernel_size=15):
    """
    Estimate the point spread function (PSF) using motion blur kernel.
    """
    # Use edge detection to approximate the motion direction of the blur
    edges = cv2.Canny(image, 50, 150)
    
    # Estimate the PSF by analyzing the dominant motion direction
    psf_kernel = np.zeros((kernel_size, kernel_size))
    psf_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    psf_kernel /= kernel_size

    return psf_kernel

# Estimate the PSF from the deblurred image
psf = estimate_psf(gray_image, kernel_size=15)

# Visualize the PSF
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.title("Estimated Point Spread Function (PSF)")
plt.imshow(psf, cmap='gray')
plt.axis('off')
plt.show()


from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy

def deconvolve_with_psf(image, psf, num_iter=30):
    """
    Perform image deconvolution using the Richardson-Lucy algorithm.
    """
    deconvolved = richardson_lucy(image, psf, num_iter=num_iter)
    return deconvolved

# Perform deconvolution with the estimated PSF
recovered_image = deconvolve_with_psf(gray_image, psf, num_iter=30)

# Save and display the recovered image
recovered_image_path = "recovered_image.jpg"
cv2.imwrite(recovered_image_path, (recovered_image * 255).astype(np.uint8))

