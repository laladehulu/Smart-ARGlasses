from skimage.restoration import estimate_sigma
from skimage.feature import canny
from skimage.restoration import unsupervised_wiener

import cv2
import numpy as np


image_path = "hao-06.jpg"

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale for processing
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Estimate the noise level in the image (used for PSF estimation)
sigma_est = estimate_sigma(gray_image)

# Edge detection to help with PSF estimation
edges = canny(gray_image, sigma=2)

# Estimate the PSF using a simple unsupervised Wiener deconvolution approach
psf = np.ones((15, 15)) / 15  # Initial guess for PSF (motion blur kernel)
deconvolved_image, psf_estimated = unsupervised_wiener(gray_image, psf)

# Save PSF and processed results for visualization
deconvolved_image_path = "deconvolved_image.jpg"
cv2.imwrite(deconvolved_image_path, (deconvolved_image * 255).astype(np.uint8))
