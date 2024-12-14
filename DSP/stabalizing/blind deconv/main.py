import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def richardson_lucy_blind(image, psf, num_iter=50):
    """
    Perform blind deconvolution using the Richardson-Lucy method.
    
    Parameters:
        image (numpy.ndarray): The blurred input image.
        psf (numpy.ndarray): Initial guess of the Point Spread Function (PSF).
        num_iter (int): Number of iterations for the deconvolution.
    
    Returns:
        numpy.ndarray: The restored image.
    """
    # Initialize the output image
    im_deconv = np.full(image.shape, 0.1, dtype='float')  # Start with small positive values

    for i in range(num_iter):
        print(i)
        # Mirror the PSF for convolution
        psf_mirror = np.flip(psf)

        # Perform Richardson-Lucy deconvolution
        conv = fftconvolve(im_deconv, psf, mode='same')
        relative_blur = image / (conv + 1e-8)  # Avoid division by zero
        im_deconv *= fftconvolve(relative_blur, psf_mirror, mode='same')

        # Update PSF based on the image
        im_deconv_mirror = np.flip(im_deconv)
        psf *= fftconvolve(relative_blur, im_deconv_mirror, mode='same')
        psf /= psf.sum()  # Normalize the PSF

    return psf, im_deconv


def estimate_initial_psf(shape, size=10):
    """
    Estimate an initial PSF for blind deconvolution.
    
    Parameters:
        shape (tuple): Shape of the PSF (should match the image).
        size (int): Size of the initial PSF (smaller is better for motion blur).
    
    Returns:
        numpy.ndarray: Initial estimate of the PSF.
    """
    psf = np.zeros(shape)
    center = (shape[0] // 2, shape[1] // 2)
    #psf[center[0], center[1] - size // 2:center[1] + size // 2] = 1
    psf[center[0] - size // 2:center[0] + size // 2, center[1] - size // 2:center[1] + size // 2] = 1
    '''
    psf[center[0] - 2:center[0] + 3, center[1] - 2:center[1] + 3] = np.array([
            [1, 4, 6, 4, 1], 
            [4, 16, 24, 16, 4], 
            [6, 24, 36, 24, 6], 
            [4, 16, 24, 16, 4], 
            [1, 4, 6, 4, 1]
        ])
    '''
    psf /= psf.sum()  # Normalize
    return psf


def main():
    # Load and preprocess the image
    image_path = "hao-06 resize.jpg"  # Replace with the path to your blurry image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image / 255.0  # Normalize to [0, 1]

    # Estimate an initial PSF
    initial_psf = estimate_initial_psf(image.shape, size=50)

    # Perform blind deconvolution
    psf, restored_image = richardson_lucy_blind(image, initial_psf.copy(), num_iter=60)

    # Display results
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1)
    plt.title("Original Blurred Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Init PSF")
    plt.imshow(initial_psf, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("PSF")
    plt.imshow(psf, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Restored Image")
    plt.imshow(restored_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
