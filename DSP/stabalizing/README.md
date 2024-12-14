# Notes and resource of stablizing shaky images

## Recent researches
[a list of recent researches](https://github.com/subeeshvasu/Awesome-Deblurring?tab=readme-ov-file), some has source code
### Single-Image-Blind-Motion-Deblurring (non-DL)
- [Removing Camera Shake from a Single Photograph](https://people.csail.mit.edu/billf/publications/Removing_Camera_Shake.pdf)
    - only matlab code
- [High-quality Motion Deblurring from a Single Image](https://www.cse.cuhk.edu.hk/~leojia/projects/motion_deblurring/index.html)
    - tested [executable](https://www.cse.cuhk.edu.hk/~leojia/programs/deblurring/deblurring.htm), takes long, no helpful
- [Psf estimation using sharp edge prediction](https://neelj.com/projects/psf_estimation/psf_estimation.pdf)
    - no existing code
    - [discussion](https://dsp.stackexchange.com/questions/132/how-do-i-remove-motion-blur)

### Single-Image-Blind-Motion-Deblurring (DL)
- [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)
    - [code](https://github.com/DmitryUlyanov/deep-image-prior)
    - tested on colab, takes super long

## deconvblind

### [implementing deconvblind](https://stackoverflow.com/questions/68270030/is-there-a-function-in-python-similar-to-matlabs-deconvblind)
- tested (blind deconv), not working (but the code make sense)

## Deconvolution
- [scikit-image](https://scikit-image.org/docs/stable/auto_examples/filters/plot_deconvolution.html)
- [wiki](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)

## chatgpt
- asked it to generate some code 
- tested (chatgpt solution), not helpful

# edge detection
- (edge), not helpful

# feature matching
- (matching test), can identify interesting points, but not helpful for PSF
