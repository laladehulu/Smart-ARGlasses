import numpy as np
import cv2
import skimage as sk
import skimage.io as skio
from skimage.feature import corner_harris, peak_local_max
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal, stats, ndimage
import random
from numpy.linalg import inv




def get_harris_corners(im, sigma, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=sigma)

    # threshold by median
    h_median = np.median(h)
    print(f"\tmedian of h: {h_median}")
    h_thres = np.copy(h)
    h_thres[h_thres<h_median] = 0

    # local max
    coords = peak_local_max(h_thres, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    
    return h, h_thres, coords



def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert(dimx == dimc, 'Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)



def conv1(im, sigma):

    # Gaussian
    size = sigma*3*2+1
    G_1d = cv2.getGaussianKernel(size, sigma)
    G_2d = np.matmul(G_1d, np.transpose(G_1d))
    mask = G_2d
    
    # convolution
    im = signal.convolve2d(im, mask, boundary='symm', mode='same')

    # clip
    im = np.clip(im, 0, 1)
    
    return im



def conv3(im, sigma):

    # Gaussian
    size = sigma*3*2+1
    G_1d = cv2.getGaussianKernel(size, sigma)
    G_2d = np.matmul(G_1d, np.transpose(G_1d))
    mask = G_2d
    
    # split channel
    height, width = im.shape[0], im.shape[1]
    R = np.dsplit(im, im.shape[2])[0].reshape((height, width))
    G = np.dsplit(im, im.shape[2])[1].reshape((height, width))
    B = np.dsplit(im, im.shape[2])[2].reshape((height, width))
    channels = [R, G, B]
    
    # convolution
    for i in range(len(channels)):
        channels[i] = signal.convolve2d(channels[i], mask, boundary='symm', mode='same')

    # clip
    channels = np.clip(channels, 0, 1)

    # save
    masked = np.dstack([channels[0], channels[1], channels[2]])
    
    return masked



# H(im1) = im2
# solve linear equations (least square)
def computeH(im1_pts,im2_pts):

    k = im1_pts.shape[0]
    A = np.zeros((2*k, 8))
    b = np.zeros((2*k, 1))

    for i in range(k):

        x, y = im1_pts[i][0], im1_pts[i][1]
        X, Y = im2_pts[i][0], im2_pts[i][1]

        # ax+by+c-gxX-hyX = X
        # dx+ey+f-gxY-hyY = Y

        A[2*i+0][0] = x
        A[2*i+0][1] = y
        A[2*i+0][2] = 1
        A[2*i+0][6] = -x*X
        A[2*i+0][7] = -y*X
        b[2*i+0][0] = X

        A[2*i+1][3] = x
        A[2*i+1][4] = y
        A[2*i+1][5] = 1
        A[2*i+1][6] = -x*Y
        A[2*i+1][7] = -y*Y
        b[2*i+1][0] = Y

    ans = np.linalg.lstsq(A, b)[0]
    ans = np.array([
            [ans[0][0], ans[1][0], ans[2][0]],
            [ans[3][0], ans[4][0], ans[5][0]],
            [ans[6][0], ans[7][0],         1],
        ])
    return ans



def normalize(M):
    '''
        [
            [x1, x2, x3, ...],
            [y1, y2, y3, ...],
            [w1, w2, w3, ...],
        ]
        --->
        [
            [x1/w1, x2/w2, x3/w3, ...],
            [y1/w1, y2/w2, y3/w3, ...]
        ]
    '''
    return np.array([M[0]/M[2], M[1]/M[2]])



def distance_map(shape):
    coor = np.indices((shape[0], shape[1]))
    dist = coor[0], shape[0]-1-coor[0], coor[1], shape[1]-1-coor[1]
    distance_map = np.min(dist, axis=0)
    distance_map = (distance_map-np.min(distance_map))/(np.max(distance_map)-np.min(distance_map))  # normalize to 0-1
    return distance_map
