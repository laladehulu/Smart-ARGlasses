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
from utils import get_harris_corners, dist2, conv1, conv3, computeH, normalize, distance_map




class image():

    def __init__(self, name):

        self.name = name

        print(f"Executing: Initializing -- {self.name}")
        
        # read image
        self.im = sk.img_as_float(skio.imread(f'{self.name}.jpg'))

        # create BW version
        self.im_BW = np.mean(self.im, axis=2)
        
        self.im_BW3 = np.dstack([self.im_BW]*3)
        skio.imsave(f'{self.name}_BW.jpg', (self.im_BW * 255).astype(np.uint8))
     
        # not important
        self.inf = 999999999999



    def GetHarrisCorners(self, HarrisSigma=5):

        print(f"Executing: Get Harris Corners -- {self.name}")
        
        self.h, self.h_thres, self.Harris_coords_tr = get_harris_corners(self.im_BW, HarrisSigma, edge_discard=20)
        self.Harris_coords = np.transpose(self.Harris_coords_tr)       
        print(f"\tnum of Harris corners: {self.Harris_coords.shape[0]}")

        # visualize h value
        h_out = self.h/np.max(self.h)
        plt.imshow(h_out, cmap='hot', interpolation='nearest')
        plt.savefig(f'{self.name}_h.jpg', dpi=300)
        plt.close()

        # visualize Harris Corners
        self.visualize([(self.Harris_coords, 'oy')], 'HarrisCorners')


        
    def ANMS(self, c_robust=0.9, num_of_ANMS_points=1000):

        print(f"Executing: ANMS -- {self.name}")

        # h_of_Harris_coords[k] = h(Harris_coords[k])
        self.h_of_Harris_coords = [self.h[self.Harris_coords[i][0]][self.Harris_coords[i][1]] for i in range(len(self.Harris_coords))]
        self.h_of_Harris_coords = np.array(self.h_of_Harris_coords)

        # h_of_col[m][n] = h(Harris_coords[n])
        self.h_of_col = np.stack([self.h_of_Harris_coords]*self.h_of_Harris_coords.shape[0])
      
        # h_of_row[m][n] = h(Harris_coords[m])
        self.h_of_row = np.transpose(self.h_of_col)

        # dist_matrix[m][n] = dist(Harris_coords[m], Harris_coords[n])
        self.Harris_coords_dist_matrix = np.sqrt(dist2(self.Harris_coords, self.Harris_coords))
  
        # consider h value
        self.Harris_coords_dist_matrix[self.h_of_row >= c_robust*self.h_of_col] = self.inf

        # min_dist_of_coords[k] = ri(Harris_coords[k])
        self.min_dist_of_Harris_coords = np.min(self.Harris_coords_dist_matrix, axis=1)

        # sort by ri(Harris_coords[k])
        ind = np.argsort(self.min_dist_of_Harris_coords)
        self.Harris_coords_sorted_by_ri_increasing = self.Harris_coords[ind]
        self.Harris_coords_sorted_by_ri_decreasing = self.Harris_coords_sorted_by_ri_increasing[::-1]

        # ANMS_coords
        self.ANMS_coords = self.Harris_coords_sorted_by_ri_decreasing[:num_of_ANMS_points]
        print(f"\tnum of points after ANMS: {self.ANMS_coords.shape[0]}")
        self.visualize([(self.ANMS_coords, 'or')], 'ANMS_Points')
        
    def visualize(self, lst, name, markersize=0.3):

        for i in range(len(lst)):
            points, marker = lst[i][0], lst[i][1]
            points_tr = np.transpose(points)
            plt.plot(points_tr[1], points_tr[0], marker, markersize=markersize)
        plt.imshow(self.im_BW3)
        plt.savefig(f'{self.name}_{name}.jpg', dpi=300)
        plt.close()



# Main

im = image("hao-06")

im.GetHarrisCorners()

im.ANMS()
