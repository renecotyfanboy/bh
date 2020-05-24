"""
Imager class wrapping various computations
"""

import numpy as np
from tools.functions import red_value, img_value, r_map, rp_map, rs_map,rn_map

class BH_imager:
    def __init__(self, angle=30, disk_size=(3,15),n_points=180):

        self.i = (angle * np.pi / 180)%(2*np.pi)
        self.rlim = disk_size

        self.x = np.linspace(-10, 10, n_points)
        self.y = np.linspace(-10, 10, n_points)

        X, Y = np.meshgrid(self.x, self.y)
        self.alpha = np.arctan2(Y,X)
        self.b = np.abs(X + 1j * Y)

        self.R_map = r_map(self.rlim[0], self.rlim[1], self.i, self.alpha, self.b)

    def compute_img(self):

        return img_value(
            self.R_map, self.rlim[0], self.rlim[1], self.i, self.alpha, self.b
        )

    def compute_redshift(self):

        return red_value(
            self.R_map, self.rlim[0], self.rlim[1], self.i, self.alpha, self.b
        )

    def compute_rp_map(self):
        
        return rp_map(self.i, self.alpha, self.b)
    
    def compute_rs_map(self):
        
        return rs_map(self.i, self.alpha, self.b)
    
    def compute_rn_map(self):

        return rn_map(self.i, self.alpha, self.b)
