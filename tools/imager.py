#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 00:16:10 2020

@author: simon
"""

import numpy as np
from tools.functions import red_value, img_value, r_map, rp_map, rs_map,rn_map

class BH_imager:
    def __init__(self, angle, disk_size):

        self.i = angle * np.pi / 180
        self.rlim = disk_size

        self.n_points = 180

        self.x = np.linspace(-16, 16, int(self.n_points * 16 / 9))
        self.y = np.linspace(-9, 9, self.n_points)

        X, Y = np.meshgrid(self.x, self.y)
        self.alpha = np.angle(X + 1j * Y)
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
