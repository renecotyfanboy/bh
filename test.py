#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:54:00 2020

@author: simon
"""

#%% Example code
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from os.path import join
from tqdm import tqdm
from matplotlib import cm
from matplotlib import colors,image
from tools.imager import BH_imager
import animatplot as amp

#with tempfile.TemporaryDirectory() as tmpdirname:
bh = BH_imager(angle=10,pixel_size=0.0125)
img = bh.compute_img()

np.savetxt('img.txt',img)

# plt.imshow(img,cmap=cm.hot,origin='lower')
# plt.axis('off')