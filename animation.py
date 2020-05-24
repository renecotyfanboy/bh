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
    
images = []

for i in tqdm(np.arange(0,361,1)):
    bh = BH_imager(angle=i,disk_size=(3,7),n_points=180)
    images.append(bh.compute_img())
    
block = amp.blocks.Imshow(images,cmap=cm.hot,origin='lower')
anim = amp.Animation([block])

anim.controls()
anim.save_gif('ising')
plt.show()