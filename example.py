#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:23:49 2020

@author: simon
"""

#%% Example code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from tools.imager import BH_imager

#%% Image Computation

bh = BH_imager(10, (3, 15))
img = bh.compute_img()

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(img, origin="lower", cmap=cm.hot)
ax.axis("off")

#%% Redshift Visualization

redshift = bh.compute_redshift()
redshift_max = np.max(np.abs(redshift[~np.isnan(redshift)]))

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(
    redshift,
    origin="lower",
    cmap=cm.RdBu_r,
    norm=colors.Normalize(vmin=-redshift_max, vmax=redshift_max),
)
ax.axis("off")

#%% Isoradius Visualization

R_p = bh.compute_rp_map()
R_s = bh.compute_rs_map()

fig, ax = plt.subplots(figsize=(16, 9))
levels = [3 + 1.2 * i for i in range(8)]
CS_p = ax.contour(bh.x, bh.y, R_p, levels=levels, cmap=cm.inferno)
CS_s = ax.contour(bh.x ,bh.y, R_s,levels=levels,cmap=cm.inferno)

for i in range(len(levels)):
    CS_p.collections[i].set_label(levels[i])

plt.legend(loc="upper left")
ax.legend()
ax.set_xlim(-16, 16)
ax.set_ylim(-9, 9)
plt.tight_layout()

#%% 

R_n = bh.compute_rn_map()

fig, ax = plt.subplots(figsize=(16, 9))
levels = [3 + 1.2 * i for i in range(8)]
CS_n = ax.contour(bh.x, bh.y, R_n, levels=levels, cmap=cm.inferno)

for i in range(len(levels)):
    CS_n.collections[i].set_label(levels[i])

plt.legend(loc="upper left")
ax.legend()
ax.set_xlim(-16, 16)
ax.set_ylim(-9, 9)
plt.tight_layout()