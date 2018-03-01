#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2/26/18 4:46 PM 

@author: Hantian Liu
"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

M=30
num_bins = M

obs_w = np.load('obs_w.npy')
obs_i = np.load('obs_i.npy')
obs_e = np.load('obs_e.npy')
obs_c = np.load('obs_c.npy')
obs_b3 = np.load('obs_b3.npy')
obs_b4 = np.load('obs_b4.npy')

#for motion_type in range(6):

x=[]
for obs in obs_w:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Wave cluster distribution")


x=[]
for obs in obs_e:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Eight cluster distribution")

x=[]
for obs in obs_c:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Circle cluster distribution")


x=[]
for obs in obs_i:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Inf cluster distribution")


x=[]
for obs in obs_b3:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Beat3 cluster distribution")


x=[]
for obs in obs_b4:
	x.append(obs)
plt.figure()
n, bins, patches = plt.hist(x, num_bins, facecolor = 'blue', alpha = 0.5)
plt.title("Beat4 cluster distribution")

plt.show()