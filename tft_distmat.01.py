# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:01:47 2021

@author: Reucus
"""

#%% import libraries

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from sklearn.manifold import MDS

#%% set up datasets

gridme = pd.read_csv('dataset.csv')

nameless = gridme.drop(columns='name')
name = np.array(gridme['name']).reshape((58,1))
names = list(name)

#%% multi dimensional scaling
#mds = MDS(2, random_state=2)
mds = MDS(2)
twodees = mds.fit_transform(nameless)

#twodeesnamed = np.concatenate((names, twodees), axis=1)

# plot MDS
x = [row[0] for row in twodees]
y = [row[1] for row in twodees]


# make the plot
plt.figure(1)
plt.rcParams['figure.figsize'] = [10,10]
plt.rc('font', size=12)
plt.scatter(x, y, s=20, color='blue')
plt.title("direct MDS from OHE traits",fontsize=15)

for i, label in enumerate(names):
    plt.text(x[i], y[i], label)
    
#%% build distance matrix

# initialize empty matrix
numchamps = gridme.shape[0]
dmat = np.zeros(shape=(numchamps, numchamps))


# fill in dmat
def gimmechamp(index):
    return np.array(list(nameless.iloc[index]))

def dboii(a, b):
    return  np.linalg.norm(a - b)

for x in range(0, numchamps):
    for y in range(0, numchamps):
        dmat[x, y] = dboii(gimmechamp(x), gimmechamp(y))


#%% visualize distance matrix

# MDS again
dmat_mds = mds.fit_transform(dmat)

# plot MDS
xd = [row[0] for row in dmat_mds]
yd = [row[1] for row in dmat_mds]


# make the plot
plt.figure()
plt.rcParams['figure.figsize'] = [10,10]
plt.rc('font', size=12)

plt.scatter(xd, yd, s=20, color='red')
plt.title("MDS from distance matrix",fontsize=15)

for i, label in enumerate(names):
    plt.text(xd[i], yd[i], label)


# voroni or force-directed graph

#%% named dist mat

name2 = np.insert(name, 0, np.nan).reshape((1, 59))
n_dmat = np.concatenate((name, dmat), axis=1)
n_dmat = np.concatenate((name2, n_dmat), axis=0)

np.savetxt("dmat.csv", dmat, delimiter=",")
np.savetxt("n_dmat.csv", n_dmat, delimiter=",")