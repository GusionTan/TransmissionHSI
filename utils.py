# -*- coding: utf-8 -*-
"""
Some utility class functions..
Since the input for each model is different, the input features are initialized in each model file.
"""
import numpy as np
from sklearn.decomposition import PCA


# Load the data
def loadData(val):
    if val == 'l':
        x = np.load(r'..\data\720_120\linear_720_120.npy')
        y = np.load(r'..\data\720_120\label_720_120.npy')

    elif val == 'h':
        x = np.load(r'..\data\1490_680\linear_1490_680.npy')
        y = np.load(r'..\data\1490_680\label_1490_680.npy')

    return x[:, :, :220], y


# 0-1 normalization
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# Creating the coefficient matrix X
def create_x(size, rank):
    x = []
    for i in range(int(2 * size + 1)):
        m = i - size
        row = [m ** j for j in range(rank)]
        x.append(row)
    x = np.mat(x)
    return x


# Calculate the average spectrum of a region
def mean(I):
    I_mean = I.reshape(I.shape[0] * I.shape[1], -1)
    I_mean = np.mean(I_mean, axis=0)
    return I_mean


# PCA
def pca(X, numComponents=30, copy=True):
    pca = PCA(n_components=numComponents, copy=copy)
    newX = pca.fit_transform(X)

    return newX


# Savitzky-Golay
def savgol(data, window_size, rank):
    m = (window_size - 1) / 2
    odata = data[:]
    # Processing edge data, adding m leading and trailing items
    for i in range(int(m)):
        odata = np.insert(odata, 0, odata[0])
        odata = np.insert(odata, len(odata), odata[len(odata) - 1])
    # Creating an X matrix
    x = create_x(m, rank)
    # The weighted coefficient matrix B was calculated
    b = (x * (x.T * x).I) * x.T
    a0 = b[int(m)]
    a0 = a0.T
    # Calculate the smoothing corrected value
    ndata = []
    for i in range(len(data)):
        y = [odata[i + j] for j in range(window_size)]
        y1 = np.mat(y) * a0
        y1 = float(y1)
        ndata.append(y1)
    return ndata


# Find the slope function
def slope(x):
    y = []
    for i in range(len(x) - 1):
        y.append(int(x[i + 1]) - int(x[i]))
    y.append(0)
    return y


# Slope to unit signal function
def sign(x):
    y = []
    for i in range(len(x)):
        if x[i] > 0:
            y.append(1)
        elif x[i] < 0:
            y.append(-1)
        else:
            y.append(0)

    return y


# Find the minimum absorption point 'd', return the index of 'd'
def findpoint(x):
    vmax = 0
    i = 0
    # s-g smooth
    s1 = savgol(x, 31, 3)
    # slope
    s = slope(s1)

    while i < len(s) - 1:
        rv = 0
        # If the slope is greater than 0
        while s[i] > 0:
            # 'rv' records the maximum continuous slope
            rv += 1
            i += 1
            if i >= len(s) - 1: break
        if rv > vmax:
            vmax = rv
            index = i - rv
        i += 1

    return index


# Calculated absorptivity
def absorptivity(I_0, I_r):
    A = np.log(I_0 / I_r)
    return A


# Bins
def getBins(la):
    length = len(np.unique(np.array(la)))
    bins = range(0, length, 1)
    return bins
