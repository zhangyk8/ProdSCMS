#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: Oct 10, 2021

Description: This script contains code for applying our proposed mean shift 
algorithm to an Earthquake dataset (directional-linear data) (Figure 5 in the 
arxiv version of the paper). This script take more than 35 minutes to run on my 
laptop with 8 CPU cores.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time, ray
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from Utility_fun import Unique_Modes, cart2sph, sph2cart
from DirLinProdSCMS_fun import DirLinProdKDE
from DirLinProdSCMS_Ray import DirLinProdKDE_Fast, DirLinProdMS_Fast

if __name__ == "__main__":
    # Read the earthquake records with magnitude 2.5+ from 2021-07-01 to 2021-09-30
    EQ_dat = pd.read_csv('./Data/earthquake0701_0930.csv')
    # Convert the times to their associated timestamps
    EQ_dat['timestamp'] = \
        [datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() for x in EQ_dat.time]
    EQ_DirLin = EQ_dat[['longitude', 'latitude', 'timestamp']].values
    X, Y, Z = sph2cart(*EQ_dat[['longitude', 'latitude']].values.T)
    EQ_DirLin_cart = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1), 
                                     EQ_dat['timestamp'].values.reshape(-1,1)], axis=1)
    # Bandwidth selection
    data = EQ_DirLin_cart
    n = EQ_DirLin_cart.shape[0]
    q = 2
    D = 1
    data_Dir = data[:,:(q+1)]
    data_Lin = data[:,(q+1):(q+1+D)]
    ## Rule-of-thumb bandwidth selector for the directional component
    R_bar = np.sqrt(sum(np.mean(data_Dir, axis=0) ** 2))
    kap_hat = R_bar * (q + 1 - R_bar ** 2) / (1 - R_bar ** 2)
    h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
             ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
    bw_Dir = h
    print("The current bandwidth for directional component is " + str(h) + ".\n")
    ## Normal reference rule of bandwidth selector for the linear component
    b = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data_Lin, axis=0))
    bw_Lin = b
    print("The current bandwidth for linear component is "+ str(b) + ".\n")
    
    # Create a set of mesh points and estimate the density on it
    nrows, ncols = (90, 180)
    lon, lat, Z = np.meshgrid(np.linspace(-180, 180, ncols), 
                              np.linspace(-90, 90, nrows), 
                              np.linspace(np.min(EQ_dat['timestamp']), 
                                          np.max(EQ_dat['timestamp']), 200))
    xg, yg, zg = sph2cart(lon, lat)
    mesh1 = np.concatenate((xg.reshape(-1, 1), yg.reshape(-1, 1),
                            zg.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)
    print('-----Estimating the density values on the set of mesh points------\n')
    start = time.time()
    ray.init()
    mesh_0 = mesh1
    dataset = EQ_DirLin_cart
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(DirLinProdKDE_Fast.remote(mesh_0[i:(i+chunksize)], 
                                                    dataset, h=[bw_Dir, bw_Lin], 
                                                    com_type=['Dir','Lin'], dim=[2,1]))
    den_pts = ray.get(result_ids)
    den_pts = np.concatenate(den_pts, axis=0)
    ray.shutdown()
    print('Elapsed time: ' + str(time.time() - start)+'s.\n')
    
    # Project the (average) estimated density values onto the linear space
    den_lin = np.mean(den_pts.reshape(ncols, nrows, 200), axis=(0,1))
    # Estimate the density values on the original dataset and remove 20% of 
    # data points with lower density values
    d_DirLin_dat = DirLinProdKDE(EQ_DirLin_cart, EQ_DirLin_cart, h=[bw_Dir, bw_Lin], 
                      com_type=['Dir','Lin'], dim=[2,1])
    EQ_DirLin_thres = EQ_DirLin_cart[d_DirLin_dat >= np.quantile(d_DirLin_dat, 0.2)]
    
    print('-----Seeking out the local modes of estimated earthquake density------\n')
    # Proposed mean shift algorithm on the denoised data
    start = time.time()
    ray.init()
    mesh_0 = EQ_DirLin_thres
    dataset = EQ_DirLin_thres
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(DirLinProdMS_Fast.remote(mesh_0[i:(i+chunksize)], 
                                                   dataset, h=[bw_Dir, bw_Lin], 
                                                   com_type=['Dir','Lin'], 
                                                   dim=[2,1], eps=1e-7, 
                                                   max_iter=5000))
    DLMS_pts = ray.get(result_ids)
    DLMS_pts = np.concatenate(DLMS_pts, axis=0)
    ray.shutdown()
    print('Elapsed time: ' + str(time.time() - start)+'s.\n')
    
    # Plot the estimated local modes projected onto the (longitude,latitude) space
    lon_m, lat_m, R = cart2sph(*DLMS_pts[:,:3].T)
    Mode_pts = np.concatenate([lon_m.reshape(-1,1), lat_m.reshape(-1,1), 
                               DLMS_pts[:,3].reshape(-1,1)], axis=1)
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 11})   ## Change the font sizes in the generated figures
    m1 = Basemap(projection='robin', lon_0=0, resolution='c')
    # Draw coastlines, country boundaries, fill continents.
    m1.drawcoastlines(linewidth=0.25)
    m1.drawcountries(linewidth=0.25)
    m1.etopo(scale=0.5, alpha=0.2)
    # Draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m1.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x1, y1 = m1(lon_m, lat_m)
    cs = m1.scatter(x1, y1, color='red', s=40, marker='D')
    fig.tight_layout()
    fig.savefig('./Figures/EQ_mode_loc.pdf')
    
    # Plot the estimated local modes projected onto the linear/temporal space
    time_m = np.linspace(np.min(EQ_dat['timestamp']), 
                         np.max(EQ_dat['timestamp']), 200)
    fig = plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 12})   ## Change the font sizes in the generated figures
    plt.plot([datetime.fromtimestamp(x) for x in time_m], den_lin)
    plt.scatter([datetime.fromtimestamp(x) for x in Mode_pts[:,2]], 
                [np.mean(den_lin[np.argsort(abs(x - time_m))[0:2]]) for x in Mode_pts[:,2]], 
                color='red', s=40)
    plt.xlabel('Time')
    plt.ylabel('Average KDE values projected to the linear data space')
    fig.tight_layout()
    fig.savefig('./Figures/EQ_mode_time.pdf')
    print("\n Save the plot as 'EQ_mode_loc.pdf' and 'EQ_mode_time.pdf' to the"\
          " folder 'Figures'.\n\n")