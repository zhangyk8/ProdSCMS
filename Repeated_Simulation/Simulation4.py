#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: June 26, 2024

Description: This script contains code for comparing the results of the regular 
SCMS and our proposed SCMS algorithms on the simulated spherical cone data 
(Simulation 4 in the paper).
"""

import numpy as np
import pandas as pd
from numpy import linalg as LA
import pickle
import ray
from Utility_fun import CircleOnSphereSampling, cart2sph, RandomPtsCone
from MS_SCMS_Ray import SCMS_Log_KDE_Fs
from DirLinProdSCMS_Ray import DirLinProdSCMSLog_Fast

# job_id = int(sys.argv[1])
# print(job_id)

#=======================================================================================#


def DistToCone(pt, theta):
    pt = pt.values
    z = pt[2]
    return abs(z*np.tan(theta) - np.sqrt(pt[0]**2 + pt[1]**2))*np.sin(theta)

def SurfRecoverError(pt, Ridge):
    pt = pt.values
    return np.min(LA.norm(pt - Ridge, axis=1))

for job_id in range(1, 1001):
    ## Simulation 4: Surface-Recovering on $\Omega_2 \times R$
    N = 2000
    lat_cir = 45   ## Latitude of the circle on the sphere
    sig = 0.1
    np.random.seed(123)  ## Set an arbitrary seed for reproducibility
    cir_samp = CircleOnSphereSampling(N, lat_c=lat_cir, sigma=sig, 
                                      pv_ax=np.array([0,0,1]))
    lon_c, lat_c, r = cart2sph(*cir_samp.T)
    cir_samp_ang = np.concatenate((lon_c.reshape(len(lon_c),1), 
                                   lat_c.reshape(len(lat_c),1)), axis=1)
    R = np.random.rand(N, 1)*2    ## Simulated radii of spheres
    sim_dat1 = np.concatenate([cir_samp, R], axis=1)
    sim_dat1_ang = np.concatenate([cir_samp_ang, R], axis=1)
    Phi = ((lon_c/180)*np.pi).reshape(-1,1)
    Eta = ((lat_c/180)*np.pi).reshape(-1,1)
    sim_dat1_3D = np.concatenate([R*np.cos(Phi)*np.cos(Eta), 
                                  R*np.sin(Phi)*np.cos(Eta), 
                                  R*np.sin(Eta)], axis=1)
    
    # Regular SCMS on the simulated data with 3D Cartesian coordinates
    ray.init(include_dashboard=False)
    mesh_0 = sim_dat1_3D
    dataset = sim_dat1_3D
    bw_Eu1 = None
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_KDE_Fs.remote(mesh_0[i:(i+chunksize)], dataset, 
                                                 d=2, h=bw_Eu1, eps=1e-7, 
                                                 max_iter=5000))
    EuSCMS_pts1 = ray.get(result_ids)
    EuSCMS_pts1 = np.concatenate(EuSCMS_pts1, axis=0)
    ray.shutdown()
    
    # Regular SCMS on the simulated data with 3D angular-linear coordinates 
    # i.e., (longitudes, latitudes, linear covariates)
    ray.init(include_dashboard=False)
    mesh_0 = sim_dat1_ang
    dataset = sim_dat1_ang
    bw_Eu2 = None
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(SCMS_Log_KDE_Fs.remote(mesh_0[i:(i+chunksize)], dataset, 
                                                 d=2, h=bw_Eu2, eps=1e-7, 
                                                 max_iter=5000))
    EuSCMS_pts2 = ray.get(result_ids)
    EuSCMS_pts2 = np.concatenate(EuSCMS_pts2, axis=0)
    ray.shutdown()

    Phi = (EuSCMS_pts2[:,0]/180)*np.pi
    Eta = (EuSCMS_pts2[:,1]/180)*np.pi
    Eu_Ridges2 = np.concatenate([(EuSCMS_pts2[:,2]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                 (EuSCMS_pts2[:,2]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                 (EuSCMS_pts2[:,2]*np.sin(Eta)).reshape(-1,1)], axis=1)
    
    # Our proposed DirLin SCMS on the simulated data in the directional-linear 
    # space $\Omega_2 \times \mathbb{R}$
    ray.init(include_dashboard=False)
    mesh_0 = sim_dat1
    dataset = sim_dat1
    chunksize = 10
    num_p = mesh_0.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(DirLinProdSCMSLog_Fast.remote(mesh_0[i:(i+chunksize)], 
                                                        dataset, d=2, h=[None,None], 
                                                        com_type=['Dir','Lin'], 
                                                        dim=[2,1], eps=1e-7, 
                                                        max_iter=5000))
    DLSCMS_pts = ray.get(result_ids)
    DLSCMS_pts = np.concatenate(DLSCMS_pts, axis=0)
    ray.shutdown()

    lon, lat, R = cart2sph(*DLSCMS_pts[:,:3].T)
    Phi = (lon/180)*np.pi
    Eta = (lat/180)*np.pi
    DL_Ridges = np.concatenate([(DLSCMS_pts[:,3]*np.cos(Phi)*np.cos(Eta)).reshape(-1,1), 
                                (DLSCMS_pts[:,3]*np.sin(Phi)*np.cos(Eta)).reshape(-1,1),
                                (DLSCMS_pts[:,3]*np.sin(Eta)).reshape(-1,1)], axis=1)
    
    # Compute the manifold recovering errors of the estimated ridges/principal surface
    ## Euclidean distance errors from estimated ridges to the true spherical cone
    DLRidge_Err = \
        pd.DataFrame(DL_Ridges).apply(lambda x: DistToCone(pt=x, theta=(90-lat_cir)*np.pi/180), 
                                      axis=1)
    EuRidge1_Err = \
        pd.DataFrame(EuSCMS_pts1).apply(lambda x: DistToCone(pt=x, theta=(90-lat_cir)*np.pi/180), 
                                        axis=1)
    EuRidge2_Err = \
        pd.DataFrame(Eu_Ridges2).apply(lambda x: DistToCone(pt=x, theta=(90-lat_cir)*np.pi/180), 
                                       axis=1)
    DistErr_df = pd.DataFrame({'SCMS_3D': EuRidge1_Err, 
                               'SCMS_2Ang_1Lin': EuRidge2_Err, 
                               'DirLinSCMS_Omega2_Lin': DLRidge_Err})
    ## Euclidean distance errors from the true spherical cone to estimated ridges
    # Sample 5000 random observations on the true spherical cone
    rand_pts_cone = RandomPtsCone(N=5000, semi_open_ang=90-lat_cir, zmin=0, zmax=2, 
                                  pv_ax=np.array([0,0,1]))
    DLRidge_SurfRecErr = \
        pd.DataFrame(rand_pts_cone).apply(lambda x: SurfRecoverError(pt=x, Ridge=DL_Ridges), 
                                          axis=1)
    EuRidge1_SurfRecErr = \
        pd.DataFrame(rand_pts_cone).apply(lambda x: SurfRecoverError(pt=x, Ridge=EuSCMS_pts1), 
                                          axis=1)
    EuRidge2_SurfRecErr = \
        pd.DataFrame(rand_pts_cone).apply(lambda x: SurfRecoverError(pt=x, Ridge=Eu_Ridges2), 
                                          axis=1)
    SurfRecErr_df = pd.DataFrame({'SCMS_3D': EuRidge1_SurfRecErr, 
                                  'SCMS_2Ang_1Lin': EuRidge2_SurfRecErr, 
                                  'DirLinSCMS_Omega2_Lin': DLRidge_SurfRecErr})
    # Manifold recovering errors of the estimated ridges
    print('The manifold recovering errors of the estimated ridges are \n')
    print((np.mean(DistErr_df, axis=0) + np.mean(SurfRecErr_df, axis=0))/2)
    print('\n')


    with open('./Results/Simulation4_'+str(job_id)+'.dat', "wb") as file:
        pickle.dump([(np.mean(DistErr_df, axis=0) + np.mean(SurfRecErr_df, axis=0))/2], file)
