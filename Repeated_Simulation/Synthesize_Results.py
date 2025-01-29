#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 26, 2025

Description: This script contains code for synthesizing the results in our repeated
simulation studies.
"""

import numpy as np
import pandas as pd
import pickle

for N in [500, 1000, 2000]:
    print('The current sample size for Simulation 1 is '+str(N)+'.\n')
    sim1_avg_EuMS = []
    sim1_avg_EuDir = []
    sim1_avg_DirLin = []
    sim1_avg_DirLin_cv = []
    for k in range(1, 1001):
        with open('./Results/Simulation1_N_'+str(N)+'_'+str(k)+'.dat', "rb") as file:
            avg_EuMS, avg_EuDir, avg_DirLinMS, avg_DirLinMS_cv = pickle.load(file)
        sim1_avg_EuMS.append(avg_EuMS)
        sim1_avg_EuDir.append(avg_EuDir)
        sim1_avg_DirLin.append(avg_DirLinMS)
        sim1_avg_DirLin_cv.append(avg_DirLinMS_cv)
        
    with open('./Syn_Results/Simulation1_N_'+str(N)+'.dat', "wb") as file:
        pickle.dump([sim1_avg_EuMS, sim1_avg_EuDir, sim1_avg_DirLin, sim1_avg_DirLin_cv], file)

    print('The average distance between the estimated and true modes for the Euclidean MS algorithm is '\
          +str(np.mean(sim1_avg_EuMS))+' with standard error: '\
          +str(np.std(sim1_avg_EuMS)/np.sqrt(len(sim1_avg_EuMS)))+'.\n')

    print('The average distance between the estimated and true modesby applying Euclidean and directional MS algorithms separately is '\
          +str(np.mean(sim1_avg_EuDir))+' with standard error: '\
          +str(np.std(sim1_avg_EuDir)/np.sqrt(len(sim1_avg_EuDir)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm is '\
          +str(np.mean(sim1_avg_DirLin))+' with standard error: '\
          +str(np.std(sim1_avg_DirLin)/np.sqrt(len(sim1_avg_DirLin)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm with LSCV bandwidth is '\
          +str(np.mean(sim1_avg_DirLin_cv))+' with standard error: '\
          +str(np.std(sim1_avg_DirLin_cv)/np.sqrt(len(sim1_avg_DirLin_cv)))+'.\n')
    

for N in [500, 1000, 2000]:
    print('The current sample size for Simulation 2 is '+str(N)+'.\n')
    sim2_avg_EuMS = []
    sim2_avg_DirDir = []
    sim2_avg_DirDir_cv = []
    for k in range(1, 1001):
        with open('./Results/Simulation2_N_'+str(N)+'_'+str(k)+'.dat', "rb") as file:
            avg_EuMS, avg_DirDirMS, avg_DirDirMS_cv = pickle.load(file)
        sim2_avg_EuMS.append(avg_EuMS)
        sim2_avg_DirDir.append(avg_DirDirMS)
        sim2_avg_DirDir_cv.append(avg_DirDirMS_cv)
    
    with open('./Syn_Results/Simulation2_N_'+str(N)+'.dat', "wb") as file:
        pickle.dump([sim2_avg_EuMS, sim2_avg_DirDir, sim2_avg_DirDir_cv], file)
        
    print('The average distance between the estimated and true modes for the Euclidean MS algorithm is '\
          +str(np.mean(sim2_avg_EuMS))+' with standard error: '\
          +str(np.std(sim2_avg_EuMS)/np.sqrt(len(sim2_avg_EuMS)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm is '\
          +str(np.mean(sim2_avg_DirDir))+' with standard error: '\
          +str(np.std(sim2_avg_DirDir)/np.sqrt(len(sim2_avg_DirDir)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm with LSCV bandwidth is '\
          +str(np.mean(sim2_avg_DirDir_cv))+' with standard error: '\
          +str(np.std(sim2_avg_DirDir_cv)/np.sqrt(len(sim2_avg_DirDir_cv)))+'.\n')
    

for N in [500, 1000, 2000]:
    print('The current sample size for Simulation 2 (Added) is '+str(N)+'.\n')
    sim2_avg_EuMS = []
    sim2_avg_DirDir = []
    sim2_avg_DirDir_cv = []
    for k in range(1, 1001):
        with open('./Results/Simulation2_N_'+str(N)+'_'+str(k)+'_added.dat', "rb") as file:
            avg_EuMS, avg_DirDirMS, avg_DirDirMS_cv = pickle.load(file)
        sim2_avg_EuMS.append(avg_EuMS)
        sim2_avg_DirDir.append(avg_DirDirMS)
        sim2_avg_DirDir_cv.append(avg_DirDirMS_cv)
    
    with open('./Syn_Results/Simulation2_N_'+str(N)+'_added.dat', "wb") as file:
        pickle.dump([sim2_avg_EuMS, sim2_avg_DirDir, sim2_avg_DirDir_cv], file)
        
    print('The average distance between the estimated and true modes for the Euclidean MS algorithm is '\
          +str(np.mean(sim2_avg_EuMS))+' with standard error: '\
          +str(np.std(sim2_avg_EuMS)/np.sqrt(len(sim2_avg_EuMS)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm is '\
          +str(np.mean(sim2_avg_DirDir))+' with standard error: '\
          +str(np.std(sim2_avg_DirDir)/np.sqrt(len(sim2_avg_DirDir)))+'.\n')
        
    print('The average distance between the estimated and true modes for our proposed MS algorithm with LSCV bandwidth is '\
          +str(np.mean(sim2_avg_DirDir_cv))+' with standard error: '\
          +str(np.std(sim2_avg_DirDir_cv)/np.sqrt(len(sim2_avg_DirDir_cv)))+'.\n')


sim3_avg_err = []
for k in range(1, 1001):
    with open('./Results/Simulation3_'+str(k)+'.dat', "rb") as file:
        avg_err = pickle.load(file)
    sim3_avg_err.append(avg_err[0].values)
    
res_err = pd.DataFrame(np.array(sim3_avg_err), columns=pd.DataFrame(avg_err[0]).index)
print('The average manifold-recovering errors for Simulation 3 are \n'+str(np.mean(res_err, axis=0))+\
      ' with standard error: \n'+str(np.std(res_err, axis=0)/res_err.shape[0]))
    

sim4_avg_err = []
for k in range(1, 1001):
    with open('./Results/Simulation4_'+str(k)+'.dat', "rb") as file:
        avg_err = pickle.load(file)
    sim4_avg_err.append(avg_err[0].values)
    
res_err = pd.DataFrame(np.array(sim4_avg_err), columns=pd.DataFrame(avg_err[0]).index)
print('The average manifold-recovering errors Simulation 4 are \n'+str(np.mean(res_err, axis=0))+\
      ' with standard error: \n'+str(np.std(res_err, axis=0)/res_err.shape[0]))