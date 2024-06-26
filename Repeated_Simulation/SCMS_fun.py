#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: March 21, 2021

Description: This script contains code for Euclidean KDE and subspace 
constrained mean shift (SCMS) algorithm with Gaussian kernel.
"""

import numpy as np
from numpy import linalg as LA


def KDE(x, data, h=None, wt=None, bw_varied=False):
    '''
    d-dim Euclidean KDE with Gaussian kernel.
    
    Parameters:
        x: (m,d)-array
            The coordinates of m query points in the d-dim Euclidean space.
    
        data: (n,d)-array
            The coordinates of n random sample points in the d-dimensional 
            Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
            
        bw_varied: boolean
            The indicator of whether bandwidth should be varied for each 
            coordinate. If True, the argument "h" should be None or (d,)-array.
            (Default: False.)
    
    Return:
        f_hat: (m,)-array
            The corresponding kernel density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Dimension of the data
    
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    f_hat = np.zeros((x.shape[0], ))
    if wt is None:
        wt = np.ones((n,))/n
    for i in range(x.shape[0]):
        f_hat[i] = np.sum(wt*np.exp(np.sum(-((x[i,:] - data)/h)**2, axis=1)/2))/ \
                   ((2*np.pi)**(d/2)*np.prod(h))
    return f_hat


def MS_KDE(mesh_0, data, h=None, eps=1e-7, max_iter=1000, wt=None, bw_varied=False):
    '''
    Mean Shift Algorithm with Gaussian kernel.
    
    Parameters:
        mesh_0: a (m,d)-array
            The coordinates of m initial points in the d-dim Euclidean space.
    
        data: a (n,d)-array
            The coordinates of n data sample points in the d-dim Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
        
        bw_varied: boolean
            The indicator of whether bandwidth should be varied for each 
            coordinate. If True, the argument "h" should be None or (d,)-array.
            (Default: False.)
    
    Return:
        MS_path: (m,d,T)-array
            The entire iterative MS sequence for each initial point.
    '''
    
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    MS_path = np.zeros((mesh_0.shape[0], d, max_iter))
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    if wt is None:
        wt = np.ones((n,))
    else:
        wt = n*wt
    MS_path[:,:,0] = mesh_0
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The MS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pt = MS_path[i,:,t-1]
                ker_w = wt*np.exp(-np.sum(((x_pt-data)/h)**2, axis=1)/2)
                # Mean shift update
                x_new = np.sum(data*ker_w.reshape(n,1), axis=0) / np.sum(ker_w)
                if LA.norm(x_pt - x_new) < eps:
                    conv_sign[i] = 1
                MS_path[i,:,t] = x_new
            else:
                MS_path[i,:,t] = MS_path[i,:,t-1]
    
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return MS_path[:,:,:t]


def MS_KDETest(mesh_0, data, h=None, eps=1e-7, max_iter=1000, wt=None, bw_varied=False,sF=1):
    '''
    Mean Shift Algorithm with Gaussian kernel.
    
    Parameters:
        mesh_0: a (m,d)-array
            The coordinates of m initial points in the d-dim Euclidean space.
    
        data: a (n,d)-array
            The coordinates of n data sample points in the d-dim Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the MS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
        
        bw_varied: boolean
            The indicator of whether bandwidth should be varied for each 
            coordinate. If True, the argument "h" should be None or (d,)-array.
            (Default: False.)
    
    Return:
        MS_path: (m,d,T)-array
            The entire iterative MS sequence for each initial point.
    '''
    
    n = data.shape[0]   ## Number of data points
    d = data.shape[1]   ## Dimension of the data
    
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    MS_path = np.zeros((mesh_0.shape[0], d, max_iter))
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], )) 
    if wt is None:
        wt = np.ones((n,))
    else:
        wt = n*wt
    MS_path[:,:,0] = mesh_0
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The MS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pt = MS_path[i,:,t-1]
                ker_w = wt*np.exp(-np.sum(((x_pt-data)/h)**2, axis=1)/2)
                # Mean shift update
                x_new = x_pt + sF*(np.sum(data*ker_w.reshape(n,1), axis=0) / np.sum(ker_w) - x_pt) / h**2
                if LA.norm(x_pt - x_new) < eps:
                    conv_sign[i] = 1
                MS_path[i,:,t] = x_new
            else:
                MS_path[i,:,t] = MS_path[i,:,t-1]
    
    if t >= max_iter-1:
        print('The MS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return MS_path[:,:,:t]


def SCMS_KDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, wt=None, bw_varied=False,
             stop_cri='proj_grad'):
    '''
    Subspace Constrained Mean Shift Algorithm with Gaussian kernel
    
    Parameters:
        mesh_0: a (m,D)-array
            The coordinates of m initial points in the D-dim Euclidean space.
    
        data: a (n,D)-array
            The coordinates of n data sample points in the D-dim Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
       
        stop_cri: string ('proj_grad'/'pts_diff')
            The indicator of which stopping criteria that will be used to 
            terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors 
            between two consecutive iteration points need to be smaller than 
            'eps' for terminating the algorithm. When stop_cri='proj_grad' or 
            others, the projected/principal gradient of the current point need to be 
            smaller than 'eps' for terminating the algorithm.)
            (Default: stop_cri='proj_grad'.)
            
    Return:
        SCMS_path: (m,D,T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Dimension of data points
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    if wt is None:
        wt = np.ones((n,1))
    else:
        wt = wt.reshape(n,1)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                ## Compute the Hessian matrix
                Hess = np.dot((x_pts-data).T, wt*(x_pts-data) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1))/(h**2) \
                       - np.diag(np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) \
                                 * np.ones(len(x_pts),))
                ## Spectral decomposition
                w, v = LA.eig(Hess)
                ## Obtain the eigenpairs
                V_d = v[:, np.argsort(w)[:(len(x_pts)-d)]]
                Grad = np.sum(-wt*(x_pts - data) /(h**2) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0)
                Grad_skew = (h**2)*Grad
                ## Mean Shift vector
                ms_v = np.sum(wt*data*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) \
                       / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) - x_pts
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, Grad))
                SCMS_grad_skew = np.dot(V_d, np.dot(V_d.T, Grad_skew))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts
                ## Stopping criteria
                if stop_cri == 'pts_diff':
                    if LA.norm(SCMS_v) < eps:
                        conv_sign[i] = 1
                else: 
                    if LA.norm(SCMS_grad_skew) < eps:
                        conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The SCMS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t], conv_sign


def SCMS_Log_KDE(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, wt=None, bw_varied=False,
                 stop_cri='proj_grad'):
    '''
    Subspace Constrained Mean Shift algorithm with log density and Gaussian kernel
    
    Parameters:
        mesh_0: a (m,D)-array
            The coordinates of m initial points in the D-dim Euclidean space.
    
        data: a (n,D)-array
            The coordinates of n data sample points in the D-dim Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
       
        stop_cri: string ('proj_grad'/'pts_diff')
            The indicator of which stopping criteria that will be used to 
            terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors 
            between two consecutive iteration points need to be smaller than 
            'eps' for terminating the algorithm. When stop_cri='proj_grad' or 
            others, the projected/principal gradient of the current point need to be 
            smaller than 'eps' for terminating the algorithm.)
            (Default: stop_cri='proj_grad'.)
    
    Return:
        SCMS_path: (m,D,T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Dimension of data points
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    if wt is None:
        wt = np.ones((n,1))
    else:
        wt = wt.reshape(n,1)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                ## Compute the gradient of the log density
                Grad = np.sum(-wt*(x_pts - data) /(h**2) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0)
                Log_grad = Grad / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2))
                Log_grad_skew = (h**2)*Log_grad
                ## Compute the Hessian matrix
                Log_Hess = np.dot((x_pts-data).T, wt*(x_pts-data) \
                                  * np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1)) \
                           /(h**4 * np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2))) \
                - np.diag(np.ones(len(x_pts,)) / (h**2)) - np.dot(Log_grad.reshape(D,1), Log_grad.reshape(1,D))
                ## Spectral decomposition
                w, v = LA.eig(Log_Hess)
                ## Obtain the eigenpairs
                V_d = v[:, np.argsort(w)[:(len(x_pts)-d)]]
                ## Mean Shift vector
                ms_v = np.sum(wt*data*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) \
                       / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) - x_pts
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, Log_grad))
                SCMS_grad_skew = np.dot(V_d, np.dot(V_d.T, Log_grad_skew))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts
                ## Stopping criteria
                if stop_cri == 'pts_diff':
                    if LA.norm(SCMS_v) < eps:
                        conv_sign[i] = 1
                else: 
                    if LA.norm(SCMS_grad_skew) < eps:
                        conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The SCMS algorithm reaches the maximum number of iterations,'\
              +str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t], conv_sign


def SCMS_KDETest(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, wt=None,
                 bw_varied=False, sF=1, stop_cri='proj_grad'):
    '''
    Subspace Constrained Mean Shift Algorithm with Gaussian kernel.
    
    Parameters:
        mesh_0: a (m,D)-array
            The coordinates of m initial points in the D-dim Euclidean space.
    
        data: a (n,D)-array
            The coordinates of n data sample points in the D-dim Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
            
        bw_varied: boolean
            The indicator of whether bandwidth should be varied for each 
            coordinate. If True, the argument "h" should be None or (d,)-array.
            (Default: False.)
            
        sF: float
            The scaled factor for the mean shift vector. (Default: sF=1.)
       
        stop_cri: string ('proj_grad'/'pts_diff')
            The indicator of which stopping criteria that will be used to 
            terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors 
            between two consecutive iteration points need to be smaller than 
            'eps' for terminating the algorithm. When stop_cri='proj_grad' or 
            others, the projected/principal gradient of the current point need 
            to be smaller than 'eps' for terminating the algorithm.)
            (Default: stop_cri='proj_grad'.)
            
    Return:
        SCMS_path: (m,D,T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Dimension of data points
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    if wt is None:
        wt = np.ones((n,1))
    else:
        wt = wt.reshape(n,1)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                ## Compute the Hessian matrix
                Hess = np.dot(((x_pts-data)/h**2).T, wt*(x_pts-data)/(h**2) \
                           *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1)) \
                      - np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) \
                                 * np.diag(np.ones(len(x_pts),)/h**2)
                ## Spectral decomposition
                w, v = LA.eig(Hess)
                ## Obtain the eigenpairs
                V_d = v[:, np.argsort(w)[:(len(x_pts)-d)]]
                Grad = np.sum(-wt*((x_pts - data)/h**2) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0)
                ## Mean Shift vector
                ms_v = (np.sum(wt*data*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) \
                       / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) - x_pts)/(h**2)
                ## Subspace constrained gradient and mean shift vector
                SCMS_grad = np.dot(V_d, np.dot(V_d.T, Grad))
                SCMS_v = np.dot(V_d, np.dot(V_d.T, sF*ms_v))
                ## SCMS update
                x_new = SCMS_v + x_pts
                ## Stopping criteria
                if stop_cri == 'pts_diff':
                    if LA.norm(SCMS_v) < eps:
                        conv_sign[i] = 1
                else: 
                    if LA.norm(SCMS_grad) < eps:
                        conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The SCMS algorithm reaches the maximum number of iterations,'\
               +str(max_iter)+', and has not yet converged.')
    return SCMS_path[:,:,:t], conv_sign


def SCMS_Log_KDETest(mesh_0, data, d=1, h=None, eps=1e-7, max_iter=1000, wt=None,
                     bw_varied=False, sF=1, stop_cri='proj_grad'):
    '''
    Subspace Constrained Mean Shift algorithm with log density and Gaussian kernel.
    
    Parameters:
        mesh_0: a (m,D)-array
            The coordinates of m initial points in the D-dim Euclidean space.
    
        data: a (n,D)-array
            The coordinates of n data sample points in the D-dim Euclidean space.
       
        d: int
            The order of the density ridge. (Default: d=1.)
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
       
        eps: float
            The precision parameter. (Default: eps=1e-7.)
       
        max_iter: int
            The maximum number of iterations for the SCMS algorithm on each 
            initial point. (Default: max_iter=1000)
            
        wt: (n,)-array
            The weights of kernel density contributions for n random sample 
            points. (Default: wt=None, that is, each data point has an equal
            weight "1/n".)
            
        bw_varied: boolean
            The indicator of whether bandwidth should be varied for each 
            coordinate. If True, the argument "h" should be None or (d,)-array.
            (Default: False.)
            
        sF: float
            The scaled factor for the mean shift vector. (Default: sF=1.)
       
        stop_cri: string ('proj_grad'/'pts_diff')
            The indicator of which stopping criteria that will be used to 
            terminate the SCMS algorithm. (When stop_cri='pts_diff', the errors 
            between two consecutive iteration points need to be smaller than 
            'eps' for terminating the algorithm. When stop_cri='proj_grad' or 
            others, the projected/principal gradient of the current point need to be 
            smaller than 'eps' for terminating the algorithm.)
            (Default: stop_cri='proj_grad'.)
    
    Return:
        SCMS_path: (m,D,T)-array
            The entire iterative SCMS sequence for each initial point.
    '''
    
    n = data.shape[0]  ## Number of data points
    D = data.shape[1]  ## Dimension of data points
    if (h is None) and (bw_varied is True):
        # Apply Silverman's rule of thumb to select the bandwidth parameter in 
        # each coordinate 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.std(data, axis=0)
    elif h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        h = (4/(D+2))**(1/(D+4))*(n**(-1/(D+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth is "+ str(h) + ".\n")
    
    SCMS_path = np.zeros((mesh_0.shape[0], D, max_iter))
    SCMS_path[:,:,0] = mesh_0
    
    ## Create a vector indicating the convergent status of every mesh point
    conv_sign = np.zeros((mesh_0.shape[0], ))  
    if wt is None:
        wt = np.ones((n,1))
    else:
        wt = wt.reshape(n,1)
    for t in range(1, max_iter):
        if all(conv_sign == 1):
            print('The SCMS algorithm converges in ' + str(t-1) + 'steps!')
            break
        for i in range(mesh_0.shape[0]):
            if conv_sign[i] == 0:
                x_pts = SCMS_path[i,:,t-1]
                ## Compute the gradient of the log density
                Grad = np.sum(-wt*((x_pts - data)/h**2) \
                              *np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0)
                Log_grad = Grad / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2))
                ## Compute the Hessian matrix
                Log_Hess = np.dot(((x_pts-data)/h**2).T, wt*((x_pts-data)/h**2) \
                                  * np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1)) \
                           / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) \
                         - np.diag(np.ones(len(x_pts),) / (h**2)) - np.dot(Log_grad.reshape(D,1), Log_grad.reshape(1,D))
                if (np.sum(np.isinf(Log_Hess)) > 0) or (np.sum(np.isnan(Log_Hess)) > 0):
                    # Set those points with infinite Hessians to NaN
                    nan_arr = np.zeros_like(x_pts)
                    nan_arr[:] = np.nan
                    conv_sign[i] = 1
                    x_new = nan_arr
                else:
                    ## Spectral decomposition
                    w, v = LA.eig(Log_Hess)
                    ## Obtain the eigenpairs
                    V_d = v[:, np.argsort(w)[:(len(x_pts)-d)]]
                    ## Mean Shift vector
                    ms_v = (np.sum(wt*data*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2).reshape(n,-1), axis=0) \
                           / np.sum(wt.reshape(n,)*np.exp(np.sum(-((x_pts - data)/h)**2, axis=1)/2)) - x_pts)/h**2
                    ## Subspace constrained gradient and mean shift vector
                    SCMS_grad = np.dot(V_d, np.dot(V_d.T, Log_grad))
                    SCMS_v = np.dot(V_d, np.dot(V_d.T, sF*ms_v))
                    ## SCMS update
                    x_new = SCMS_v + x_pts
                    ## Stopping criteria
                    if stop_cri == 'pts_diff':
                        if LA.norm(SCMS_v) < eps:
                            conv_sign[i] = 1
                    else: 
                        if LA.norm(SCMS_grad) < eps:
                            conv_sign[i] = 1
                SCMS_path[i,:,t] = x_new
            else:
                SCMS_path[i,:,t] = SCMS_path[i,:,t-1]
        # print(t)
    
    if t >= max_iter-1:
        print('The SCMS algorithm reaches the maximum number of iterations,'\
              +str(max_iter)+', and has not yet converged.')
    nan_cri = np.isnan(SCMS_path[:,0,t-1])
    SCMS_path = SCMS_path[~nan_cri,:,:t]
    conv_sign = conv_sign[~nan_cri]
    return SCMS_path[:,:,:t], conv_sign