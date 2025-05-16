#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:31:12 2024

@author: andrey
"""


import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pickle as pkl
from copy import deepcopy
#global variable: path to ForceSMIP folder

def moving_average(ind,window,axis=-2,mode='same'):
    '''

    Parameters
    ----------
    ind : np.array
        Any shape. Moving average is performed along axis (time)
    window : int

    Returns
    -------
    res : np.array of the same shape

    '''
    
    ind=np.swapaxes(ind,0,axis)
    init_shape=list(ind.shape)
    ind=np.reshape(ind,(init_shape[0],-1))
    import scipy.signal
    res=scipy.signal.convolve2d(ind,np.ones((window,1)),mode=mode)/scipy.signal.convolve2d(np.ones(ind.shape),np.ones((window,1)),mode=mode)
    res=np.reshape(res,tuple([res.shape[0]]+init_shape[1:]))
    res=np.swapaxes(res,0,axis)
    return res

def linear_fit(Y,X):

    A = np.vstack([X.T, np.ones(len(X))]).T
    coef = np.linalg.lstsq(A, Y,rcond=None)[0]

    m,n = coef.shape
    
    a = coef[0:m-1][:].T
    b = coef[m-1:m][:]

    #rms = (Y-np.dot(A,coef)).std(axis=0)
    rms=np.sqrt(((Y-np.dot(A,coef))**2).mean(axis=0))
    return a,b,rms

def linear_run(X,a,b):
    Y=np.dot(X,a.T)
    for i in range(Y.shape[0]):
        Y[i,:]=Y[i,:]+b
    return Y

class smooth_linear_regressor:
    
    def __init__(self):
        return
    
    def fit(self,data,forced_signals,windows=None):
        """
        Fits the smoothed linear regression of forced_signals on data (separately for each j-th component of forced_signals): 
        
            forced_signals [...,j]=np.matmul(moving_average(data,opt_window[j]), a[j])+b[j]+residual[j]
        
            where forced_signals has shape [M,N,df], data has shape [M,N,K], a[j] has shape [K], opt_window[j] and b[j] are scalars.
            
            moving_average(data,opt_window[j]) is a box-car running mean of the data along the time axis=-2 with the averaging window opt_window[j].
            
            opt_window[j] is a window from the set of 'windows', which minimizes the variance of the residual[j]
        
        After this function the class stores the fitted regression parameters and windows.
        
        
        This procedure is called "smoothed linear regression" (SLR).
        
        Parameters
        ----------
        data : np.array [M,N,K]
            the K-dimensional input for the regression model. N is the time duration, M is the number of realizations.
        forced_signals : np.array [M,N,df]
            the df-dimensional output for the regression model
        windows : list of ints, optional
            List of window sizes to choose from. The default is equivalent to windows=[1].

        Returns
        -------
        None.

        """
        
        
        ns1,nt1,d=data.shape
        ns,nt,df=forced_signals.shape
        if ns1!=ns or nt1!=nt:
            print('wrong shapes of data and forced signals')
            exit()
        if windows is None: windows=[1]
        A,B,RMS=[],[],[]
        for window in windows:
            data_sm=moving_average(data,window,axis=-2)
            x=data_sm.reshape((ns*nt,-1))  
            y=forced_signals.reshape((ns*nt,-1))
            a,b,rms=linear_fit(y,x)
            A.append(a)
            B.append(b)
            RMS.append(rms)
        
        opt=np.argmin(np.stack(RMS,axis=0),axis=0)
        
        self.a=[]
        self.b=[]
        self.rms=[]
        self.opt_windows=np.array(windows)[opt]
        for j in range(df):
            self.a.append(A[opt[j]][j,:])
            self.b.append(B[opt[j]][:,j])
            self.rms.append(RMS[opt[j]][j])
        self.df=df
        
    def run(self,new_data):
        """
        Based on the fitted SLR parameters, produces the forced response independently for each of realizations stored in new_data.

        Parameters
        ----------
        new_data : np.array [M,N,K]
            Ensemble of M realizations of length N and dimension K.

        Returns
        -------
        forced_signal : np.array [M,N,df]
            Forced response computed by the SLR model for each of the input realizations.

        """
        ns,nt,d=new_data.shape
        forced_signal=[]
        for j in range(self.df):
            data_sm=moving_average(new_data,self.opt_windows[j],axis=-2)
            x=data_sm.reshape((ns*nt,-1))
            y=linear_run(x,self.a[j][None,:],self.b[j][:,None])
            forced_signal.append(y.reshape((ns,nt)))
        forced_signal=np.stack(forced_signal,axis=-1)
        return forced_signal
    
    def save(self):
        '''

        Returns
        -------
        pars : dictionary with the regressor fields

        '''
        pars={}
        for key in self.__dict__.keys():
            pars[key]=deepcopy(self.__dict__[key])
        return pars
    
    def load(self,pars):
        '''
        Loads the regressor fields from the saved dictionary.

        Parameters
        ----------
        pars : dictionary
            The fields and values to load.

        Returns
        -------
        None.

        '''
        for key in pars.keys():
            self.__dict__[key]=deepcopy(pars[key])        
   
        

    
