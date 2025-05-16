#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The program implements LDM-SLR method to estimate forced signal response in a single spatio-temporal 
climate realization after training on large ensembles produced by Earth System Models.
As a particular example, the main section at the bottom of this file can be used to estimate forced 
response in the ForceSMIP evaluation data, if provided with the right input (see the comments there). 

LDM-step is implemented via the functions 'train_ldm_with_optimization', 'train_ldm_without_optimization' and 'analyse_ldm' which can be used to perform LDM decomposition independently.
See the comments and docstrings there.

SLR-step is implemented via the class smooth_linear_regressor ant its methods 'fit', 'run', 'save', 'load' and can be used independently.

LDM-SLR training step is implemented via the function 'train_ldm_slr'

LDM-SLR-based estimation of forced response from new data is implemented via the function 'process_new_realization'


Commands to install environment in conda:
    
conda create --name my_env python=3.10 numpy matplotlib
conda activate my_env
pip install netcdf4 scipy scikit-learn

'''




### For linux environments: disable parallelism in numpy operations (before importing anything else)
import os
os.environ["OMP_NUM_THREADS"] = "1"

### Import from standard libraries
import numpy as np
import pickle as pkl
from subprocess import Popen
import sys

### Import from the sub-modules
from preprocessing_module import preprocessing, preprocess_new_data, postprocess_pcs, write_data
from preprocessing_module import get_evaluation_metadata, get_data
from ldm_optimizer import ldm_optimizer
from ldm_model import ldm_model
from smooth_linear_regressor import smooth_linear_regressor


#############################################################################################
# LDM-SLR TRAINING FUNCTIONS ################################################################
#############################################################################################

def train_ldm_slr(training_data_file,hyperparameters,optimization_flag,output_ldm_slr_parameters_file):
    '''
    This function is the example of how to train LDM-SLR method with predefined discrete hyperparameters and (optionally) optimize continuous LDM-hyperparameters.
    The discrete parameter opt_windows in SLR (set of 'df' windows used to smooth the data before the regression in SLR method) is always re-optimized because it depends on the actual forced LDM modes obtained. Therefore, the list 'slr_windows' is used as a hyperparameter to define the window values to choose from.
    
    The function takes multi-model ensemble data from training_data_file and saves the trained parameters into output_ldm_slr_parameters_file.

    Parameters
    ----------
    training_data_file : str
        The name of the file with training data in pickle format, which contains a tuple of ensemble EOF time series P with the shape [M,N,D] and array of model IDs of the shape [M].
        See more details about these arguments in train_ldm_with_optimization and train_ldm_without_optimization functions.
    hyperparameters: dictionary
        Must contain the keys:
            'dp' : int>=0, number of internal LDM modes to look for.
            'df' : int>=0, number of forced LDM modes to look for.
            'K' : int>0, number of ensemble EOF time series to fit LDM model into; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.
            'slr_windows': non-empty list of ints, a set of smoothing window sizes to choose from in SLR method
        If optimization_flag is False, must also contain:
            'tau_p': np.array with the shape [dp]
            'tau_f': np.array with the shape [df]
            'sigma_p': np.array with the shape [dp]
            'sigma_f': np.array with the shape [df]
            'sigma_c': float
            These continuous hyperparameters are the prior time scales and standard deviations of forced/internal LDM modes, and the prior deviation of the parameter 'c' in LDM decomposition. 
    optimization_flag : bool
        If True, then the continuous hyperparameters 'tau_p','tau_f','sigma_p', 'sigma_f', 'sigma_c' will be optimized.
        If False, then the continuous hyperparameters 'tau_p','tau_f','sigma_p', 'sigma_f', 'sigma_c' will not be optimized and must be defined in 'hyperparameters' dictionary.
    output_ldm_slr_parameters_file : str
        Name of the file which is used to save trained LDM-SLR parameters.

    Returns
    -------
    None.

    '''

    ### Reading multi-model ensemble EOF time series and model identifiers
    with open(training_data_file,'rb') as f: P, model_ids=pkl.load(f)  


    ### Training LDMs in either of the modes (with/without optimization of continuous hyperparameters). 
    print('LDM Training started')
    sys.stdout.flush()
    ### Taking K leading ensemble PCs to fit LDM model into
    y_train=P[...,:hyperparameters['K']] 
    if optimization_flag is False:
        ### THE DEFAULT BRANCH.
        ### Below is the core LDM training function (without optimization) which may be used independently in various tasks.
        ### The parameter final_eps defines the final accuracy of the cost function minimization (in some internal dimensionless units which are roughly relative to 1.).
        ### There are also other adjustable parameters. See the DOCSTRING for further details.        
        model_parameters=train_ldm_without_optimization(y_train,hyperparameters,model_ids=model_ids,final_eps=1e-12)   
    else:        
        ### Name of temporary file to backup the optimizer state when training with optimization
        tmp_file_name=output_ldm_slr_parameters_file+'.optimization_backup'     
        ### Below is the core LDM training function (with optimization) which may be used independently in various tasks.
        ### The parameter final_eps defines the final accuracy of the cost function minimization (in some internal dimensionless units which are roughly relative to 1.).
        ### There are also other adjustable optimization parameters. See the DOCSTRING for further details.
        model_parameters=train_ldm_with_optimization(y_train,hyperparameters,tmp_file_name,model_ids=model_ids,final_eps=1e-12)


    ### Saving optimal trained LDM parameters (optional backup) 
    with open(output_ldm_slr_parameters_file+'.optimal_ldm_parameters','wb') as f:
        pkl.dump(model_parameters,f)
    
    ### Postprocessing the trained LDM to extract forced signal components with the shape [M,N,df] 
    ### and the parameters of their mapping to the initial phase space: patterns B and c with shapes [K,df] and [K]   
    ### This function also demonstrates the basic example of LDM analysis
    B, c, forcing = analyse_ldm(model_parameters)


    ### Training SLR method
    print('SLR training started')
    sys.stdout.flush()    
    ### Creating the SLR object
    SLR=smooth_linear_regressor()
    ### Fitting SLR mapping from K_slr ensemble EOF time series to their correspopnding LDM forced components (including optimization along 'windows')
    SLR.fit(P[...,:hyperparameters['K_slr']],forcing,hyperparameters['slr_windows'])
    ### Getting the dictionary with the trained SLR_parameters
    SLR_parameters=SLR.save()
    ### Printing 'df' optimal window sizes found by SLR method
    print(SLR.opt_windows)

    
    ### Saving final LDM-SLR parameters needed for the evaluation step
    with open(output_ldm_slr_parameters_file,'wb') as f: 
        pkl.dump((B,c,SLR_parameters),f)  
    print('Training stage finished')

      
def train_ldm_with_optimization(y_train,hyperparameters,tmp_file_name,model_ids=None,nsteps=10000,eps=1e-4,exact_eps=1e-6,final_eps=1e-12,custom_optimizer_config={}):
    '''
    For a given discrete hyperparameters of LDM method (dp and df), performs optimization of continuous hyperparameters.
    The cost function to be maximized during the optimization is Bayesian evidence.
    
    The optimizer is implemented as a sampler which follows several cycles of some annealing scenario, each  ending with a special perturbation in a 'good' direction.
    Along with basic configuration parameters, it contains a number of further configuration parameters tunable via custom_optimizer_config, and even more can be tuned via optimizer.init() arguments.
    Adjusting these parameters can significantly increase the convergence speed which can be tracked by monitoring optimizer state, e.g. via optimizer.plot_status().
    The default configuration parameters were found optimal and stable for the exemplary ForceSMIP data, but note that the result is still (numerically) random because of random perturbations in the sampler.      

    The resulting LDM model is the best after 'nsteps' steps of the sampler. It is possible to continue the sampling by loading optimizer from 'tmp_file_name' file and increasing nsteps (look at the code for details but note that this file is deleted at the end of the function).

    A good approach is to run the sampler several times (e.g. in parallel) for different 'dp' and 'df', and choose the most optimal result from them (according to the Bayesian evidence optimality, see 'analyse_ldm' function).

    
    Generally, this decomposition is a multi-model ensemble LDM method which finds 'dp' internal modes and 'df' forced modes (see the hyperparameters). If model_ids=None, it becomes single-model LDM method.
    If M=1 (see the shape of y_train), it becomes simple LDM decomposition (forced and internal modes are equivalent).
    The particular cases dp=0 (forced modes only) and df=0 (internal modes only) are possible.


    Parameters
    ----------
    y_train : np.array with the shape [M,N,K]
        Pre-computed ensemble EOF time series with M ensemble members of the duration N and dimension K
        The particular case of M=1 corresponds to simple LDMs (when forced and internal LDMs will be equivalent)
    hyperparameters: dictionary
        Must contain the keys:
            'dp' : int>=0, number of internal LDM modes to look for.
            
            'df' : int>=0, number of forced LDM modes to look for.
            
    tmp_file_name : str
        Name of the file to back up the optimizer state. If this file exists, optimizer is initialized from it. This file is deleted at the end of the function.
    model_ids : None or np.array with the shape [M], optional
        If not None, must contain integers from 0 to nmodels-1, defining the models which the realizations are drawn from. 
        If None, it is equivalent to np.zeros((M)), i.e. the case of single-model ensemble of M realizations.
        The default is None.
    nsteps : int>0, optional
        Total number of optimizer steps. The default is 10000.
    eps : float>0, optional
        Parameter of accuracy (internal stop criterion for one minimization). It is used at every sampler step. The larger this parameter is, the faster the optimizer steps are, and the lower the accuracy of minimization is. The default is 1e-4.
    exact_eps : float>0, optional
        The same as eps, but it is used by the optimizer before it updates the 'optimal' LDM solution. Should be small enough to ensure that the saved solutions are not too biased. The default is 1e-6.
    final_eps : TYPE, optional
        The same as eps, but applied only once after the optimizer has finished its steps. Performs final bias correction which may appear due to (still) low accuracy of exact_eps. It makes one minimization really long but accurate.The default is 1e-12.
    custom_optimizer_config : dictionary, optional
        Additional configuration parameters, such as:
            'coeffstart': perturbation amplitude scale at the start of annealing cycle (default 0.05).
            
            'coeffend': perturbation amplitude scale at the end of annealing cycle (default 0.01).
            
            'nmax': number of sampler steps per annealing cycle (annealing time, default 1000).
            
            'threshold': internal threshold. The mode with the contribution below this threshold has a probability to be 'perturbed' at the end of the annealing cycle. Default is 1e-4.


    Returns
    -------
    model_parameters : dictionary containing the result of save() method of the ldm_model class after training.
        Can be used to initialize ldm_model class via its load() method.

    '''

    optimizer_config={'coeffstart':0.05,'coeffend':0.01,'nmax':1000,'threshold':1e-4,
                'taumin':1.,'taumax':y_train.shape[-2],
                'sigmapmin':0.005,'sigmapmax':0.05,'sigmafmin':0.05,'sigmafmax':0.5}
    optimizer_config.update(custom_optimizer_config)  
        
    import os
    optimizer=ldm_optimizer()
    if os.path.isfile(tmp_file_name):
        optimizer.load_from_file(tmp_file_name)
    else:     
        optimizer.init(y_train,dp=hyperparameters['dp'],df=hyperparameters['df'],model_ids=model_ids,
                       optimizer_config=optimizer_config,train_config={'eps':eps,'exact_eps':exact_eps})

    for i in range(nsteps-len(optimizer.history)+1): 
        if i>0 and i%25==0:   
            optimizer.save_to_file(tmp_file_name)
        if i>0 and i%25==0:
            optimizer.print_status()
        
        optimizer.iterate()
    
    optimizer.print_status()
    optimizer.save_to_file(tmp_file_name)
    
    print('Final training')
    model=optimizer.final_training(final_eps=1e-12)
    model_parameters=model.save()
    
    Popen('rm %s'%tmp_file_name,shell=True)
    return model_parameters
        

def train_ldm_without_optimization(y_train,hyperparameters,model_ids=None,n_trials=1,final_eps=1e-12):
    '''
    For a given hyperparameters, performs n_trials of LDM training attempts from random initial conditions.
    The resulting LDM model is the one with the best Bayesian evidence from these attempts.
    Note that this result is (numerically) random because of random initial conditions (and possible but rare non-uniqueness of a minimum), while increasing n_trials ensures its stability.
    
    This is a multi-model ensemble LDM method which finds 'dp' internal modes and 'df' forced modes (see the hyperparameters). If model_ids=None, it becomes single-model LDM method.
    If M=1 (see the shape of y_train), it becomes simple LDM decomposition (forced and internal modes are equivalent).
    The particular cases dp=0 (forced modes only) and df=0 (internal modes only) are possible.

    This method is usually recommended AFTER finding optimial hyperparameters.

    Parameters
    ----------
    y_train : np.array with the shape [M,N,K]
        Pre-computed ensemble EOF time series with M ensemble members of the duration N and dimension K
        The particular case of M=1 corresponds to simple LDMs (when forced and internal LDMs will be equivalent)
    hyperparameters: dictionary
        Must contain the keys:
            'dp' : int>=0, number of internal LDM modes to look for.
            'df' : int>=0, number of forced LDM modes to look for.
        Also:    
            'tau_p': np.array with the shape [dp]
            'tau_f': np.array with the shape [df]
            'sigma_p': np.array with the shape [dp]
            'sigma_f': np.array with the shape [df]
            'sigma_c': float
            These continuous hyperparameters are the prior time scales and standard deviations of forced/internal LDM modes, and the prior deviation of the parameter 'c' in LDM decomposition. 

    model_ids : None or np.array with the shape [M], optional
        If not None, must contain integers from 0 to nmodels-1, defining the models which the realizations are drawn from. 
        If None, it is equivalent to np.zeros((M)), i.e. the case of single-model ensemble of M realizations.
        The default is None.
        
    n_trials : int>0, optional
        Number of minimization attempts starting from random initial conditions. The default is 1.
    final_eps : float>0, optional
        Parameter of accuracy (internal stop criterion for one minimization). Can be played with, but must be small, otherwise the minimum of the cost function is not guaranteed, and the Bayesian evidence (optimality estimation, see the analyse_ldm function) will be biased. 
        The default is 1e-12.

    Returns
    -------
    model_parameters : dictionary containing the result of save() method of the ldm_model class after training.
        Can be used to initialize ldm_model class via its load() method.

    '''
    
    par=hyperparameters
    opt_log_evidence=None
    for i in range(n_trials):
        
        #initialization of ldm model
        model=ldm_model()
        model.init(y_train,dp=par['dp'],df=par['df'],
                            tau_p=par['tau_p'],tau_f=par['tau_f'],
                            sigma_p=par['sigma_p'],sigma_f=par['sigma_f'],
                            sigma_c=par['sigma_c'],model_ids=model_ids)
        
        #generating random initial conditions based on prior PDF parameters
        model.gen_pars()
        
        #training with (presumably) high accuracy
        model.fit_pars(eps=final_eps)  
        log_evidence=-model.minus_log_evidence_nn_pars()
        print('%d/%d: log_evidence=%f'%(i+1,n_trials,log_evidence))
        
        if (opt_log_evidence is None) or (log_evidence>opt_log_evidence):
            opt_log_evidence=log_evidence
            model_parameters=model.save()
    return model_parameters    


def analyse_ldm(model_parameters):
    '''
    This function is an example of analysis of the trained LDMs. Particularly, it reconstructs the LDM from its parameters and then returns the patterns allowing to reconstruct the forced part of LDM, as well as the forced component itself. 

    Parameters
    ----------
    model_parameters : dictionary  with the LDM parameters obtained by ldm_model.save() function

    Returns
    -------
    B : np.array [K,df]
        Patterns of the LDM forced components.
    c : np.array[K]
        Additive LDM pattern to form LDM forced response (see below).
    forcing : np.array [M,N,df]
        LDM forced components for each of M realizations.
        Can be demapped to form an LDM-based forced response in EOF-space via: c+np.matmul(forcing,B.T)

    '''
    
    ### Example of LDM model loading from dictionary
    model=ldm_model()
    model.load(model_parameters)
    
    ### Calling LDM patterns A [K,dp], B [K,df] and c[K]
    A,B,c=model.call_A(),model.call_B(),model.call_c()
    ### Calling internal LDM mode time series p [M,N,dp]
    p=model.call_p()
    ### Example of calling forced LDM mode time series forcing_compact[nmodels,N,df]
    forcing_compact=model.call_f()
    ### Calling extended forced LDM mode time series forcing[M,N,df]
    forcing=model.call_f_extended()
    ### Full LDM decomposition is written as c+np.matmul(p,A.T)+np.matmul(forcing,B.T)
    
    ### Example of Bayesian Evidence estimation (rescaled)
    ### Can be used to pick optimal LDM after many parallel optimizations
    log_evidence=-model.minus_log_evidence_nn_pars()
    print('log_evidence=',log_evidence)
    
    ### Including numerical mean of p into the additive forced response constant (for the case it's not exactly zero)
    pmean=p.mean(axis=0).mean(axis=0)
    c=c+np.matmul(pmean[None,:],A.T)[0,:]
    
    ### Now the way to get the LDM forced part in the EOF-space from the LDM components F[...,df] is given by:
    ### c+np.matmul(F,B.T)
    ### This is used when new data is processed
    
    return B, c, forcing


def get_optimal_hyperparameters(variable):
    '''
    Custom function which returns the dictionary with optimal hyperparameters for a given variable from exemplary ForceSMIP data.
    These hyperparameters are estimated by two-step procedure.
    
    First, 10 realizations were taken from each model, and the function 'train_ldm_with_optimization' was launched with different discrete hyperparameters (dp,df) to train LDMs on these realizations. 
    Then the discrete hyperparameters corresponding to the LDM with the best Bayesian evidence were chosen (they were additionally validated on the rest of the training data).
    The hyperparameter K was fixed to be 20 there. Then the hyperparameter K_slr was chosen by training SLR with K_slr=20,50,100,200 on the same training data and validation of the forced response reconstruction on the rest of the training data. 
    The resulting discrete hyperparameters were recorded as optimal.
    
    Second, 25 realizations were taken from each model, and the function 'train_ldm_slr' with optimization of continuous hyperparameters was launched for discrete hyperparameters found in the previous step.
    The resulting continuous hyperparameters were recorded as optimal.
    
    The list 'slr_windows' in SLR method was always kept the same.

    Parameters
    ----------
    variable : str, name of the variable

    Returns
    -------
    hyperparameters : dictionary with optimal hyperparameters pre-computed as described

    '''
      
    if variable=='pr': 
        dp,df,K,K_slr=3,5,20,100
        continuous_hyperparameters={'tau_p': np.array([78.86750689, 23.40077718, 13.45580892]), 
                                    'tau_f': np.array([  54.02300796,   96.65012646,  265.25637618, 2123.90979324, 1580.55626491]), 
                                    'sigma_p': np.array([0.00758817, 0.00771449, 0.00573033]), 
                                    'sigma_f': np.array([0.02719668, 0.00951979, 0.00481991, 0.03212232, 0.03008038]), 
                                    'sigma_c': 4.27720047756149e-06}        
    if variable=='monmaxpr': 
        dp,df,K,K_slr=5,4,20,50
        continuous_hyperparameters={'tau_p': np.array([ 1.        ,  1.        ,  1.        ,  9.71050092, 28.8593592 ]), 
                                    'tau_f': np.array([ 898.56620654, 3490.56469423, 1911.7760996 ,  241.3264509 ]), 
                                    'sigma_p': np.array([1.        , 1.        , 1.        , 0.0058705 , 0.00783939]), 
                                    'sigma_f': np.array([0.03238206, 0.19422863, 0.03832878, 0.02056916]), 
                                    'sigma_c': 0.10473050878040038}
    if variable=='psl': 
        dp,df,K,K_slr=4,3,20,200
        continuous_hyperparameters={'tau_p': np.array([34.71930666,  3.20615066,  0.85864459, 15.08920992]), 
                                    'tau_f': np.array([5542.70317735, 4582.50193257, 1266.84078371]), 
                                    'sigma_p': np.array([0.00747636, 0.00057048, 0.00535743, 0.00405033]), 
                                    'sigma_f': np.array([0.01812209, 0.04940789, 0.07381424]), 
                                    'sigma_c': 0.0003403715373315843}
    if variable=='tas': 
        dp,df,K,K_slr=5,5,20,200
        continuous_hyperparameters={'tau_p': np.array([20.49634917, 51.00575207, 14.11641804, 29.07817735, 30.37428997]), 
                                    'tau_f': np.array([3056.81101608, 1410.75438183,  432.52568972,  979.42778527, 1139.38762123]), 
                                    'sigma_p': np.array([0.00659515, 0.00856813, 0.00474966, 0.00715576, 0.00760228]), 
                                    'sigma_f': np.array([0.16904323, 0.06339107, 0.12309428, 0.05154127, 0.04172203]), 
                                    'sigma_c': 2.5052617640483014e-06}
    if variable=='monmintasmin': 
        dp,df,K,K_slr=5,5,20,200
        continuous_hyperparameters={'tau_p': np.array([38.78360384, 24.34286752, 26.32973806, 32.78918155, 24.52596424]), 
                                    'tau_f': np.array([1657.9600042 , 5109.23906213,  720.64607716, 1724.9826587 , 1033.74286751]), 
                                    'sigma_p': np.array([0.00697417, 0.00684556, 0.00618176, 0.00802805, 0.00602842]), 
                                    'sigma_f': np.array([0.05479447, 0.10543785, 0.16934647, 0.10279562, 0.0419265 ]), 
                                    'sigma_c': 0.0028197119452521634}
    if variable=='monmaxtasmax': 
        dp,df,K,K_slr=5,5,20,100
        continuous_hyperparameters={'tau_p': np.array([37.08125328, 18.43999067, 50.27729494, 14.229307  , 31.82070608]), 
                                    'tau_f': np.array([1105.78515613,  800.12979081, 7656.98686479,  724.87163457, 1172.79902035]), 
                                    'sigma_p': np.array([0.00735013, 0.0058449 , 0.00828128, 0.00628506, 0.0081724 ]), 
                                    'sigma_f': np.array([0.03589381, 0.06601621, 0.09787999, 0.05902746, 0.1106272 ]), 
                                    'sigma_c': 0.16443894467882797}
    if variable=='tos': 
        dp,df,K,K_slr=5,5,20,200
        continuous_hyperparameters={'tau_p': np.array([27.83084476, 19.46239711, 12.67344636, 71.48962834, 25.68152246]), 
                                    'tau_f': np.array([1842.36960056, 1578.77681959,  739.49116517, 1819.32732377, 604.19684967]), 
                                    'sigma_p': np.array([0.0076847 , 0.00668544, 0.00648194, 0.01001164, 0.00733076]), 
                                    'sigma_f': np.array([0.41247711, 0.14209036, 0.07980106, 0.13615338, 0.05909242]), 
                                    'sigma_c': 0.0014793599996061}
    if variable=='zmta': 
        dp,df,K,K_slr=7,8,20,200
        continuous_hyperparameters={'tau_p': np.array([4.78633409,   0.18336506,   2.01845695,  39.34223859, 3.78759221,  26.79670108, 113.3898944 ]), 
                                    'tau_f': np.array([2512.01164558,  541.18355854, 1209.84123862, 7238.02826133, 391.2133156 , 1487.22914162, 3452.99573648,  856.86746778]), 
                                    'sigma_p': np.array([4.02011571e-03, 2.14431902e-05, 4.42925466e-03, 7.39153643e-03, 2.89966363e-04, 2.43387738e-03, 2.08098702e-03]), 
                                    'sigma_f': np.array([0.06811461, 0.0253809 , 0.08905377, 0.48516015, 0.01813795, 0.03205983, 0.03861944, 0.26437183]), 
                                    'sigma_c': 0.01566529226254056}    
    hyperparameters={'dp':dp,'df':df,'K':K,'K_slr':K_slr,'slr_windows':list(range(1,64+1))+[128,256,512]}
    hyperparameters.update(continuous_hyperparameters)
    return hyperparameters


#############################################################################################
# EVALUATION DATA PROCESSING FUNCTIONS ######################################################
#############################################################################################

def process_evaluation_data(root_folder,variable,hyperparameters,training_parameters_file,ldm_slr_parameters_file):
    '''
    This function takes new (unseen) realizations from evaluation dataset and estimates forced response there using the traind LDM-SLR algorithm. 
    The results are written to the output .nc files of the format 'variable_MEMBERID_tier1_LDM-SLR.nc'

    Parameters
    ----------
    root_folder : str, Path to a folder with the datasets (e.g. root_folder='/home/user/some/path/ForceSMIP')

    variable : str, name of the variable as in the training step.

    hyperparameters: dictionary, the same as in traininig step
        Must contain the keys:
            'dp' : int>=0, number of internal LDM modes.
            'df' : int>=0, number of forced LDM modes.
            'K' : int>0, number of ensemble EOF time series to fit LDM model into; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.
    
    training_parameters_file : file with the training parameters saved at the preprocessing step.
        
    ldm_slr_parameters_file : file with the trained LDM-SLR parameters saved at the preprocessing step

    Returns
    -------
    None.

    '''
    #Reading the preprocessing parameters (normalization constants, EOF patterns etc.)
    with open(training_parameters_file,'rb') as f:
        training_parameters=pkl.load(f) 

    #Reading from trained ldm_slr_parameters_file (parameters of the trained LDM and SLR methods)
    with open(ldm_slr_parameters_file,'rb') as f:
        trained_ldm_slr_parameters=pkl.load(f) 
        
    ### Defining list of .nc files with new realizations (e.g. ForceSMIP evaluation dataset members) 
    files=get_evaluation_metadata(root_folder,variable)
    
    ### Reading evaluation metadata fron one of the files. This is optional.
    ### This is a custom reading function, see the docstring for more details.
    times,x_grid,y_grid,var_dict=get_data(root_folder+files[-1],variable,mode='evaluation_metadata')
    
    ### Loop over files with evaluation dataset members
    for file in files:
        ### Reading data from the corresponding .nc file
        ### new_data and new_sea_masks have shape [nt,nx,ny].
        new_data,new_sea_masks,MEMBERID=get_data(root_folder+file,variable,mode='evaluation_data')
        
        ### This is the core function to estimate forced response in one new realization. Note that it uses only new_data and new_sea_masks from evaluation data set. 
        ### See the docstring for details.
        new_forced_response=process_new_realization(new_data,new_sea_masks,variable,hyperparameters,
                                                    training_parameters,trained_ldm_slr_parameters)
        
        ### Defining the output .nc file and writing the forced response to it.
        file_name='%s_%s_%s_%s.nc'%(variable,MEMBERID,'tier1','LDM-SLR')
        ### This is a custom writing function. See the docstring for the details.
        write_data(file_name,variable,new_forced_response,new_sea_masks,MEMBERID,times,x_grid,y_grid,var_dict)
        

def process_new_realization(new_data,new_sea_masks,variable,hyperparameters,training_parameters,trained_ldm_slr_parameters):
    '''
    This function gets new data (one realization) and estimates the forced response using the trained LDM-SLR method

    Parameters
    ----------
    new_data : np.array [N,nx,ny]
        Spatio-temporal data (one realization) with the time duration N and spatial dimensions nx, ny.
    new_sea_masks : np.array [N,nx,ny]
        Boolean array with the True values masking the valid data.
    variable : str, name of the variable as in the training step.

    hyperparameters: dictionary, the same as in traininig step
        Must contain the keys:
            'dp' : int>=0, number of internal LDM modes.
            'df' : int>=0, number of forced LDM modes.
            'K' : int>0, number of ensemble EOF time series to fit LDM model into; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.

    training_parameters : tuple of the training_parameters saved at the preprocessing step.
    
    trained_ldm_slr_parameters : tuple of the trained LDM-SLR parameters saved at the training step.

    Returns
    -------
    new_forced_response : np.array [N,nx,ny]
        Estimated forced response of the same shape as new_data, with valid values according to new_sea_masks.

    '''
    
    ### Preprocessing of the new data using the preprocessing parameters 
    ### (annual cycle removal, normalization, EOF compression)
    ### Returned values are the new time series P with the shape [1,N,D] and the subtracted annual cycle with the shape [1,12,nx,ny]
    P,subtracted_annual_cycles=preprocess_new_data(new_data[None,...],new_sea_masks[None,...],
                                                   variable,training_parameters)   
    
    ### Unpacking LDM-SLR parameters and loading SLR parameters into a new regressor object
    B,c,SLR_parameters=trained_ldm_slr_parameters
    SLR=smooth_linear_regressor()
    SLR.load(SLR_parameters)
    
    ### Applying SLR to approximate LDM forced components from new time series P
    ### 'new_ldm_forced_components' has shape [1,N,df]
    new_ldm_forced_components=SLR.run(P[...,:hyperparameters['K_slr']])
    
    #Returning back to EOF-space by applying LDM model (forced part)
    ### 'new_eof_forced_components' has shape [1,N,K]
    new_eof_forced_components=c+np.matmul(new_ldm_forced_components,B.T)
    
    #Returning back to physical space: EOF back-rotation, de-normalization, adding the subtracted annual cycle
    new_forced_response=postprocess_pcs(new_eof_forced_components,subtracted_annual_cycles,variable,training_parameters) 
    new_forced_response=new_forced_response[0,...]
    return new_forced_response


    
    
    
if __name__=='__main__':
    ######################################################################################################
    # PRE-DEFINITIONS ####################################################################################
    ######################################################################################################    

    
    ### Reading command line arguments:
    ### 1. A folder with the datasets (e.g. root_folder='/home/user/some/path/ForceSMIP')
    ### 2. Target variable name (e.g. can be chosen from the list 'pr','monmaxpr','psl',tas',monmintasmin','monmaxtasmax','tos','zmta')    
    ### They are used for exemplary data input, output and preprocessing operations
    root_folder=sys.argv[1]   
    variable=sys.argv[2]
    
    
    ### Creating a directory for temporary files, and defining some file names
    tmp_dir=variable+'_tmp'
    Popen('mkdir %s'%tmp_dir,shell=True).communicate()
    ### The names of preprocessed data and parameters files to be created during the preprocessing step 
    training_data_file=tmp_dir+'/training_data.pkl3'
    training_parameters_file=tmp_dir+'/training_parameters.pkl3'
    ### The name of the file which will be created after LDM-SLR method training step
    ldm_slr_parameters_file=tmp_dir+'/ldm-slr-parameters.pkl3'


    ### Defining hyperparameters and configuring optimization for LDM-SLR method training step.
    
    ### There are discrete and continuous hyperparameters. Discrete hyperparameters must be predefined, 
    ### while the continuous hyperparameters can be (optionally) optimized. 
    ### By default, the precomputed optimal hyperparameters (both discrete and continuous) are loaded 
    ### for a given variable of exemplary ForceSMIP dataset.
    ### See the docstring of 'get_optimal_hyperparameters' function for the details of this optimization.
    if len(sys.argv)<=3:
        hyperparameters=get_optimal_hyperparameters(variable)
        ### Also, by default, the optimization of continuous hyperparameters is turned off (optimization_flag=False)
        ### (this makes the re-computation faster, leading to approximately 7 hours on one CPU core)
        optimization_flag=False
    ### Alternatively, the discrete hyperparameters can be defined manually, e.g. via command line args
    ### And the optimization of the continuous hyperparameters can be turned on.
    else:
        hyperparameters={'dp':int(sys.argv[3]),'df':int(sys.argv[4]),
                         'K':int(sys.argv[5]),'K_slr':int(sys.argv[6]),
                         'slr_windows':list(range(1,64+1))+[128,256,512]}
        optimization_flag=True




    ######################################################################################################
    # COMPUTATION STEPS ##################################################################################
    ######################################################################################################
    
    ### Preprocessing step
    
    ### Here we load the training data, perform preprocessing and create two files with the data and parameters
    ### Annual cycle removal, data normalization and ensemble-EOF rotation is here. See the DOCSTRING and the code for details.
    preprocessing(root_folder,variable,output_training_data_file=training_data_file,output_training_parameters_file=training_parameters_file)

    ### LDM-SLR training step
    
    ### Here we train LDM-SLR method method using the preprocessed data, in either of the chosen 
    ### configurations (with or without optimization). The resulting LDM-SLR parameters are saved to the file. 
    ### See the DOCSTRING and the code for details.
    train_ldm_slr(training_data_file,hyperparameters,optimization_flag,ldm_slr_parameters_file)
   
    ### Evaluation data processing step
    
    ### Here we take new (unseen) realizations from the evaluation dataset and estimated forced response there
    ### using the traind LDM-SLR algorithm. The results are written to the output files.
    ### See the code and the comments for details.
    process_evaluation_data(root_folder,variable,hyperparameters,training_parameters_file,ldm_slr_parameters_file)



    
    ### Removing the directory for temporary files
    Popen('rm -r %s'%tmp_dir,shell=True).communicate()
    

    
