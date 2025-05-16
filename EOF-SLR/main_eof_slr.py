#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The program implements EOF-SLR method to estimate forced signal response in a single spatio-temporal 
climate realization after training on large ensembles produced by Earth System Models.
As a particular example, the main section at the bottom of this file can be used to estimate forced 
response in the ForceSMIP evaluation data, if provided with the right input (see the comments there). 

The main EOF-SLR training step is implemented via the function 'train_eof_slr'. 
SLR is implemented via the class smooth_linear_regressor ant its methods 'fit', 'run', 'save', 'load' and can be used independently.

EOF-SLR-based estimation of forced response from new data is implemented via the function 'process_new_realization'


Commands to install environment in conda:
    
conda create --name my_env python=3.10 numpy
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
from smooth_linear_regressor import smooth_linear_regressor


#############################################################################################
# EOF-SLR TRAINING FUNCTIONS ################################################################
#############################################################################################

def train_eof_slr(training_data_file,hyperparameters,output_eof_slr_parameters_file):
    '''
    This function trains EOF-SLR method with predefined hyperparameters.
    The list 'slr_windows' is used as a hyperparameter to define the window values to choose from when smoothing inputs in the SLR training step.
    
    The function takes multi-model ensemble data from training_data_file and saves the trained parameters into output_eof_slr_parameters_file.

    Parameters
    ----------
    training_data_file : str
        The name of the file with training data in pickle format, which contains a tuple of ensemble EOF time series P with the shape [M,N,D] and array of model IDs of the shape [M].
        Array of Model IDs' must contain integers from 0 to nmodels-1, defining the models which the realizations are drawn from. 
    hyperparameters: dictionary
        Must contain the keys:
            'K' : int>0, number of ensemble EOF time series used to determine K forced components by model-wise ensemble averaging; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.
            'slr_windows': non-empty list of ints, a set of smoothing window sizes to choose from in SLR method
    output_eof_slr_parameters_file : str
        Name of the file which is used to save trained EOF-SLR parameters.

    Returns
    -------
    None.

    '''

    print('Training started')

    ### Reading multi-model ensemble EOF time series and model identifiers
    with open(training_data_file,'rb') as f: P, model_ids=pkl.load(f)  

    ### Extracting array of unique models represented in model_ids    
    models=np.sort(np.unique(model_ids))
    print(models)
    nmodels=models.shape[0]

    
    # EOF step: approximating forced signal components from ensemble EOF time series
    
    ### Taking K leading ensemble EOF time series to define forced signal components
    P_eof=P[...,:hyperparameters['K']]    
    ### forced signal components are defined as model-wise ensemble mean of K leading ensemble EOF time series
    ### 'unique_forced_signal_components' array has shape [nmodels,N,K]        
    s=[P_eof[model_ids==imodel,:,:].mean(axis=0) for imodel in range(nmodels)]
    unique_forced_signal_components=np.stack(s,axis=0) 
    ### 'forced_signal_components' array has shape [M,N,K] and corresponds to M multi-model realizations 
    ### They are defined in the EOF space and can be demapped to original coordinates by further EOF bac-rotation and de-normalization
    forced_signal_components=unique_forced_signal_components[model_ids,:,:]


    ### SLR step: training SLR method

    ### Taking K_slr leading ensemble EOF time series to define the input for SLR method
    P_slr=P[...,:hyperparameters['K_slr']] 
    ### Creating the SLR object
    SLR=smooth_linear_regressor()
    ### Fitting SLR mapping from K_slr ensemble EOF time series to their correspopnding forced signal components (including optimization along 'windows')
    SLR.fit(P_slr,forced_signal_components,hyperparameters['slr_windows'])
    ### Getting the dictionary with the trained SLR_parameters
    SLR_parameters=SLR.save()
    ### Printing 'df' optimal window sizes found by SLR method
    print(SLR.opt_windows)

    
    ### Saving final SLR parameters needed for the evaluation step
    with open(output_eof_slr_parameters_file,'wb') as f: 
        pkl.dump(SLR_parameters,f)  
    print('Training finished')


def get_optimal_hyperparameters(variable):
    '''
    Custom function which returns the dictionary with optimal hyperparameters for a given variable from exemplary ForceSMIP data.
    To estimate these hyperparameters, 10 realizations were taken from each model, and the function 'train_eof_with_optimization' was launched with different hyperparameter K_slr=20,50,100,200.
    The hyperparameter K was fixed to be 20 there. The optimal value of K_slr was chosen based on validation of the forced response reconstruction on the rest of the training data.
    
    The list 'slr_windows' in SLR method was always kept the same.

    Parameters
    ----------
    variable : str, name of the variable

    Returns
    -------
    hyperparameters : dictionary with optimal hyperparameters pre-computed as described

    '''
       
    if variable=='pr': 
        K,K_slr=20,100      
    if variable=='monmaxpr': 
        K,K_slr=20,20
    if variable=='psl': 
        K,K_slr=20,100
    if variable=='tas': 
        K,K_slr=20,200
    if variable=='monmintasmin': 
        K,K_slr=20,100
    if variable=='monmaxtasmax': 
        K,K_slr=20,100
    if variable=='tos': 
        K,K_slr=20,200
    if variable=='zmta': 
        K,K_slr=20,200   
    hyperparameters={'K':K,'K_slr':K_slr,'slr_windows':list(range(1,64+1))+[128,256,512]}
    return hyperparameters


#############################################################################################
# EVALUATION DATA PROCESSING FUNCTIONS ######################################################
#############################################################################################

def process_evaluation_data(root_folder,variable,hyperparameters,training_parameters_file,eof_slr_parameters_file):
    '''
    This function takes new (unseen) realizations from evaluation dataset and estimates forced response there using the traind EOF-SLR algorithm. 
    The results are written to the output .nc files of the format 'variable_MEMBERID_tier1_EOF-SLR.nc'

    Parameters
    ----------
    root_folder : str, Path to a folder with the datasets (e.g. root_folder='/home/user/some/path/ForceSMIP')

    variable : str, name of the variable as in the training step.

    hyperparameters: dictionary, the same as in traininig step
        Must contain the keys:
            'K' : int>0, number of ensemble EOF time series used to determine K forced components by model-wise ensemble averaging; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.
    
    training_parameters_file : file with the training parameters saved at the preprocessing step.
        
    eof_slr_parameters_file : file with the trained EOF-SLR parameters saved at the preprocessing step

    Returns
    -------
    None.

    '''
    #Reading the preprocessing parameters (normalization constants, EOF patterns etc.)
    with open(training_parameters_file,'rb') as f:
        training_parameters=pkl.load(f) 

    #Reading from trained eof_slr_parameters_file (parameters of the trained SLR method)
    with open(eof_slr_parameters_file,'rb') as f:
        SLR_parameters=pkl.load(f) 
        
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
                                                    training_parameters,SLR_parameters)
        
        ### Defining the output .nc file and writing the forced response to it.
        file_name='%s_%s_%s_%s.nc'%(variable,MEMBERID,'tier1','EOF-SLR')
        ### This is a custom writing function. See the docstring for the details.
        write_data(file_name,variable,new_forced_response,new_sea_masks,MEMBERID,times,x_grid,y_grid,var_dict)
        

def process_new_realization(new_data,new_sea_masks,variable,hyperparameters,training_parameters,SLR_parameters):
    '''
    This function gets new data (one realization) and estimates the forced response using the trained EOF-SLR method

    Parameters
    ----------
    new_data : np.array [N,nx,ny]
        Spatio-temporal data (one realization) with the time duration N and spatial dimensions nx, ny.
    new_sea_masks : np.array [N,nx,ny]
        Boolean array with the True values masking the valid data.
    variable : str, name of the variable as in the training step.

    hyperparameters: dictionary, the same as in traininig step
        Must contain the keys:
            'K' : int>0, number of ensemble EOF time series used to determine K forced components by model-wise ensemble averaging; must be less or equal than D.
            'K_slr' : int>0, number of ensemble EOF time series used as an input in SLR method; must be less or equal than D.
    
    training_parameters : tuple of the training_parameters saved at the preprocessing step.
    
    SLR_parameters : SLR parameters saved at the training step.

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
    
    ### Loading SLR parameters into a new regressor object
    SLR=smooth_linear_regressor()
    SLR.load(SLR_parameters)
    
    ### Applying SLR to approximate forced components from new time series P
    ### 'new_eof_forced_components' has shape [1,N,K]
    new_eof_forced_components=SLR.run(P[...,:hyperparameters['K_slr']])
    
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
    ### The name of the file which will be created after EOF-SLR method training step
    eof_slr_parameters_file=tmp_dir+'/eof-slr-parameters.pkl3'


    ### Defining hyperparameters for EOF-SLR method training step. 
    ### By default, the precomputed optimal hyperparameters are loaded for a given variable of exemplary ForceSMIP dataset.
    ### See the docstring of 'get_optimal_hyperparameters' function for the details of this optimization.
    if len(sys.argv)<=3:
        hyperparameters=get_optimal_hyperparameters(variable)
    ### Alternatively, the discrete hyperparameters can be defined manually, e.g. via command line args
    else:
        hyperparameters={'K':int(sys.argv[3]),'K_slr':int(sys.argv[4]),
                         'slr_windows':list(range(1,64+1))+[128,256,512]}




    ######################################################################################################
    # COMPUTATION STEPS ##################################################################################
    ######################################################################################################
    
    ### Preprocessing step
    
    ### Here we load the training data, perform preprocessing and create two files with the data and parameters
    ### Annual cycle removal, data normalization and ensemble-EOF rotation is here. See the DOCSTRING and the code for details.
    preprocessing(root_folder,variable,output_training_data_file=training_data_file,output_training_parameters_file=training_parameters_file)

    ### EOF-SLR training step
    
    ### Here we train EOF-SLR method method using the preprocessed data. The resulting EOF-SLR parameters are saved to the file. 
    ### See the DOCSTRING and the code for details.
    train_eof_slr(training_data_file,hyperparameters,eof_slr_parameters_file)
   
    ### Evaluation data processing step
    
    ### Here we take new (unseen) realizations from the evaluation dataset and estimated forced response there
    ### using the traind EOF-SLR algorithm. The results are written to the output files.
    ### See the code and the comments for details.
    process_evaluation_data(root_folder,variable,hyperparameters,training_parameters_file,eof_slr_parameters_file)



    
    ### Removing the directory for temporary files
    Popen('rm -r %s'%tmp_dir,shell=True).communicate()
    

    
