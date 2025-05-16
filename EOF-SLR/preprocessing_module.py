#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pickle as pkl
from scipy.interpolate import griddata


def get_one_cycle(data,tau,n1=None,n2=None,axis=0):
    '''
    

    Parameters
    ----------
    data : np.array[nt,...]
    tau : int
        must be less than nt
    n1 : int , optional
        Beginning of the interval, 0<=n1<=nt, the Default is 0.
    n2 : int, optional
        End of the interval, n1<=n2<=nt. The default is 0.

    Returns
    -------
    cycle : np.array[tau,...]
        cycle calculated for the data[n1:n2,...].

    '''
    
    data=np.swapaxes(data,0,axis)
    if n1 is None: n1=0
    if n2 is None: n2=data.shape[0]
    cycle=np.zeros(data[:tau,...].shape)
    for i in range(tau): 
        cycle[((n1+i)%tau),...]=data[n1+i:n2:tau,...].mean(axis=0)
        
    cycle=np.swapaxes(cycle,0,axis)
    return cycle

def extend_one_cycle(cycle,n1,n2,axis=0):
    '''
    extends the cycle from the interval [0:tau] to the interval [n1:n2]

    Parameters
    ----------
    cycle : np.array[tau,...]
    n1 : int >=0
    n2 : int >=n1

    Returns
    -------
    np.array[n2-n1,...]

    '''
    cycle=np.swapaxes(cycle,0,axis)
    tau=cycle.shape[0]
    new_shape=list(cycle.shape)
    new_shape=tuple([n2]+new_shape[1:])
    base=np.zeros(new_shape)
    for i in range(tau): 
        base[i::tau,...]=cycle[i,...]
    base=base[n1:n2,...]
    base=np.swapaxes(base,0,axis)
    return base


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

def get_weights_plev_lats(plev,lats):
    import math
    sqrt_cos_lat=np.sqrt(np.cos(math.pi*lats/180.))
    plev=-np.diff(plev)
    plev=np.concatenate([plev,np.array([plev[-1]])],axis=0)
    weights=sqrt_cos_lat*np.sqrt(plev)[:,None]
    return weights

def get_weights_lats_lons(lats,lons):
    import math
    sqrt_cos_lat=np.sqrt(np.cos(math.pi*lats/180.)*(abs(lons[0]-lons[1])*math.sin(math.pi*0.5*abs(lats[0]-lats[1])/180.)*0.5/180.))
    weights=np.repeat(sqrt_cos_lat[:,None],lons.shape[0],axis=-1)
    return weights

def get_weights(x,y,variable):
    if variable in ['zmta']:
        return get_weights_plev_lats(x,y)
    if variable in ['pr','monmaxpr','psl','tas','monmintasmin','monmaxtasmax','tos']:
        return get_weights_lats_lons(x,y)

def flatten_2d_data(data,valid_mask=None):
    if valid_mask is None: valid_mask=np.full((data.shape[-2],data.shape[-1]),True)
    shape=tuple(list(data.shape)[:-2]+[np.sum(valid_mask)])
    flat_data=np.full(shape,1e+31)
    flat_data[...]=data[...,valid_mask]
    return flat_data    

def unflatten_2d_data(data,lats,lons,valid_mask=None):
    if len(lats.shape)==1:
        if valid_mask is None: valid_mask=np.full((lats.shape[0],lons.shape[0]),True)
        shape=tuple(list(data.shape)[:-1]+[lats.shape[0]]+[lons.shape[0]])
    else:
        if valid_mask is None: valid_mask=np.full((lats.shape),True)
        shape=tuple(list(data.shape)[:-1]+list(lats.shape))
    gridded_data=np.full(shape,1e+31)
    gridded_data[...,valid_mask]=data        
    return gridded_data


def interpolate_masked_values(data,x,y,valid_mask,variable):
    '''
    Arguments:
    
    data: np.array[nt,nx,ny]
    x: np.array[nx]
    y: np.array[ny]
    valid_mask: np.array[nt,nx,ny]
    
    Returns:
    
    interpolated_data: np.array[nt,nx,ny]
    
    Data with all masked values interpolated by the 'nearest' interpolator.
    '''

    nt,nx,ny=data.shape    
    y_target,x_target=np.meshgrid(y,x)    
    xi=np.stack([x_target.flatten(),y_target.flatten()],axis=-1)
    
    if variable in ['pr','monmaxpr','psl','tas','monmintasmin','monmaxtasmax','tos']:
        y=np.concatenate([y-360.,y,y+360.],axis=-1)
        data=np.concatenate([data,data,data],axis=-1)
        valid_mask=np.concatenate([valid_mask,valid_mask,valid_mask],axis=-1)
    elif variable!='zmta':
        print('Variable is not in the intended list')
        exit()
    
    y,x=np.meshgrid(y,x)
    
    data_interpolated=[]
    for i in range(data.shape[0]):
        points=np.stack([x[valid_mask[i,:,:]].flatten(),y[valid_mask[i,:,:]].flatten()],axis=-1)        
        values=data[i,valid_mask[i,:,:]].flatten()        
        newdata=griddata(points, values, xi=xi, method='nearest')    
        data_interpolated.append(newdata.reshape((nx,ny)))
    return np.stack(data_interpolated,axis=0)  

def interpolate_height_values(data,x,y,valid_mask):
    '''
    Arguments:
    
    data: np.array[nt,nx,ny]
    x: np.array[nx]
    y: np.array[ny]
    valid_mask: np.array[nt,nx,ny]
    
    Returns:
    
    interpolated_data: np.array[nt,nx,ny]
    
    Data with all masked values interpolated by the 'nearest' interpolator.
    '''
    
    y,x=np.meshgrid(y,x)
    
    xi=np.stack([x.flatten(),y.flatten()],axis=-1)
    data_interpolated=[]
    for i in range(data.shape[0]):
        points=np.stack([x[valid_mask[i,:,:]].flatten(),y[valid_mask[i,:,:]].flatten()],axis=-1)        
        values=data[i,valid_mask[i,:,:]].flatten()        
        newdata=griddata(points, values, xi=xi, method='nearest')    
        data_interpolated.append(newdata.reshape(x.shape))
    return np.stack(data_interpolated,axis=0)  

def interpolate_land_values(data,lats,lons,valid_mask):
    '''
    Arguments:
    
    data: np.array[nt,nlats,nlons]
    lats: np.array[nlats]
    lons: np.array[nlons]
    valid_mask: np.array[nt,nlats,nlons]
    
    Returns:
    
    interpolated_data: np.array[nt,nlats,nlons]
    
    Data with all masked values interpolated by the 'nearest' interpolator.
    '''
    
    lons_ext=np.concatenate([lons-360.,lons,lons+360.],axis=-1)
    data_ext=np.concatenate([data,data,data],axis=-1)
    valid_mask_ext=np.concatenate([valid_mask,valid_mask,valid_mask],axis=-1)
    lons_ext,lats_ext=np.meshgrid(lons_ext,lats)
    lons,lats=np.meshgrid(lons,lats)
    
    xi=np.stack([lats.flatten(),lons.flatten()],axis=-1)
    data_interpolated=[]
    for i in range(data_ext.shape[0]):
        points=np.stack([lats_ext[valid_mask_ext[i,:,:]].flatten(),lons_ext[valid_mask_ext[i,:,:]].flatten()],axis=-1)
        values=data_ext[i,valid_mask_ext[i,:,:]].flatten()        
        newdata=griddata(points, values, xi=xi, method='nearest')    
        data_interpolated.append(newdata.reshape(lats.shape))
    return np.stack(data_interpolated,axis=0)  
    
    
def pca_sklearn(data,n,alarm=True,seed=1):
    '''
    Approximate data by
    X=np.matmul(P,np.conjugate(V))+residual

    Parameters
    ----------
    data : np.array[nt,d]
    n : int >=0
    alarm : boolean, optional
        The default is True.
    seed : random seed, optional
        The default is 1.

    Returns
    -------
    
    V,P,L,total
    
    V=np.array[n,d] -- n EOFs
    P=np.array[nt,n] -- n PCs
    L=np.array[n] -- eigenvalues (mean squares of PCs)
    total=float -- np.sum((data*np.conjugate(data)).mean(axis=0))
    '''    
    state=np.random.get_state()
    np.random.seed(seed)   
    total=np.sum((data*np.conjugate(data)).mean(axis=0)) 
    average=data.mean(axis=0)
    sum2average=np.sum(average*np.conjugate(average))
    if (sum2average>1e-10*total) and alarm:
        import sys
        sys.stderr.write('Attention: data is not centered! dev='+str(sum2average)+'\n')
    from sklearn.decomposition import PCA
    pca=PCA(n_components=n)
    pca.fit(data)
    P=pca.transform(data)
    V=pca.components_
    L=pca.explained_variance_ratio_

    L=L*total
    np.random.set_state(state)
    return V,P,L,total    


def log_transform(data):
    return np.log(data+0.1)

def inverse_log_transform(data):
    return np.maximum(np.exp(data)-0.1,0.)


from netCDF4 import Dataset
def get_data(datafile,variable,mode):
    '''
    Custom .nc file reading function. Reads variable from a datafile.
    The returning values depend on the 'mode'.
    

    Parameters
    ----------
    datafile : str

    variable : str

    mode : 'training' or 'evaluation_metadata' or 'evaluation_data' or 'evaluation_mask'
        DESCRIPTION. The default is 'training'.

    Returns
    -------
    if mode=='training':
        returns a tuple of (x_grid[nx],y_grid[ny],data[N,nx,ny],valid_mask[N,nx,ny]) -- two one-dimensional spatial axis arrays, data array and mask array
    
    if mode=='evaluation_metadata':
        returns a tuple of metadata (times,x_grid,y_grid,var_dict) used in the write_data function    
    
    if mode=='evaluation_data':
        returns a tuple of (data[N,nx,ny],valid_mask[N,nx,ny],evaluation_id) -- data and mask arrays, and 'evaluation_id' field
    
    if mode=='evaluation_mask':
        returns valid_mask boolean array with the shape [N,nx,ny] which masks valid data in the dataset

    '''
    varname={
        'pr':'pr',  
        'monmaxpr':'pr',
        'psl':'psl',  
        'tas':'tas',
        'monmintasmin':'tasmin',
        'monmaxtasmax':'tasmax',
        'tos':'tos',
        'zmta':'ta'
        }    
    
    dataset=Dataset(datafile,'r')
    
    if varname[variable] in ['ta']:
        xname='plev'
        yname='lat'
    elif varname[variable] in ['pr','psl','tas','tasmin','tasmax','tos']:
        xname='lat'
        yname='lon'
        
    x=np.array(dataset.variables[xname][:])
    y=np.array(dataset.variables[yname][:])
    times=dataset.variables['time'][:]

    data=dataset.variables[varname[variable]][:]#[...,0]
    if len(data.shape)>3 and data.shape[-1]==1:
        #squeezing the longitude dimension for zmta data
        data=data[...,0]
    ret_data=data.data    
    if varname[variable]=='pr':    
        ret_data=log_transform(ret_data)
     
    full_valid_mask=np.logical_and(data.mask==False,np.abs(data.data)<=1e+20)


    if mode=='training':
        return x,y,ret_data,full_valid_mask
    if mode=='evaluation_metadata':
        return times,x,y, dataset.variables[varname[variable]].__dict__    
    if mode=='evaluation_data':
        return ret_data,full_valid_mask,dataset.__dict__['evaluation_id']
    if mode=='evaluation_mask':
        return full_valid_mask    
    
    print('Wrong "mode" argument in get_data()')
    exit()
    #return times,x,y,ret_data,full_valid_mask, dataset.__dict__, dataset.variables[varname[variable]].__dict__

def write_data(file_name,variable,data,valid_mask,evaluation_id,times,x_grid,y_grid,var_dict):
    '''
    Custom .nc file writing function. Writes the data as a 'variable' with valid values specified by valid_mask,
    and adds 'time' and two spatial dimensions with some metadata.

    Parameters
    ----------
    file_name : str, name of the output netCDF-file which will appear
        
    variable : str, name of the variable to write
        
    data : np.array [N,nx,ny] 
        data to write
    valid_mask : np.array [N,nx,ny]
        Boolean mask of the valid values for the data to write
    evaluation_id : str
        'evaluation_id' field to write
    times,x_grid,y_grid,var_dict : parameters returned by get_data function in 'evaluation_metadata' mode

    Returns
    -------
    None.

    '''
    varname={
        'pr':'pr',  
        'monmaxpr':'pr',
        'psl':'psl',  
        'tas':'tas',
        'monmintasmin':'tasmin',
        'monmaxtasmax':'tasmax',
        'tos':'tos',
        'zmta':'ta'
        }
    if varname[variable]=='pr': 
        data=inverse_log_transform(data)
    data=np.ma.masked_array(data,mask=np.logical_not(valid_mask))

    dataset = Dataset(file_name,'w', format='NETCDF4')

    if varname[variable] in ['ta']:
        xname='plev'
        yname='lat'  
        
    elif varname[variable] in ['pr','psl','tas','tasmin','tasmax','tos']:
        xname='lat'      
        yname='lon'

    dataset.createDimension('time', None)
    dataset.createDimension(xname, len(x_grid)) 
    dataset.createDimension(yname, len(y_grid))

    var = dataset.createVariable(varname[variable], 'f8' , ('time',xname,yname),fill_value=var_dict['_FillValue'])
    time = dataset.createVariable('time', 'f8' , 'time')
    x = dataset.createVariable(xname, 'f8', xname)
    y = dataset.createVariable(yname, 'f8', yname)
  
    if varname[variable] in ['ta']:
        y[:]=y_grid
        y.axis='Y'
        y.units='degrees_north'
        y.long_name='latitude'
        y.standard_name='latitude'  

        x[:]=x_grid
        x.axis='Z'
        x.units='Pa'
        x.long_name='pressure'
        x.standard_name='air_pressure'  

        
    elif varname[variable] in ['pr','psl','tas','tasmin','tasmax','tos']:
        x[:]=x_grid
        x.axis='Y'
        x.units='degrees_north'
        x.long_name='latitude'
        x.standard_name='latitude'  
        
        y[:]=y_grid
        y.axis='X'
        y.units='degrees_east'
        y.long_name='longitude'
        y.standard_name='longitude'  

    
    time[:]=times
    time.axis='T'
    time.calendar='365_day'
    time.units='days since 1850-01-01'
    time.long_name='time'
    time.standard_name='time'
    
    dataset.evaluation_id=evaluation_id
    
    var[:]=data
    var.long_name=var_dict['long_name']
    var.units=var_dict['units']
    dataset.close()
    return



###############################################################

def get_training_metadata(root_folder,folder,nsamples):
    models=sorted(os.listdir(root_folder+folder))

    files=[]
    model_ids=[]
    
    for imodel in range(len(models)):
        model=models[imodel]
        wd=root_folder+folder+'/'+model
        tmp_files=list(sorted(filter(lambda x: x[-3:]=='.nc',os.listdir(wd))))
        for file in tmp_files:
            model_ids.append(imodel)
            files.append(folder+'/'+model+'/'+file)

    model_ids=np.array(model_ids)
    files=np.array(files)
    
    model_training_mask=np.full(len(model_ids),False)
    for imodel in range(len(models)):
        tmp_mask=(model_ids==imodel)
        insert_mask=np.full(tmp_mask.sum(),False)
        insert_mask[:nsamples]=True
        model_training_mask[tmp_mask]=insert_mask 
        
    return models,files[model_training_mask],model_ids[model_training_mask]

def get_evaluation_metadata(root_folder,variable):  
    folder_evaluation={
        'pr':'/Evaluation-Tier1/Amon/pr',  
        'monmaxpr':'/Evaluation-Tier1/Aday/monmaxpr',
        'psl':'/Evaluation-Tier1/Amon/psl',  
        'tas':'/Evaluation-Tier1/Amon/tas',
        'monmintasmin':'/Evaluation-Tier1/Aday/monmintasmin',
        'monmaxtasmax':'/Evaluation-Tier1/Aday/monmaxtasmax',
        'tos':'/Evaluation-Tier1/Omon/tos',
        'zmta':'/Evaluation-Tier1/Amon/zmta'
        }
    files=list(sorted(filter(lambda x: x[-3:]=='.nc',os.listdir(root_folder+folder_evaluation[variable]))))
    files=[folder_evaluation[variable]+'/'+file for file in files]
    files=np.array(files)
    return files


def read_training_data(root_folder,variable,nsamples):
    if variable=='pr':
        folder_training='/Training/Amon/pr'
    elif variable=='monmaxpr':
        folder_training='/Training/Aday/monmaxpr'
    elif variable=='psl':
        folder_training='/Training/Amon/psl'
    elif variable=='tas':
        folder_training='/Training/Amon/tas'
    elif variable=='monmintasmin':
        folder_training='/Training/Aday/monmintasmin'
    elif variable=='monmaxtasmax':
        folder_training='/Training/Aday/monmaxtasmax'
    elif variable=='tos':
        folder_training='/Training/Omon/tos'
    elif variable=='zmta':
        folder_training='/Training/Amon/zmta'    
        
    
    # Get metadata of the first 25 models of each available model
    models, files, model_ids = get_training_metadata(root_folder,folder_training,nsamples)  


    data=[]
    valid_masks=[]
    for file in files:
        x_grid,y_grid,tmp_data,tmp_valid_mask=get_data(root_folder+file,variable,mode='training')
        data.append(tmp_data[(1950-1880)*12:(2022-1880+1)*12,...])  
        valid_masks.append(tmp_valid_mask[(1950-1880)*12:(2022-1880+1)*12,...])
    data=np.stack(data,axis=0)    
    print(data.shape)
    
    return x_grid,y_grid,data,np.stack(valid_masks,axis=0), model_ids



def read_evaluation_masks(root_folder,variable):
    files=get_evaluation_metadata(root_folder,variable)
    valid_masks=[]
    for file in files:
        tmp_valid_mask=get_data(root_folder+file,variable,mode='evaluation_mask')
        valid_masks.append(tmp_valid_mask)
    return np.stack(valid_masks,axis=0)


def subtract_annual_cycle(data,n1,n2):
    cycle=get_one_cycle(data=data,tau=12,n1=n1,n2=n2,axis=1)
    x_anom=data-extend_one_cycle(cycle=cycle,n1=0,n2=data.shape[-3],axis=1)
    return x_anom, cycle   

def perform_pca(x_grid,y_grid,variable,data,valid_mask,dim_x):  
    #annual cycle and normalization
    data1,subtracted_annual_cycles=subtract_annual_cycle(data,n1=0,n2=data.shape[-3])
    x0=data1.reshape((-1,data1.shape[-2],data1.shape[-1])).mean(axis=0)
    data1=data1-x0
    x2=data1.reshape((-1,data1.shape[-2],data1.shape[-1])).std(axis=0)
    weights=get_weights(x_grid,y_grid,variable)  
    data1=data1/x2*weights  
    
    #Ensemble EOFs
    flat_data=flatten_2d_data(data1,valid_mask=valid_mask)
    print('PCA')
    V,P,L,total=pca_sklearn(flat_data.reshape((-1,flat_data.shape[-1])),dim_x)   
    P=P.reshape((flat_data.shape[0],flat_data.shape[1],P.shape[-1]))
   
    return x0,x2,V,P,L,total,subtracted_annual_cycles
            
            
def preprocessing(root_folder,variable,output_training_data_file,output_training_parameters_file,nsamples=25, D=200):
    '''
    Exemplary data preprocessing function.
    
    1. Reads the data related with the given variable from the 'Training' subfolder of root_folder.
    The data is assumed to be generated by Earth System Models, and we take the first 'nsamples' (default 25) members of each model. Their spatial and temporal grid is assumed to be the same.

    2. Extends their spatial masks taking into account spatiotemporal masks of the data from 'Evaluation-Tier1' subfolder.

    3. Preprocesses the training data: removes annual cycle from each realization, then normalizes the data to have zero mean and unit std across time and multi-model ensemble, then computes ensemble EOFs.
    
    4. Generates two output files defined by output_training_data_file and output_training_parameters_file.
 
    Parameters
    ----------
    root_folder : str
    
    variable : str
    
    output_training_data_file : str
        
        The name of the file which will contain a tuple (P,model_ids).
        P is np.array with the shape (M,N,D) and contains M time series of the length N and dimension D. 
        These time series are the principal components obtained by projection of the data onto leading ensemble EOFs.
        model_ids is np.array with the shape (M) and contains integers from 0 to nmodels-1, defining the models which the realizations are drawn from. 
    
    output_training_parameters_file : str

        The name of the file which will contain realizations-independent preprocessing parameters, like EOF patterns, normalization coefficients, grid, masks, etc.
        They will be used to preprocess new data.
        More exactly, they are contained in a form of a tuple (x0,x2,V,L,total,x_grid,y_grid,valid_mask), where x0[nx,ny] and x2[nx,ny] are mean and std patterns, V[D,number_of_valid_grid_nodes] -- EOF patterns, L[D] -- EOF eigenvalues, total -- full EOF variance, x_grid[nx] and y_grid[ny] -- spatial grid axes, valid_mask -- the extended mask of valid values used to compute EOFs 
        
        
    nsamples : number of ensemble members to take from each model (default is 25).
    
    D: number of ensemble EOFs to use (default is 200).

    Returns
    -------
    None.

    '''    
    
    # Read training datasets
    x_grid,y_grid,training_data,training_valid_masks,model_ids=read_training_data(root_folder,variable,nsamples=nsamples)
    
    # Read evaluation data masks
    evaluation_valid_masks=read_evaluation_masks(root_folder,variable)
    
    # Get extended spatial mask and extrapolate all training data on it (nearest type of extrapolation) 
    nx,ny=x_grid.shape[0],y_grid.shape[0]
    all_masks=np.concatenate([training_valid_masks.reshape((-1,nx,ny)),evaluation_valid_masks.reshape((-1,nx,ny))],axis=0)
    valid_mask=np.any(all_masks,axis=0)
  
    unknown_area=np.logical_not(valid_mask==np.all(all_masks,axis=0))
    print(np.sum(unknown_area),np.sum(valid_mask))
    if np.sum(unknown_area)>0:
        for i in range(training_data.shape[0]):
            training_data[i,:,:,:]=interpolate_masked_values(training_data[i,:,:,:],x_grid,y_grid,training_valid_masks[i,:,:,:],variable)          
    else:
        print('Equal sea masks. Skipping interpolation')

    
    # Preprocessing is here: annual cycle removal, normalization and EOF-compression
    x0,x2,V,P,L,total,subtracted_annual_cycles=perform_pca(x_grid,y_grid,variable,training_data,valid_mask,dim_x=D)   
    
    #realization-independent parameters (will be used to preprocess new data)
    with open(output_training_parameters_file,'wb') as f:
        pkl.dump((x0,x2,V,L,total,x_grid,y_grid,valid_mask),f)    

    #realization-dependent parameters (should be recomputed for new data)
    with open(output_training_data_file,'wb') as f:
        pkl.dump((P,model_ids),f)
           

def preprocess_new_data(new_data,new_valid_masks,variable,training_parameters): 
    '''
    New data preprocessing using the parameters (normalization constants, EOF patterns) of the training data preprocessing.
    The same operations are performed as in training data preprocessing.

    Parameters
    ----------
    new_data : np.array [M_new,N,nx,ny]
        M_new realizations of new time series of the duration N and spatial dimensions nx,ny.
    new_valid_masks : np.array [M_new,N,nx,ny]
        boolean array of the same shape as new_data which masks valid values in new_data.
    variable : str,    name of the variable
        Used to correctly weight the spatial data before EOF decomposition.
    training_parameters : tuple of training parameters saved at the training data preprocessing step, see the description there.

    Returns
    -------
    P : np.array [M_new,N,D]
        Ensemble EOF time series of new data.
    subtracted_annual_cycles : np.array[M_new,12,nx,ny]
        Annual cycles subtracted from each of new realizations. They should be added back at the postprocessing step.

    '''

    ### Unpacking parameters
    x0,x2,V,L,total,x_grid,y_grid,valid_mask=training_parameters             
    nx,ny=x_grid.shape[0],y_grid.shape[0]
    
    ### Checking whether masks of new data is covered by valid_mask used in the preprocessing
    all_masks=new_valid_masks.reshape((-1,nx,ny))
    unknown_area=np.logical_not(valid_mask==np.all(all_masks,axis=0))
    print(np.sum(unknown_area),np.sum(valid_mask))
    ### If there is uncovered area, interpolate the data in this area
    if np.sum(unknown_area)>0:
        for i in range(new_data.shape[0]):
            new_data[i,:,:,:]=interpolate_masked_values(new_data[i,:,:,:],x_grid,y_grid,new_valid_masks[i,:,:,:],variable)          
    else:
        print('Equal sea masks. Skipping interpolation')
        
    ### Removing annual cycle from each realization    
    data1,subtracted_annual_cycles=subtract_annual_cycle(new_data,n1=0,n2=new_data.shape[-3])
    ### Normalizing and weighting the data with the coefficients determined at the preprocessing step
    weights=get_weights(x_grid,y_grid,variable)
    data1=(data1-x0)/x2*weights
    ### Projecting normalized data onto ensemble EOFs from the preprocessing step
    flat_data=flatten_2d_data(data1,valid_mask=valid_mask)     
    P=np.tensordot(flat_data,V,axes=(-1,-1))  
    
    return P, subtracted_annual_cycles

def postprocess_pcs(P,subtracted_annual_cycles,variable,training_parameters):
    '''
    The procedure which reverse preprocessing steps and applies them in the reversed order to the data given in the form of ensemble EOF time series.

    Parameters
    ----------
    P : np.array [M,N,D]
        The data to postprocess in the form of M realizations of ensemble EOF time series.
    subtracted_annual_cycles : np.array [M,12,nx,ny]
        The annual cycles which will be added to the data during postprocessing.
    variable : str, name of the variable
        Only used to correctly weight spatial grid.
    training_parameters : tuple of training parameters saved at the training data preprocessing step, see the description there.


    Returns
    -------
    data : the result of EOF back-rotation, de-normalization and annual cycle addition to each of M realizations.
    .

    '''
    ### Unpacking parameters
    x0,x2,V,L,total,x_grid,y_grid,valid_mask=training_parameters  
    npcs=P.shape[-1]
    
    ### Producing the coefficients of EOF-back rotation including de-normalization
    weights=get_weights(x_grid,y_grid,variable)
    V=V[:npcs,:]
    V=unflatten_2d_data(V,x_grid,y_grid,valid_mask=valid_mask)/weights*x2
    ### Performing EOF back-rotation of input array P
    data=x0+np.tensordot(P,V,axes=(-1,0))
    ### Adding annual cycles
    data=data+extend_one_cycle(cycle=subtracted_annual_cycles,n1=0,n2=P.shape[-2],axis=-3)  
    return data

        


    