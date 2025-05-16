import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
import pydmd
import pandas as pd

import os

import pydmd

import datetime


def load_data(data_path: str = './data/Evaluation-Tier1', dir_names: dict = {'Omon': ['tos']}, group_name: str = 'UV-ISP'):
    '''
    Load all the data into a dictionary using path data_path with specific paths as a dictionary in dir_names

    Inputs:
        data_path: path to evaluation data
        dir_names: dictionary with 
            key: group name
            value: variable name
    Outputs:
        datasets: dictionary with data: 
            key:  $VARIABLE_$MEMBERID_$TIER_$METHOD_$GROUPNAME
            value: numpy array of anomalies
        nlat: number of latitude
        nlon: number of longitude
        t: number of time steps
        ds: xarray dataset 
        var_short: string with variable name
    '''

    if 'Training' in data_path:
        tier = 'Training'
    else:
        tier = data_path[-1]

    #load data

    datasets = {}
    for dir_name in dir_names.keys():

        vars = dir_names[dir_name]

        for var in vars:
            
            # Loop through each file
            directory = os.path.join(data_path, dir_name, var)
            # Define the file path
            listdir = os.listdir(directory)

            for i, file in enumerate(listdir, start=1):

                print('File {}/{}'.format(i,len(listdir)), end='\r')
                # Reading temperature file
                ## Open the NetCDF file using xarray
                file_path = os.path.join(directory, file)
                ds = xr.open_dataset(file_path)

                var_short = ds.to_dataframe().columns[0]

                # Specify the coarsening factor
                # Compute monthly anomalies
                # raw_data = np.array(ds[var_short])
                climatology = ds[var_short].groupby('time.month').mean(dim='time')
                anomalies = ds[var_short].groupby('time.month') - climatology
                ds.close()

                #convert to numpy array
                tas_cube = np.array(anomalies)
                [t, nlat, nlon] = tas_cube.shape
                tas_all = np.reshape(tas_cube, (t, nlat*nlon))


                split_fpath = file_path.split('/')
                if 'Training' in file_path:
                    name_memID = split_fpath[-1].split('.')[0].split('_')[-1] + split_fpath[-1].split('.')[1]
                    group_name = split_fpath[-4]
                else:
                    #match the submission style
                    #$VARIABLE_$MEMBERID_$TIER_$METHOD_$GROUPNAME.nc
                    name_memID = split_fpath[-1].split('.')[0]
                data_name = f'{name_memID}_{tier}_{method}_{group_name}'

                datasets[data_name] = tas_all

    return datasets, t, nlat, nlon, ds, var_short




if __name__ == '__main__':

    #set parameters and load forcings

    # Load Forcings
    forcings_df = pd.read_csv('./interpolatedForcing.csv')
    forcings  = np.array(forcings_df.iloc[840:]['totalforcing']) #for testing
    # forcings  = np.array(forcings_df['totalforcing'])              #for training


    #path to evaluation tier
    # data_path = '/data/databases/ForceSMIP/Training/'  #for training
    data_path = '/data/databases/ForceSMIPNew/Evaluation-Tier1'  #for testing

    out_path = './predictions' #directory to save results

    method = 'DMDc' #your method name here

    group_name = 'UV-ISP'

    #evaluaton variables

    #for testing
    dir_names = {'Aday': ['monmaxpr', 'monmaxtasmax', 'monmintasmin'],
                'Amon': ['pr', 'psl', 'tas'],
                'Omon': ['tos']
                }

    #for training
    # dir_names = {'Amon': ['tas/CESM2']
    #             }   


    datasets, t, nlat, nlon, ds, var_short = load_data(data_path, dir_names, group_name)


    #predict forced response on the data

    for data_name in datasets.keys():
        data_all  = datasets[data_name]

        #remove nans
        nan_mask = np.all(np.isfinite(data_all), axis=0)
        data = data_all[:,nan_mask]

        ################################
        ################################
        #YOUR METHOD HERE!

        #PCA
        pca = PCA(n_components = .3) #3)
        pca.fit(data)
        data_pca = pca.transform(data)

        #DMDc      
        my_dmdc = pydmd.DMDc(svd_rank=-1)
        my_dmdc.fit(data_pca.T, np.expand_dims(forcings[:-1],0))
        dmdc_eigs = my_dmdc.eigs

        #select mode with biggest real part
        idx = np.argsort(np.real(dmdc_eigs))[::-1][:1]
        sel_modes = my_dmdc.modes[:,idx]
        sel_dynamics = my_dmdc.dynamics[idx,:]
        sel_eigs = dmdc_eigs[idx]

        #predict forced response
        dmdc_pred_forced_resp_pca = np.linalg.multi_dot(
                                [sel_modes, np.diag(sel_eigs), np.linalg.pinv(sel_modes), data_pca[:-1,:].T]) + my_dmdc.B @ np.expand_dims(forcings[:-1],0)

        #add the mask back in
        dmdc_pred_forced_resp = np.empty((t-1,nlat*nlon))
        dmdc_pred_forced_resp[:, nan_mask] = pca.inverse_transform(dmdc_pred_forced_resp_pca.T.real)
        dmdc_pred_forced_resp[:,~nan_mask] = np.nan

        #reshape and predict forced response at first time as the signal at time 1
        dmdc_pred_forced_resp = np.vstack( [data_all[0], dmdc_pred_forced_resp] )
        dmdc_pred_forced_resp = dmdc_pred_forced_resp.reshape((t, nlat, nlon))

        ################################
        ################################

        #make a xarray object
        forced_resp_xr = ds.copy()
        forced_resp_xr[var_short] = ({'time':ds['time'].values,'lat': ds['lat'].values, 'lon': ds['lon'].values},dmdc_pred_forced_resp)
        forced_resp_xr.attrs = {'creation_date': str(datetime.datetime.now()), 'method': 'DMDc'}

        #save the result
        f_name = os.path.join(out_path, f'{data_name}.nc')
        forced_resp_xr.to_netcdf(f_name)



