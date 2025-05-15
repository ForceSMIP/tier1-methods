import time as clocktime
import glob
import xcdat as xc
import xarray as xr
import numpy as np
import os
import pickle

# principal component analysis
from eofs.xarray import Eof

from matplotlib import pyplot as plt
import cartopy
from matplotlib import colors
import matplotlib as mpl
# define a lambda function to perform natural sort
import re
natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("(\d+)", s)]
import argparse
from utils import load_data, evaluate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str, default='pr')
    # parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--neofs', type=int, default=1)
    args = vars(parser.parse_args())
    return args

cmipTable = {
    "pr": "Amon",
    "psl": "Amon",
    "tas": "Amon",
    "zmta": "Amon",
    "tos": "Omon",
    "siconc": "OImon",
    "monmaxpr": "Aday",
    "monmaxtasmax": "Aday",
    "monmintasmin": "Aday",
}
cmipVar = {
    "pr": "pr",
    "psl": "psl",
    "tas": "tas",
    "zmta": "ta",
    "tos": "tos",
    "siconc": "siconc",
    "monmaxpr": "pr",
    "monmaxtasmax": "tasmax",
    "monmintasmin": "tasmin",
}
nlat = 72
nlon = 144

stand = True

if __name__ == '__main__':
    args = get_args()
    variable = args['variable']
    neofs = args['neofs']
    # test_model = args['test_model']
    models = ['CESM2', 'MIROC6', 'MPI-ESM1-2-LR', 'MIROC-ES2L', 'CanESM5']
    training_models = models.copy()
    root_dir = "/p/lustre3/shiduan/ForceSMIP"  # path to forcesmip data (NCAR)
    ncvar = variable  # variable to be used: pr, psl, tas, zmta, tos, siconc, monmaxpr, monmaxtasmax, monmintasmin
    vid = cmipVar[ncvar]  # the variable id in the netcdf file differs – this maps to the standard CMIP variable name
    reference_period = ("1950-01-01", "2022-12-31") # climatological period (for anomaly calculations)
    print(ncvar)
    print(vid)
    # choose evaluation data
    eval_tier = "Tier1"  # Tier1, Tier2, or Tier3
    tv_time_period = ("1950-01-01", "2023-01-01")
    # get training models
    files = glob.glob(root_dir + '/Training/' + cmipTable[ncvar] + '/' + ncvar + '/*')
    models = [p.split('/')[-1] for p in files]
    if not os.path.exists('data/'):
        os.mkdir('data')
    model_mean_list = []
    un_forced_list = []
    
    for imodel, model in enumerate(training_models):
        # start timer
        stime = clocktime.time()
        # get model files
        mpath = root_dir + '/Training/' + cmipTable[ncvar] + '/' + ncvar + '/' + model
        mfiles = glob.glob(mpath + '/*')
        # parse file names to get list of model members
        # CESM2 has a non-CMIP naming convention
        if model == "CESM2":
            members = [p.split("ssp370_")[-1].split(".1880")[0] for p in mfiles]
        else:
            members = [p.split("_")[-1].split(".")[0] for p in mfiles]
        members.sort(key=natsort)
        # print progress
        print(str(imodel + 1) + " / " + str(len(models)) + ": " + model + " (" + str(len(members)) + " members)")
        # initialize model ensemble xarray dataset
        ds_model = None
        for im, member in enumerate(members):
            # print member progress
            print('.', end='')
            # get member filename
            fn = glob.glob(mpath + "/*_" + member + ".*.nc")
            # make sure filename is unique
            if len(fn) != 1:
                raise ValueError("Unexpected number of model members")
            else:
                fn = fn[0]
            # load data
            ds = xc.open_dataset(fn)
            ds = ds.bounds.add_missing_bounds(axes=['T'])
            # remove singletons / lon
            ds = ds.squeeze()
            #ds = ds.drop_vars('lon')
            # subset data to user-specified time period
            ds = ds.sel(time=slice(tv_time_period[0], tv_time_period[1]))
            # calculate departures (relative to user-specified reference time period)
            ds = ds.temporal.departures(vid, freq='month', reference_period=reference_period)
            if 'file_qf' in ds.variables:
                ds = ds.drop('file_qf')
            if 'ref_time' not in locals():
                ref_time = ds.time
                #print(ref_time)
            # validate model/member time is the same as reference time
            #ref_time = ds.time
            for i, t in enumerate(ds.time.values):
                m = t.month; y = t.year
                #print(m)
                #print(y)
                rt = ref_time.values[i]; rm = rt.month; ry = rt.year
                #print(rt)
                if ((ry != y) | (rm != m)):
                    raise ValueError("model time and reference time do not match")
            ds["time"] = ref_time.copy()
            if ds_model is None:
                ds_model = ds
            else:
                ds_model = xr.concat((ds_model, ds), dim='member')
        # after looping over members, compute model ensemble mean time series
        ds_model_mean = ds_model.mean(dim='member', skipna=False) # get forced component 
        un_forced = ds_model-ds_model_mean
        un_forced_list.append(un_forced)
        std = un_forced.groupby(un_forced.time.dt.month).std(dim=['time', 'member'])
        ds_model = ds_model.groupby(ds_model.time.dt.month)/std
        ds_model_mean = ds_model.mean(dim='member', skipna=False) # get normalized ensemble mean. 
        model_mean_list.append(ds_model_mean.load())
        del ds_model, ds_model_mean #, ds_model_anomaly
        # print time elapse for model
        etime = clocktime.time()
        print()
        print("Time elapsed: " + str(etime - stime) + " seconds")
        print()
    ds_multi_model = xr.concat(model_mean_list, dim='model')
    lat_weights = ds.spatial.get_weights(axis=['Y'])
    un_forced_concat = xr.concat(un_forced_list, dim='member')
    un_forced_std = un_forced_concat.groupby(un_forced_concat.time.dt.month).std(dim=['time', 'member'])

    if ncvar == "tas" or ncvar == "pr" or ncvar == "psl" or ncvar == "monmaxpr" or ncvar == "monmaxtasmax" or ncvar == "monmintasmin":
        maskfile = "nomask.nc"
        missing_data_maskx = xr.open_dataset(maskfile)
        missing_data = np.where(np.isnan(missing_data_maskx.tas.squeeze().transpose('lon', 'lat')), np.nan, 1)
    else:
        maskfile = "maskland.nc"
        missing_data_maskx = xr.open_dataset(maskfile)
        missing_data = np.where(np.isnan(missing_data_maskx.tos.squeeze().transpose('lon', 'lat')), np.nan, 1)
    del maskfile
    #missing_data.shape
    index_array = xr.DataArray([0, 1, 2, 3, 4], dims="model") # change the index to pick up models
    # index_array = xr.DataArray([0], dims="model") # change the index to pick up models
    ds_multi_model_mean5 = ds_multi_model.isel(model=index_array).mean(dim='model', skipna=False)
    ds_multi_model_mean5 = ds_multi_model_mean5.bounds.add_missing_bounds()
    print(ds_multi_model_mean5)
    ds_multi_model_mean5[ncvar] = ds_multi_model_mean5[vid].transpose('time', 'lon', 'lat')

    masked = ds_multi_model_mean5[ncvar] * np.tile(np.expand_dims(missing_data, axis=0), (ds_multi_model_mean5[ncvar].shape[0], 1, 1) )
    ds = xc.open_dataset('/p/lustre3/shiduan/ForceSMIP/Training/Amon/pr/CanESM5/pr_mon_CanESM5_historical_ssp585_r9i1p2f1.188001-202212.nc')
    lat_weights = ds.spatial.get_weights(axis=['Y'])
    solvers = []
    eofs = []
    for month in range(1, 13):
        ds_in = masked.sel(time=masked.time.dt.month==month)
        solver = Eof(ds_in, weights=lat_weights)
        solvers.append(solver)
        eofs.append(solver.eofs(neofs=neofs))
    
    # evaluate 
    # get evaluation models
    eval_dir = root_dir + '/Evaluation-' + eval_tier + '/' + cmipTable[ncvar] + '/' + ncvar
    eval_models = glob.glob(eval_dir + '/*')
    eval_models = [p.split('_')[-1].split('.')[0] for p in eval_models]
    eval_models.sort(key=natsort)
    for model in eval_models:
        print(model)
        # get file for model
        fn = glob.glob(eval_dir + '/*_' + model + '.*.nc')
        print(fn)
        # make sure filename is unique
        if len(fn) != 1:
            raise ValueError("Unexpected number of model members")
        else:
            fn = fn[0]
        # open dataset
        ds = xc.open_dataset(fn)
        ds = ds.bounds.add_missing_bounds(axes=['T'])
        # remove singletons
        ds = ds.squeeze()
        ds = ds.sel(time=slice(tv_time_period[0], tv_time_period[1]))
        
        # ds_climate = ds.groupby(ds.time.dt.month).mean()
        ds = ds.temporal.departures(vid, freq='month', reference_period=reference_period)
        if 'file_qf' in ds.variables:
            ds = ds.drop('file_qf')
        target = ds.copy()
        if stand:
            target = target.groupby(ds.time.dt.month)/un_forced_std
        target = target[vid].transpose('time', 'lon', 'lat') * np.tile(np.expand_dims(missing_data, axis=0), (ds[vid].shape[0], 1, 1) )
        tmp_all = []
        for month in range(1, 13):
            solver = solvers[month-1]
            eof = eofs[month-1]
            target_month = target.sel(time=target.time.dt.month==month)
            pseudo_pcs = solver.projectField(target_month - target_month.mean(dim="time"), neofs=neofs)
            pseudo_pcs_new = pseudo_pcs.isel(mode=0)
            tmp1 = pseudo_pcs_new * eof[0,:,:]
            if neofs>1:
                for n in range(1, neofs):
                    pseudo_pcs_new = pseudo_pcs.isel(mode=n)
                    tmp1 += pseudo_pcs_new * eof[n,:,:]
            tmp_all.append(tmp1)
        tmp1 = xr.concat(tmp_all, dim='time')
        tmp1 = tmp1.sortby('time')
        
        tmp2 = tmp1 / lat_weights
        tmp3 = tmp2.transpose('time', 'lat', 'lon')
        if stand:
            tmp3 = tmp3.groupby(tmp3.time.dt.month)*un_forced_std
            tmp3 = tmp3[vid]
        tmp4 = tmp3 * np.tile( np.expand_dims(np.transpose(missing_data), axis=0), (tmp3.shape[0], 1, 1) )
        tmp4.attrs['_FillValue'] = 1e20
        # tmp4 = tmp4.groupby(tmp4.time.dt.month) + ds_climate[vid]
        tmp4['time'] = tmp1['time']
        output = xr.Dataset({vid: tmp4})
        output = output.fillna(1e20)
        print(output)
        output.to_netcdf(ncvar + '_' + model + '_' + eval_tier + '_MonthFinger_TheFingerprinters.nc')
        