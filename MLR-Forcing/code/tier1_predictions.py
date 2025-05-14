# %% imports
import glob
from fx import get_predictors, get_forced_prediction
import os
from joblib import Parallel, delayed

# %% parameters
dpath = '/p/user_pub/climate_work/pochedley1/ForceSMIP/Evaluation-Tier1/'
vdict = {'tas': ['tas', 'Amon'],
         'pr': ['pr', 'Amon'],
         'psl': ['psl', 'Amon'],
         'tos': ['tos', 'Omon'],
         'monmaxpr': ['pr', 'Aday'],
         'monmaxtasmax': ['tasmax', 'Aday'],
         'monmintasmin': ['tasmin', 'Aday'],
         'zmta': ['ta', 'Amon'],}
dpathout = '/p/user_pub/climate_work/pochedley1/forcesmip/tier1/'
fnp = '/p/user_pub/climate_work/pochedley1/forcesmip/damip/predictors.nc'
period = slice('1950-01-01', '2022-12-31')
ref_period = ('1950-01-01', '1980-12-31')

# %% get predictors
preds = get_predictors(fnp, period)

# %% loop over variables
for key in vdict.keys():
    # get metadata for variable
    print(key)
    vid = vdict[key][0]
    table = vdict[key][1]
    # get evaluation files
    files = glob.glob(dpath + table + '/' + key + '/*nc')
    # loop over files and do prediction
    for fn in files:
        member = fn.split('/')[-1].split('_')[2].split('.')[0]
        print('   ' + member)
        get_forced_prediction(fn, vid, period, preds, ref_period, dpathout, dpath)

# rename files
files = glob.glob(dpathout + '*nc')
tier = 'tier1'
method = 'mlr'
groupname = 'pochedley'
for fn in files:
    vid = fn.split('/')[-1].split('_')[0]
    mid = fn.split('/')[-1].split('_')[2].split('.')[0]
    fnout = dpathout + ('_').join([vid, mid, tier, method, groupname]) + '.nc'
    os.rename(fn, fnout)

