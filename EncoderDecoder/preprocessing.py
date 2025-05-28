machine = 'asha' # 'casper' or 'asha'

# I/O / data wrangling
import glob
import numpy as np
import xarray as xr
import gc
import os

if machine == 'casper':
    root_dir = "/glade/campaign/cgd/cas/asphilli/ForceSMIP/"
elif machine == 'asha':
    root_dir = "/barnes-scratch/DATA/ForceSMIP/"

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
evalPeriods = {
    "Tier1": ("1950-01-01", "2022-12-31"),
    "Tier2": ("1900-01-01", "2022-12-31"),
    "Tier3": ("1979-01-01", "2022-12-31"),
}
fullmems = {
    "MIROC6": 50,
    "CESM2": 50,
    "CanESM5": 25,
    "MPI-ESM1-2-LR": 30,
    "MIROC-ES2L": 30,
}

# load an individual member
def load_model(file, var, timecut="Tier1", month="annual"):
    # time period to cut on
    timebds = evalPeriods[timecut]
    ds = xr.open_dataset(file)
    # select the variable
    varin = ds[cmipVar[var]]
    # get the correct time period
    varcut = varin.sel(time=slice(timebds[0],timebds[1]))
    if month == 'annual':
        varcut = varcut.groupby('time.year').mean()  
    else:
        month_idxs=varcut.groupby('time.month').groups[int(month)]
        varcut=varcut.isel(time=month_idxs)
    return varcut

# load all members from a model
def load_models(model, var, timecut="Tier1", ntrainmems=20, month="annual"):
    print(root_dir, cmipTable[var], var, model)
    # get the list of all files (members) for this model and variable
    filelist = root_dir + "Training/" + cmipTable[var] + "/" + var + "/" + model + "/" + var + "*.nc"
    filelist = np.array(glob.glob(filelist))
    sorted = np.argsort([float(filelist[ii][filelist[ii].find('_r')+2:filelist[ii].find('i1')]) for ii in range(len(filelist))])
    filelist = filelist[sorted]
    
    # go through each file and load, append
    all_ens = []
    for ifile, file in enumerate(filelist[:ntrainmems]):
        varcut = load_model(file, var, timecut, month)
        # add coordinate for concatenating after the loop
        varcut = varcut.assign_coords({"variant":ifile+1})
        # append into a list
        all_ens.append(varcut)
    # get all members into a single xarray data array
    all_ens = xr.concat(all_ens, dim="variant")
    return all_ens

def make_data(models=["CESM2", "MIROC6", "CanESM5"], var="tos", timecut="Tier1", mems=np.arange(20), month="annual"):
    """     
    mems: a list of members to use
    """
    # number of members requested
    nmems = len(mems)

    # go through each model separately (targets are model specific)
    for imod, model in enumerate(models):
        # get the number of members available for this model
        nfullmems = fullmems[model]
        # check that we can load the number of members requested
        assert nmems <= nfullmems, f"more members requested ({nmems}) than available ({nfullmems})"
        # load the members for this model/variable into an xarray
        # returns the correct time period given the tier
        # note it loads all members, so the forced response is correct, then picks members based on mems
        da = load_models(model, var, timecut, nfullmems, month=month)
        # convert from xarray to a numpy array: dimensions are [members, time, lat, lon]
        da_np = np.asarray(da)
        # get the forced response as the ensemble mean across all members (0th dimension)
        da_f_np = np.mean(da_np, axis=0, keepdims=True)
        # internal variability is the difference between the full signal and the forced response
        # only use the requested members
        da_i_np = da_np[mems] - da_f_np
        # reshape these to dimensions: [samples, lat, lon]
        shape2 = da_np.shape[2]
        shape3 = da_np.shape[3]
        da_np = np.reshape(da_np[mems], (-1, shape2, shape3))
        # for forced response, repeat nmems times to match full and internal
        da_f_np = np.reshape([da_f_np]*nmems, (-1, shape2, shape3))
        da_i_np = np.reshape(da_i_np, (-1, shape2, shape3))
        # stack the models together        
        if imod == 0:
            Xfull = da_np
            Yforced = da_f_np
            Yinternal = da_i_np
        else:
            Xfull = np.vstack([Xfull, da_np])
            Yforced = np.vstack([Yforced, da_f_np])
            Yinternal = np.vstack([Yinternal, da_i_np])
    
    return Xfull, Yforced, Yinternal


def save_npz(settings, A_train, F_train, I_train, A_val, F_val, I_val, A_test, F_test, I_test):
    os.system('mkdir ' +  settings['npz_dir'])
    np.savez(settings['npz_dir'] + settings['data_name'] + '.npz', Atr=A_train, Ftr=F_train, Itr=I_train, 
             Ava=A_val, Fva=F_val, Iva=I_val, Ate=A_test, Fte=F_test, Ite=I_test)

def load_npz(settings):
    npzdat = np.load(settings['npz_dir'] + settings['data_name'] + '.npz')
    Atr, Ftr, Itr, Ava, Fva, Iva, Ate, Fte, Ite = (npzdat['Atr'], npzdat['Ftr'], npzdat['Itr'],
                                                        npzdat['Ava'], npzdat['Fva'], npzdat['Iva'],
                                                        npzdat['Ate'], npzdat['Fte'], npzdat['Ite'],)
    return Atr, Ftr, Itr, Ava, Fva, Iva, Ate, Fte, Ite
    
def stack_variable(X_tuple):
    Xout = np.stack(X_tuple, axis=-1)
    return Xout

def make_all_eval_mem(var, timecut, month):
    # evaluation members are by tier number and letter
    letters = "ABCDEFGHIJ"
    tiernum = timecut[-1]
    evalmems = [tiernum + letter for letter in letters]
    # for each evaluation member, load that data
    mem_var_arrays = []
    for evalmem in evalmems:
        mem_var_arrays.append(make_eval_mem(evalmem=evalmem, var=var, timecut=timecut, month=month))
    eval_array = np.concatenate(mem_var_arrays, axis=0)
    return eval_array

def make_eval_mem(evalmem="1H", var="tos", timecut="Tier1", month="annual"):
    # "evalmem can be 1A through 1J, so determines the member"
    # get a member from one of the evaulation data sets
    filelist = root_dir + "Evaluation-"+timecut+"/" + cmipTable[var] + "/" + var + "/" + var + "*" + evalmem + "*.nc"
    # glob the full file name (zero index to remove from list)
    file = glob.glob(filelist)[0]
    varmean = load_model(file, var, timecut="Tier1", month=month)
    # return as a numpy array    
    return np.asarray(varmean)

