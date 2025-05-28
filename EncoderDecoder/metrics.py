import numpy as np

def MAE(Dtrue, Dpred, weights = None):
    
    metrics_arr = np.abs(Dtrue - Dpred)
    nonanbool = ~np.isnan(metrics_arr)
    if weights is not None:
        weights = weights * np.ones_like(nonanbool)
        weights = weights[nonanbool]
        if len(weights) == 0:
            weights = None
    metrics_arr = metrics_arr[nonanbool]
    return np.average(metrics_arr, weights=weights)


def MSE(Dtrue, Dpred, weights = None):
    metrics_arr = np.square(Dtrue - Dpred)
    nonanbool = ~np.isnan(metrics_arr)
    if weights is not None:
        weights = weights * np.ones_like(nonanbool)
        weights = weights[nonanbool]
        if len(weights) == 0:
            weights = None
    metrics_arr = metrics_arr[nonanbool]
    return np.average(metrics_arr, weights=weights)

# pattern correlation
def R2(Dtrue, Dpred, weights=1):
    if weights is None:
        weights = 1
    eps = np.finfo(float).eps
    ss_res = np.nansum(np.square(Dtrue-Dpred)*weights) # sum of squares of the residual
    ss_tot = np.nansum(np.square(Dtrue-np.nanmean(Dtrue))*weights) # total sum of squares
    return ( 1 - ss_res/(ss_tot + eps) )