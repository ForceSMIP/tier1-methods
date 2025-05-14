# imports
import calendar
import numpy as np
from scipy import signal
import xcdat as xc
import xarray as xr
from sklearn.linear_model import LinearRegression
from PyEMD import EEMD
import scipy
import re


def get_predictors(fnp, period, ignore=[]):
    """
    preds = get_predictors(fnp, period, ignore=[])

    Gets pre-defined forced change predictors and places them into a matrix
    [n, time] for a user-defined period. The predictors are listed below in the
    notes.

    Parameters:
    -----------
    fnp (str) : filename for predictor time series file (e.g., '/home/predictors.nc')
    period (slice) : period with which to extract predictors (e.g., slice('1880-01-01', '2022-12-31'))
    ignore (optional, List) : optional list of predictors to exclude (should be among ghg,
                              aerosol_NorthAmerica, aerosol_Europe, aerosol_EastAsia, aerosol_SouthAsia,
                              aerosol_Africa, solar, volcanic_krakatoa, volcanic_santamaria,
                              volcanic_novarupta, volcanic_agung, volcanic_chichon, volcanic_pinatubo)

    Returns:
    --------
    np.array : array of predictors [predictor, time]

    Notes:
    ------
    The predictors include:
        * Global-mean DAMIP-derived tas timeseries for:   GHG
                                                          Solar

        * Global-mean, multimodel mean DAMIP volcanic tas time series that
          have zeros except for the 4-year window starting at the volcanic
          eruption for six major eruptions:
                Krakatoa (May 1883)
                Santa Maria (October 1902)
                Novarupta (June 1912)
                Mt. Agung (March 1963)
                El Chichon (April 1982)
                Mt. Pinatubo (June 1992)

        * Sulfate Aerosol Emission Timeseries for the following regions:
            * North America
            * Europe
            * East Asia
            * South Asia
            * Africa
    
    Future work could consider dropping some of the aerosol emission time series,
    which are correlated, and/or including different emissions time series
    (e.g., black carbon). Note the file is created with wrangle.py and spans
    1850 - 2022.

    Function drops predictors that are all zeros in time period.

    Aerosol emissions timeseries from inputs4mips (see doi: 10.5194/gmd-2017-43)
    and regions boxes are defined in doi: 10.5194/gmd-16-4451-2023.
    """
    # open predictor file
    dsp = xc.open_dataset(fnp)
    # subset by time period
    dsp = dsp.sel(time=period)
    # create predictor list
    preds = []
    # loop over all predictors in file and extract
    for v in dsp.data_vars:
        # ignore predictors selected to ignore
        if v in ignore:
            continue
        # get predictor time series
        ts = dsp[v]
        # if all values are zero for time period,
        # drop predictor
        if np.alltrue(ts.values == 0):
            continue
        # append predictor to list
        preds.append(ts)
    # create numpy array
    preds = np.array(preds)
    # close predictor file
    dsp.close()
    return preds


def get_forced_prediction(fn, vid, period, preds, ref_period, dpathout, dpath):
    """
    get_forced_prediction(fn, vid, model, period, preds, ref_period, dpathout, dpath)

    Function takes in prediction parameters and makes a prediction of the forced timeseries
    [time, lat, lon] or [time, plev, lat]. The prediction is saved to a NetCDF file.
    
    Parameters:
    -----------
    fn (str) : target file (with which to extract forced response)
    vid (str) : variable id of interest
    model (str) : model name
    period (slice) : period with which to extract predictors (e.g., slice('1880-01-01', '2022-12-31'))
    preds (np.array) : prediction matrix [n_preds, time] with time aligned to period of interest
    ref_period (tuple(str)) : start / stop time for reference period (for departure calculations)
    dpathout : directory where predictions should be stored
    dpath : directory for tas data files

    Notes:
    ------
    The function relies on ForceSMIP standard directory structure.

    The prediction model attempts to fit each grid point using all the forced predictors (see
    get_predictors) and the de-trended time series of Nino 3.4 with 3- and 6-month lead times.
    The fit is performed on a monthly basis (i.e., 12 linear models for each grid point). The
    forced prediction is then made using all of the forced predictors (ignoring the Nino 3.4 indices).
    """
    # get corresponding tas time series
    if 'Evaluation-Tier' in fn:
        fntas = dpath + '/Amon/tas/' + '/' + fn.split('/')[-1]
        v = fn.split('/')[-2]
    else:
        fntas = dpath + '/Amon/tas/' + fn.split('/')[-2] + '/' + fn.split('/')[-1]
        v = fn.split('/')[-3]
    fntas = fntas.replace(v, 'tas')
    fntas = fntas.replace('_day_', '_mon_')
    # get de-trended Nino3.4 time series from tas file
    tsn34 = get_n34_timeseries(fntas, period, ref_period)
    # get 3- and 6-month lead Nino3.4 values
    tsn34l = np.zeros((2, len(tsn34)))
    tsn34l[0, 3:] = tsn34[0:-3].values
    tsn34l[1, 6:] = tsn34[0:-6].values
    # open target file
    ds = xc.open_dataset(fn)
    # subset by time period
    ds = ds.sel(time=period)
    # do forced prediction
    YP = do_monthly_prediction(ds, vid, preds, tsn34l, ref_period)
    # generate output filename (from input file)
    fnOut = dpathout + fn.split('/')[-1]
    ds.close()
    # save prediction to output file
    YP.to_netcdf(fnOut)


def get_n34_timeseries(fn, period, ref_period):
    """
    tsn34 = get_n34_timeseries(fn, period)

    Function gets a data array of the de-trended Nino 3.4 timeseries.
    The data is de-trended using the two lowest frequency modes derived
    from extended empirical model decomposition.

    Parameters:
    -----------
    fn (str) : filename of tas file
    period (slice) : period with which to extract timeseries (e.g., slice('1880-01-01', '2022-12-31'))
    ref_period (tuple(str)) : start / stop time for reference period (for departure calculations)
    """
    # open tas file
    dsn34 = xc.open_dataset(fn)
    # subset to period of interest
    dsn34 = dsn34.sel(time=period)
    # get average for nino3.4 region
    dsn34 = dsn34.spatial.average('tas', lat_bounds=(-5, 5), lon_bounds=(190, 240))
    # calculate departures
    dsn34 = dsn34.bounds.add_missing_bounds('T')
    dsn34 = dsn34.temporal.departures('tas', freq='month', reference_period=ref_period)
    # calculate EEMD
    emd = EEMD()
    ts = emd(dsn34.tas.values)
    # remove 2-lowest order modes from nino3.4 time series
    tsn34 = dsn34['tas']
    tsn34[:] = dsn34.tas - np.sum(ts[-2:], axis=0)
    return tsn34

def natural_sort(l):
    """
    Convenience function to sort a list of strings "naturally" (e.g., r1i1p1f1, r2i1p1f1).

    Notes:
    ------
    https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def prepare_matrices(ds, v, preds, n34, ref_period=('1900-01-01', '1949-12-31')):
    """
    X, Y, XM, XS, YM, YS, naninds = prepare_matrices(ds, v, preds, n34, ref_period)

    Function takes in the target data, forced predictors, and leading Nino3.4 predictors,
    and creates standardized predictor (X) and predictand (Y) matrices.

    Parameters:
    -----------
    ds (xr.Dataset) : target dataset (with which to extract forced response)
    vid (str) : target variable id
    preds (np.array) : array of forced predictor [n_pred, time]
    n34 (np.array) : array of leading nino3.4 predictors [2, time] (or None if it should be excluded)
    ref_period (tuple(str), optional) : start / stop time for reference period (for departure calculations)

    Returns:
    --------
    X (np.array) : predictor matrix (n_samples, n_features)
    Y (np.array) : predictand matrix (n_samples, n_targets)
    XM (np.array) : predictor matrix mean (n_features)
    XS (np.array) : predictand matrix mean (n_targets)
    YM (np.array) : predictor matrix standard deviation (n_features)
    YS (np.array) : predictand matrix standard deviation (n_targets)
    naninds (np.array) : array of non-nan indices
    """
    # check if Nino3.4 is used
    if n34 is not None:
        # if so, place into predictor matrix [2, time]
        # check to make sure size is as expected
        if n34.shape[0] == 2:
            X = np.concatenate((preds, n34), axis=0)
        else:
            ValueError('Unexpected Nino3.4 index shape')
    else:
        # if not, simply return the forced predictors as-is
        X = preds
    # prepare X matrix
    X = np.reshape(X, (X.shape[0], -1)) # reshape to [features, time]
    XM = np.expand_dims(np.mean(X, axis=1), axis=1) # compute mean
    XS = np.expand_dims(np.std(X, axis=1), axis=1) # compute standard deviation
    X = (X - XM) / XS # standardize
    X = X.T # transpose to [time, features]
    # prepare Y matrix
    # get departures
    ds = ds.bounds.add_missing_bounds(['T']) # add time bounds if needed
    ds = ds.temporal.departures(v, reference_period=ref_period, freq='month') # compute departures
    Y = ds[v].to_numpy() # cast to numpy
    Y = np.reshape(Y, (Y.shape[0], -1)) # reshape to [time, spatial targets]
    YM = np.expand_dims(np.mean(Y, axis=0), axis=0) # compute mean
    YS = np.expand_dims(np.std(Y, axis=0), axis=0) # compute standard deviation
    Y = (Y-YM) / YS # standardize
    # get non-nan values in predictand matrix
    YNAN = np.mean(Y, axis=0)
    naninds = np.where(~np.isnan(YNAN))[0]
    # subset predictand matrix to exclude nan values
    Y = Y[:, naninds]
    return X, Y, XM, XS, YM, YS, naninds


def do_monthly_prediction(ds, vid, preds, n34, ref_period):
    
    """
    yp = do_monthly_prediction(ds, vid, preds, n34, ref_period)

    Function takes in the target data, forced predictors, and leading Nino3.4 predictors,
    creates standardized predictor (X) and predictand (Y) matrices and fits the target dataset on a
    monthly basis. A dataset with the forced prediction is returned.

    Parameters:
    -----------
    ds (xr.Dataset) : target dataset (with which to extract forced response)
    vid (str) : target variable id
    preds (np.array) : array of forced predictor [n_pred, time]
    n34 (np.array) : array of leading nino3.4 predictors [2, time] (or None if it should be excluded)
    ref_period (tuple(str)) : start / stop time for reference period (for departure calculations)

    Returns:
    --------
    xr.Dataset : dataset with the forced response
    """
    # get predictor / predictand matrices
    X, Y, XM, XS, YM, YS, naninds = prepare_matrices(ds, vid, preds, n34, ref_period=ref_period)
    # check if nino3.4 predictors are included
    if n34 is not None:
        n34 = True
    # pre-allocate output matrices
    YP = np.zeros((Y.shape[0], YM.shape[1])) * np.nan # forced
    YPE = np.zeros((Y.shape[0], YM.shape[1])) * np.nan # enso
    # loop over 12 months
    for i in range(12):
        # subset X / Y into monthly predictor matrices
        Xm = X[i::12, :]
        Ym = Y[i::12, :]
        # fit monthly X / Y matrices
        reg = LinearRegression().fit(Xm, Ym)
        # forced response should exclude nino 3.4 predictors (if applicable)
        if n34:
            # get coefficients for forced / enso components
            beta = reg.coef_[:, 0:-2]
            beta_e = reg.coef_[:, -2:]
            # compute forced / unforced components
            yp1 = np.sum(np.expand_dims(Xm[:, 0:-2].T, axis=2) * np.expand_dims(beta.T, axis=1), axis=0) + np.expand_dims(reg.intercept_, axis=0)
            yp1e = np.sum(np.expand_dims(Xm[:, -2:].T, axis=2) * np.expand_dims(beta_e.T, axis=1), axis=0)
        else:
            # if nino 3.4 isn't included, total prediction is forced response
            yp1 = reg.predict(Xm)
        # allocate monthly output matrices
        yp = np.zeros((yp1.shape[0], YP.shape[1]))*np.nan
        ype = np.zeros((yp1.shape[0], YP.shape[1]))*np.nan
        # place data into non-nan points
        yp[:, naninds] = yp1
        # un-standardize prediction
        yp = yp * YS + YM
        # place prediction into monthly array
        YP[i::12] = yp
        # do the same for the unforced component (if applicable)
        if n34:
            ype[:, naninds] = yp1e
            ype = ype * YS
            YPE[i::12] = ype
    # reshape data to original form
    YP = np.reshape(YP, ds[vid].shape)
    # place data into a datarray
    YP = xr.DataArray(data=YP,
                      dims=ds[vid].dims,
                      coords=ds[vid].coords,
                      name=vid)
    # create dataset
    YP = YP.to_dataset()
    # if nino 3.4 predictors are used, include in dataset
    if n34:
        YPE = np.reshape(YPE, ds[vid].shape)
        YPE = xr.DataArray(data=YPE,
                          dims=ds[vid].dims,
                          coords=ds[vid].coords)
        YP[vid + '_nat'] = YPE
    # add bounds for convenience
    YP = YP.bounds.add_missing_bounds(['T', 'X', 'Y'])
    return YP

def get_decimal_time(time):
    """
    Function takes an array of cftime objects and converts the
    array to decimal year.

    Parameters:
    -----------
    time (array(cftime object)) : array of cftime objects

    Returns:
    --------
    np.array : array of decimal year values
    """
    # create output list
    timeOut = []
    # loop over time steps
    for t in time:
        # get current step data
        year = t.year
        doy = t.dayofyr
        h = t.hour
        m = t.minute
        s = t.second
        # get total seconds in year
        leap = 1 if calendar.isleap(t.year) else 0
        siy = (365+leap)*24*60*60
        # get seconds at time point
        svalue = s + m*60 + h*3600 + doy*24*3600
        # compute decimal time
        # year + fractional seconds of year
        timeOut.append(year + svalue/siy)
    return np.array(timeOut)

def extend_timeseries(da, end_year=2022, last_years=10):
    """
    da_extended = extend_timeseries(da, end_year=2022, last_years=10)

    Function linearly extrapolates a dataarray forward in time.

    Parameters:
    -----------
    da (xr.DataArray) : dataarray of time series to be extended
    end_year (int) : desired end year
    last_years (int) : number of years at the end of the time series to fit
                       for extrapolation

    Returns:
    --------
    xr.DataArray : dataarray with time series extended to end_year

    Notes:
    ------
    Assumes input time series is monthly.
    """
    # first extend time
    time = da.time.values  # get time objects
    time_obj = type(time[0])  # get cftime type
    time_out = [t for t in time[:-1]]  # get all but last time value
    # iteratively extend time series month-by-month
    # until we reach December of end year
    t = time[-1]
    while t.year <= end_year:
        # get current time / month
        y = t.year
        m = t.month
        # if current month is december, next month is 1
        # else add one to the month
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
        # append time to output time array
        time_out.append(t)
        # advance timestep
        t = time_obj(y, m, 15)
    # to array
    time_out = np.array(time_out)
    # now regress over last n years to get trend
    dtime = get_decimal_time(da.time.values)
    m, b = np.polyfit(dtime[-12*last_years:], da.values[-12*last_years:], 1)
    # project time series forward with linear relationship
    dtime_forward = get_decimal_time(time_out[len(dtime):])
    ts_forward = dtime_forward*m + b
    # concatenate previous / projected data together
    ts = np.concatenate((da.values, ts_forward))
    # form a dataarray
    xda = xr.DataArray(data=ts,
                       dims=['time'],
                       coords={'time': time_out},
                       name='tas')
    return xda
        
def smooth(ts, months=120, order=3):
    """
    ts_smooth = smooth(ts, months=120, order=3)

    Function smooths a time series using a Savitzky-Golay filter.

    Parameters:
    -----------
    ts (array) : array to be smoothed
    months (int, optional) : window size (in months) used for filtering (default 120)
    order (int, optional) : order of fitted polynomial (default 3)

    Returns:
    --------
    array : array of smoothed data

    Notes:
    ------
    Function uses scipy signal toolbox (savgol_filter)

    """
    # smooth data with Savitzky-Golay filter
    tss = signal.savgol_filter(ts,
                       months, # window size used for filtering
                       order), # order of fitted polynomial
    ts[:] = tss[0]
    return ts
