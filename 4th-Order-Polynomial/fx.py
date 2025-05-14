import xarray as xr

def verify_monthly_times_match(time1, time2):
    """
    verify_monthly_times_match(time1, time2)

    Function verifies that two monthly time axes match
    (for year/month – not day).

    Parameters:
    -----------
    time1 (xr.DataArray) : first time axis
    time2 (xr.DataArray) : second time axis

    Returns:
    --------
    Boolean True if each time axis matches, else False
    """
    # check if lengths are the same
    if len(time1) != len(time2):
        return False

    # iterate over each value and ensure year/month matches
    for i in range(len(time1)):
        # get time steps
        t1 = time1.values[i]
        t2 = time2.values[i]
        # check for match, if match, keep going
        # else, return False
        if ((t1.month == t2.month) & (t1.year == t2.year)):
            continue
        else:
            return False
    return True

def get_days_in_year(t):
    year = t.year
    dtype = type(t)
    diy = 0
    for m in range(1, 13):
        diy += dtype(year, m, 1).daysinmonth
    return diy

def get_decimal_year_time(time):
    # create empty output list
    dtime = []
    for t in time.values:
        # get time components
        year = t.year
        doy = t.dayofyr
        hod = t.hour
        mod = t.minute
        sod = t.second
        msod = t.microsecond
        # partial/fractional day
        pd = hod/24 + mod/(24*60) + sod/(24*60*60) + msod/(24*60*60*1000)
        # numerator is the day of year plus fractional day
        num = doy + pd
        # denominator
        diy = get_days_in_year(t)
        # take ratio to get fractional year
        fy = num / diy
        # append data to list
        dtime.append(year + fy)
    # convert to dataarray
    dtime = xr.DataArray(
        data=dtime,
        dims=["time"],
        coords=dict(time=dtime),
        attrs=dict(
            description="decimal time",
            units="year",
            axis='T'
        ),
    )
    return dtime