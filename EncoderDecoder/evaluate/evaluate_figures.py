import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import importlib as imp
import netCDF4 as nc
import os
import tensorflow as tf
import keras
from keras.models import load_model
import numpy.linalg as LA
import evaluate_functions as ef
import pandas as pd
import sys
sys.path.insert(1, "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/")
import experiments as exp


data_name = "Train4_Val4_CESM2_all_pr"
model_name = "internal_feature"

complete_name = model_name + "_" + data_name
settings = exp.get_experiment(model_name)
data_settings = exp.data_dictionary[data_name]
datapath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/exp_data/"
predpath = "/barnes-scratch/mafern/ForceSMIP/ForceSMIP/saved_predictions/"
savepath = "../barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/"
# datapath = os.path.abspath("exp_data")+'/'
# predpath = os.path.abspath("saved_predictions")+'/'
# savepath = os.path.abspath("evaluate/figures")+'/'
savefig_name = data_settings['target_variable'] + '_' + settings['target_component']

VARIABLE = data_settings['target_variable']
TIER = data_settings['time_range']
evalPeriods = {
    "Tier1": ("1950-01-01", "2022-12-31"),
    "Tier2": ("1900-01-01", "2022-12-31"),
    "Tier3": ("1979-01-01", "2022-12-31")
}
time_range = evalPeriods[TIER]

sample_n = 1 #used for functions that plot a single sample
PLOT_PRED_EXAMPLE = False
PLOT_AVG_ABS_DIFF = False

PATTERN_CORRELATION = False

SUB_YEAR_N_PERIOD = 44
SUB_PERIOD_TREND = False #"Sub-period trend (pattern correlation and RMSE)"
FULL_PERIOD_TREND = True #"Full-period trend (pattern correlation and RMSE)"

#"Monthly and annual time series of global means and means in latitude bands (correlation and RMSE)"
#"Grid point, monthly & annual time series (global-mean correlation and MSE)"
#"Indices: ENSO, East-West Pacific SST gradient, NAO, SAM, Southern Ocean SST, Arctic Amplification factor (in tas)"

EVALUATE_ENSO = False #SST
EVALUATE_EW_PAC_SST = False #incomplete
EVALUATE_NAO = False #SLP 
EVALUATE_SAM = False #SLP
EVALUATE_SOI = False #incomplete
EVALUATE_AA = False #incomplete

lat = np.linspace(-88.75, 88.75, 72)
lon = np.linspace(1.25, 358.8, 144)
lat_n = np.size(lat)
lon_n = np.size(lon)

lower = plt.cm.RdBu_r(np.linspace(0,.49, 49))
white = plt.cm.RdBu_r(np.ones(2)*0.5)
upper = plt.cm.RdBu_r(np.linspace(0.51, 1, 49))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)

# PFtrain = prediction of forced for training data
# PFtest = predication of forced for evaluation (truth not known) or testing
# PItrain = prediction of internal for training data
# PItest = predication of internal for evaluation (truth not known) or testing
# Ftrain = true forced for training data
# Ftest = true forced for evaluation (truth not known) or testing
# Itrain = true internal for training data
# Itest = true internal for evaluation (truth not known) or testing
f = np.load(predpath + complete_name + str(settings['seed']) + "_preds.npz")
unpacked = [np.nan_to_num(f[key]) for key in f]
PFtrain, PFtest, PItrain, PItest, Ftrain, Ftest, Itrain, Itest = unpacked

# get times and members for reshaping
print(type(data_settings["train_members"]))
n_train_members = np.size(data_settings["train_members"])
n_test_members = np.size(data_settings["test_members"])

if data_settings['month'] == 'annual':
    freq = 'Y'
    time = pd.date_range(start=time_range[0], end=time_range[1], freq=freq)
else:
    freq = 'M'
    time = pd.date_range(start=time_range[0], end=time_range[1], freq=freq)
    time = time[::12] #Added by Charlie to get the right time length when only predicting a single month. 
time_n = np.size(time)

#######################################
# set Truth variable and Prediction variable to the respective variable for the rest of the code to run
Truth = Ftest.squeeze().reshape(n_test_members, time_n, lat_n, lon_n)
# Prediction = PFtest.reshape(n_test_members, time_n, lat_n, lon_n)
Prediction = (Itest+Ftest-PItest).reshape(n_test_members, time_n, lat_n, lon_n)
Full = (Itest+Ftest).squeeze().reshape(n_test_members, time_n, lat_n, lon_n)
PItest = (PItest).squeeze().reshape(n_test_members, time_n, lat_n, lon_n)
Itest = (Itest).squeeze().reshape(n_test_members, time_n, lat_n, lon_n)
PItest = (PItest).squeeze().reshape(n_test_members, time_n, lat_n, lon_n)
Itest = (Itest).squeeze().reshape(n_test_members, time_n, lat_n, lon_n)

Prediction = xr.DataArray(Prediction, dims = ['members','time','lat','lon'])
Prediction["members"] = np.arange(n_test_members)
Prediction["time"] = time
Prediction["lat"] = lat[:]
Prediction["lon"] = lon[:]

Truth = xr.DataArray(Truth, dims = ['members','time','lat','lon'])
Truth["members"] = np.arange(n_test_members)
Truth["time"] = time
Truth["lat"] = lat[:]
Truth["lon"] = lon[:]

Full = xr.DataArray(Full, dims = ['members','time','lat','lon'])
Full["members"] = np.arange(n_test_members)
Full["time"] = time
Full["lat"] = lat[:]
Full["lon"] = lon[:]

PItest = xr.DataArray(PItest, dims = ['members','time','lat','lon'])
PItest["members"] = np.arange(n_test_members)
PItest["time"] = time
PItest["lat"] = lat[:]
PItest["lon"] = lon[:]

Itest = xr.DataArray(Itest, dims = ['members','time','lat','lon'])
Itest["members"] = np.arange(n_test_members)
Itest["time"] = time
Itest["lat"] = lat[:]
Itest["lon"] = lon[:]
PItest = xr.DataArray(PItest, dims = ['members','time','lat','lon'])
PItest["members"] = np.arange(n_test_members)
PItest["time"] = time
PItest["lat"] = lat[:]
PItest["lon"] = lon[:]

Itest = xr.DataArray(Itest, dims = ['members','time','lat','lon'])
Itest["members"] = np.arange(n_test_members)
Itest["time"] = time
Itest["lat"] = lat[:]
Itest["lon"] = lon[:]
#######################################

member = 2
GLOBALMEAN_TIMESERIES = True
if GLOBALMEAN_TIMESERIES:
    Truth_GM = ef.CalcGlobalMean(Itest[member,:,:,:], lat)
    Prediction_GM = ef.CalcGlobalMean(PItest[member,:,:,:], lat) 

    plt.figure()
    plt.plot(Truth_GM, color = "black", label = "truth")
    plt.plot(Prediction_GM, color = "red", label = "prediction")
    plt.legend()
    plt.title("Internal Variability")
    plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/I_Global_TimesSeries_" + savefig_name)

    Truth_GM = ef.CalcGlobalMean(Truth[member,:,:,:], lat)
    Prediction_GM = ef.CalcGlobalMean((Full[member,:,:,:] - Itest[0,:,:,:]), lat) 

    plt.figure()
    plt.plot(Truth_GM, color = "black", label = "truth")
    plt.plot(Prediction_GM, color = "red", label = "Full - T IV")
    plt.legend()
    plt.title("True Forced Response")
    plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/FT_Global_TimesSeries_" + savefig_name)

    Truth_GM = ef.CalcGlobalMean(Truth[member,:,:,:], lat)
    Prediction_GM = ef.CalcGlobalMean((Full[member,:,:,:] - PItest[0,:,:,:]), lat) 

    plt.figure()
    plt.plot(Truth_GM, color = "black", label = "truth")
    plt.plot(Prediction_GM, color = "red", label = "Full - P IV")
    plt.legend()
    plt.title("Predicted Forced Response")
    plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/FP_Global_TimesSeries_" + savefig_name)

    Truth_GM = ef.CalcGlobalMean(Truth[member,:,:,:], lat)
    Prediction_GM = ef.CalcGlobalMean(Prediction[0,:,:,:], lat) 
    Full_GM = ef.CalcGlobalMean(Full[member,:,:,:], lat)

    plt.figure()
    plt.plot(Truth_GM, color = "black", label = "truth")
    plt.plot(Prediction_GM, color = "red", label = "prediction")
    plt.plot(Full_GM, color = "green", label = "full")
    plt.legend()
    plt.title("Forced Response")
    plt.savefig("/barnes-scratch/mafern/ForceSMIP/ForceSMIP/evaluate/figures/F_Global_TimesSeries_" + savefig_name)

    #############
sys.exit(0)

if EVALUATE_SAM:
    for member in range(n_test_members):
        Pred = Prediction[member]
        Tru = Truth[member]
        SAM_truth, eigenvectors = ef.CalcSAMIndex(lat, lon, Tru, None)
        SAM_pred, eigenvectors= ef.CalcSAMIndex(lat, lon, Pred, eigenvectors)
        
        plt.figure(figsize = (10,6))
        ef.PlotLines(time, SAM_pred, "time", "SAM index", "", "black", "Prediction", None, None)
        ef.PlotLines(time, SAM_truth, "time", "SAM index", "", "red", "Truth", None, None)
        plt.savefig(savepath + str(member) + "_SAM_" + savefig_name)

if EVALUATE_NAO:
    for member in range(n_test_members):
        Pred = Prediction[member]
        Tru = Truth[member]
        NAO_pred = ef.CalcNAOIndex(lat, lon, Pred)
        NAO_truth = ef.CalcNAOIndex(lat, lon, Tru)

        plt.figure(figsize = (10,6))
        ef.PlotLines(time, NAO_pred, "time", "NAO index", "", "black", "Prediction", None, None)
        ef.PlotLines(time, NAO_truth, "time", "NAO index", "", "red", "Truth", None, None)
        plt.savefig(savepath + str(member) + "_NAO_" + savefig_name)

if EVALUATE_ENSO:
    ENSO_pred = ef.CalcENSOIndex(lat, lon, Prediction)
    ENSO_truth = ef.CalcENSOIndex(lat, lon, Truth)
    
    plt.figure(figsize = (10,6))
    ef.PlotLines(time, ENSO_pred, "time", "ENSO index", "", "black", "Prediction", None, None)
    ef.PlotLines(time, ENSO_truth, "time", "ENSO index", "", "red", "Truth", None, None)
    plt.savefig(savepath + 'ENSO_' + savefig_name)

if FULL_PERIOD_TREND:
    # for member in range(n_members):
    for member in range(n_test_members):
        Pred = Prediction[member]
        Tru = Truth[member]
        Ful = Full[member]

        Truth_Trend = np.empty(shape = (72,144))
        Predicted_Trend = np.empty(shape = (72,144))
        Full_Trend = np.empty(shape = (72, 144))
        x_range = np.arange(0,time_n,1)

        for la in np.arange(0, lat_n, 1):
            for lo in np.arange(0, lon_n, 1):
                values = np.polyfit(x_range, Tru[:,la,lo], 1)
                Truth_Trend[la,lo] = values[0] * time_n

                values = np.polyfit(x_range, Pred[:,la,lo], 1)
                Predicted_Trend[la,lo] = values[0] * time_n

                values = np.polyfit(x_range, Ful[:,la,lo], 1)
                Full_Trend[la,lo] = values[0] * time_n

        # Plot_Gobal_Map(Truth_Trend, "Spatial trend of Truth", -.2, .2, tmap, "")
        # Plot_Gobal_Map(Predicted_Trend, "Spatial trend of Prediction", -.2, .2, tmap, "")
        # Plot_Gobal_Map(Predicted_Trend-Truth_Trend, "Difference between truth and prediction", -.2, .2, tmap, "")
        weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lat)),]*144))
        RMSE = np.sum((weighted_difference)**2/time_n)
        PC = ef.CalcPatternCorrelation(Truth_Trend, Predicted_Trend)

        axs = ef.Plot2b2_colormesh_o3(Truth_Trend, Predicted_Trend, Predicted_Trend-Truth_Trend, Full_Trend, lat, lon, -2, 2)
        # axs.text(430, 8, "The weighted RMSE is: " +str(round(RMSE, 2)), fontsize = 10)
        # axs.text(430, 30, "The weighted Pattern Correlation is: " +str(round(PC)), fontsize = 10)

        print(round(PC))

        plt.savefig(savepath + str(member) + "_Full_trend_" + savefig_name)

if SUB_PERIOD_TREND:
    for member in range(n_test_members):
        Pred = Prediction[member]
        Tru = Truth[member]

        sub_time = np.arange(0, time_n, SUB_YEAR_N_PERIOD)
        x_range_sub = np.arange(0,SUB_YEAR_N_PERIOD,1)

        for s, sub in enumerate(sub_time[:-1]):
            Truth_Trend = np.empty(shape = (72,144))
            Predicted_Trend = np.empty(shape = (72,144))
            for la in np.arange(0, lat_n, 1):
                for lo in np.arange(0, lon_n, 1):
                    values = np.polyfit(x_range_sub, Tru[sub:sub_time[s+1],la,lo], 1)
                    Truth_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

                    values = np.polyfit(x_range_sub, Pred[sub:sub_time[s+1],la,lo], 1)
                    Predicted_Trend[la,lo] = values[0] * SUB_YEAR_N_PERIOD

            # Plot_Gobal_Map(Predicted_Trend-Truth_Trend, "Difference between truth and prediction", -1, 1, tmap, "")

            weighted_difference = (Predicted_Trend-Truth_Trend) * np.transpose(np.array([np.cos(np.deg2rad(lat)),]*144))
            RMSE = np.sum((weighted_difference)**2/SUB_YEAR_N_PERIOD)
            PC = ef.CalcPatternCorrelation(Truth_Trend[:,:], Predicted_Trend[:,:]) 

            axs = ef.Plot2b2_colormesh_o3(Truth_Trend, Predicted_Trend, Predicted_Trend-Truth_Trend, lat, lon, -.2, .2)
            axs.text(430, 8, "The weighted RMSE is: " +str(round(RMSE, 2)), fontsize = 10)
            axs.text(430, 30, "The weighted Pattern Correlation is: " +str(round(PC)), fontsize = 10)

            plt.savefig(savepath + str(member) + "_Sub_trend_" + savefig_name)

if PLOT_PRED_EXAMPLE:
    ef.Plot_Gobal_Map(lat, lon, Prediction[sample_n,0,:,:], "Example of a Prediction", Prediction[sample_n,0,:,:].min(), Prediction[sample_n,0,:,:].max(), "Reds", "")
    plt.savefig(savepath + "sample_" + savefig_name)

if PLOT_AVG_ABS_DIFF:
    Difference = np.mean(np.abs(Prediction[sample_n,:,:,:] - Truth[sample_n,:,:,:]), axis = 0)
    # Plot_Gobal_Map(plot_data, title, min, max, colorbar, colorbar_title)
    ef.Plot_Gobal_Map(lat, lon, Difference, "Average Absolute Difference between \nTruth and Prediction", Difference.min(), Difference.max(), tmap, "")
    plt.savefig(savepath + "sample_difference_" + savefig_name)
    
print("DONE!")


