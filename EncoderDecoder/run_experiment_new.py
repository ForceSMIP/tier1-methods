import argparse
import numpy as np
import experiments as exp
import tensorflow as tf
import random
import os
from main_functions import preprocess, prep_data_for_training, train_and_predict, calculate_metrics, save_experiment_specs


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# set up parser (run via command line, e.g. python3 run_experiment.py exp_name)
PARSER = argparse.ArgumentParser()
PARSER.add_argument('exp_name', type=str)
ARGS = PARSER.parse_args()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# get experiment settings
# assert np.isin(ARGS.exp_name, exp.experiments.keys()), "no experiment with the name" + ARGS.exp_name + ", current experiments:" + exp.experiments.keys()
settings = exp.get_experiment(ARGS.exp_name)

os.system('mkdir ' + settings["npz_dir"])
os.system('mkdir ' + settings["pred_dir"])
os.system('mkdir ' + settings["exp_specs_dir"])
os.system('mkdir ' + settings["tune_specs_dir"])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# set up random seeds
tf.keras.backend.clear_session()
np.random.seed(settings["seed"])
random.seed(settings["seed"])
tf.random.set_seed(settings["seed"])

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# pre-process data
Atrain, Ftrain, Itrain, Aval, Fval, Ival, Atest, Ftest, Itest  = preprocess(settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# prepare the data for training (deal with NaNs, standardize data)
Xtrain_stand, Xval_stand, Xtest_stand, \
    Ttrain_stand, Tval_stand, Ttest_stand, \
    Ttrain_mean, Ttrain_std, Tval_mean, Tval_std, Ttest_mean, Ttest_std = prep_data_for_training(Atrain, Ftrain, Itrain,
                                                                                                 Aval, Fval, Ival,
                                                                                                 Atest, Ftest, Itest, settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# build model, train, save, make predictions
PFtrain, PFval, PFtest, PItrain, PIval, PItest = \
    train_and_predict(Atrain, Aval, Atest,
                      Ftrain, Itrain, Ftest, Itest,
                      Xtrain_stand, Xval_stand, Xtest_stand,
                      Ttrain_stand, Tval_stand, Ttest_stand,
                      Ttrain_mean, Ttrain_std, 
                      Tval_mean, Tval_std, 
                      Ttest_mean, Ttest_std,
                      ARGS, settings,)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# calculate metrics
settings = calculate_metrics(PFtrain, PFval, PFtest, PItrain, PIval, PItest, 
                            Ftrain, Fval, Ftest, Itrain, Ival, Itest, 
                            verbose=False, settings_dict = settings)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# save the calculated metrics
save_directory = settings['exp_specs_dir']
save_experiment_specs(ARGS.exp_name, settings, save_directory)