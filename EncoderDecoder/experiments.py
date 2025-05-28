import numpy as np

# allow same data to be used for different experiments
def add_data_to_experiment(settings, data_name):
    for key in data_dictionary[data_name]:
        settings[key] = data_dictionary[data_name][key]
    return settings

# get settings for this experiment, set up directories
def get_experiment(exp_name, settings_overwrite=None):

    if settings_overwrite is None:
        settings = experiments[exp_name]
    else:
        settings = settings_overwrite[exp_name]
    settings["exp_name"] = exp_name
    settings["npz_dir"] = 'exp_data/'
    settings["pred_dir"] = 'saved_predictions/'
    settings["exp_specs_dir"] = 'exp_results/'
    settings["tune_specs_dir"] = 'tuning_results/'

    return settings

# dictionary of different data splits and variables
data_dictionary = {
    # "data_template":{
    #     "input_variable": ["tos",], # List of variables for input
    #     "target_variable": "tos", # Single variable output
    #     "train_models"  : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",], # Models for training
    #     "val_models"    : ["MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",], # Models for validations
    #     "test_models"   : ["CESM2",], # Models for testing
    #     "train_members" : np.arange(18), # Ensembles members for training
    #     "val_members"   : np.arange(18, 25), # Ensemble members for validation
    #     "test_members"  : np.arange(25), # Ensemble members for testing
    #     "time_range" : "Tier1", # ForceSMIP Tier
    #     "evaluate" : True or False # Whether or not there is a 
    # },

    "Train4_Val4_CESM2_tos_tos":{
        "input_variable": ("tos",),
        "target_variable": "tos",
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CESM2",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_MIROC6_tos_tos":{
        "input_variable": ("tos",),
        "target_variable": "tos",
        "train_models"  : ("CESM2","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("CESM2","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("MIROC6",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_CanESM5_tos_tos":{
        "input_variable": ("tos",),
        "target_variable": "tos",
        "train_models"  : ("CESM2","MIROC6","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("CESM2","MIROC6","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CanESM5",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_MPI-ESM1-2-LR_tos_tos":{
        "input_variable": ("tos",),
        "target_variable": "tos",
        "train_models"  : ("CESM2","MIROC6","CanESM5","MIROC-ES2L",),
        "val_models"    : ("CESM2","MIROC6","CanESM5","MIROC-ES2L",),
        "test_models"   : ("MPI-ESM1-2-LR",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_MIROC-ES2L_tos_tos":{
        "input_variable": ("tos",),
        "target_variable": "tos",
        "train_models"  : ("CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR",),
        "val_models"    : ("CESM2","MIROC6","CanESM5","MPI-ESM1-2-LR",),
        "test_models"   : ("MIROC-ES2L",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_CESM2_all_pr":{
        "input_variable": ("pr","tos",'psl','tas',),
        "target_variable": "pr",
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CESM2",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "Train4_Val4_CESM2_pr_pr":{
        "input_variable": ("pr",),
        "target_variable": "pr",
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CESM2",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
        "month": "annual",
    },

    "standard":{
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L","CESM2"),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L","CESM2"),
        "test_models"   : (),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": True,
    },

    "CESMval":{
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CESM2",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "time_range" : "Tier1",
        "evaluate": False,
    },

    "CESMvalall":{
        "train_models"  : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "val_models"    : ("MIROC6","CanESM5","MPI-ESM1-2-LR","MIROC-ES2L",),
        "test_models"   : ("CESM2",),
        "train_members" : np.arange(18),
        "val_members"   : np.arange(18, 25),
        "test_members"  : np.arange(25),
        "input_variable" : ('tos','tas','psl','pr',),

        "time_range" : "Tier1",
        "evaluate": False,
    },
}

# handle running a different model for each month
months = ["annual","1","2","3","4","5","6","7","8","9","10","11","12",]
vars = ["pr", "psl", "tas", "zmta", "tos", "siconc", "monmaxpr", "monmaxtasmax", "monmintasmin",]
for var in vars:
    for month in months:
        key = 'standard_' + var + '_' + str(month)
        data_dictionary[key] = data_dictionary['standard'].copy()
        data_dictionary[key]["input_variable"] = (var,)
        data_dictionary[key]["target_variable"] = var
        data_dictionary[key]["month"] = month

        key = 'CESMval_' + var + '_' + str(month)
        data_dictionary[key] = data_dictionary['CESMval'].copy()
        data_dictionary[key]["input_variable"] = (var,)
        data_dictionary[key]["target_variable"] = var
        data_dictionary[key]["month"] = month

        key = 'CESMvalall_' + var + '_' + str(month)
        data_dictionary[key] = data_dictionary['CESMvalall'].copy()
        data_dictionary[key]["target_variable"] = var
        data_dictionary[key]["month"] = month

# experiment dictionary for experimental setup (architecture), target
experiments = {
    # "exp_template" : {
    #     #Ran by: 
    #     #Metrics:
    #     #Notes:
    #     "seed"          : 0 ,
    #     "learn_rate"    : .001,
    #     "patience"      : 50,
    #     "batch_size"    : 64,
    #     "max_epochs"    : 5_000,
    #     "metric"        : "",
        
    #     "input_std_method"    : "", #self, feature, overall
    #     "output_std_method"   : "", # feature, overall
    #     "target_component": "internal", #forced or internal

    #     "CNN_blocks"    : [2],
    #     "CNN_filters"   : [32],
    #     "CNN_kernals"   : [3],
    #     "CNN_strides"   : [1],

    #     "encoding_nodes" : [20, 20, 10],
    #     "code_nodes"     : 1,
    #     "activation"     : "linear",
    #     "variational_loss": .001
    # },

    "internal_feature" : {

        # Standardization + prediction component
        "target_component": "internal",
        'save_predictions': True,

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        
        # Architecture specs
        "encoding_nodes" : [10,10,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "forced_feature" : {

        # Standardization + prediction component
        "target_component": "forced",
        'save_predictions': False,

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 2000,
        
        # Architecture specs
        "encoding_nodes" : [10,10,],
        "code_nodes"     : 3,
        "activation"     : "tanh",
        "variational_loss": 0.0001
    },

    "internal_test" : {

        # Standardization + prediction component
        "target_component": "internal",
        'save_predictions': True,

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.0001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 15,
        
        # Architecture specs
        "encoding_nodes" : [1000,1000,],
        "code_nodes"     : 100,
        "activation"     : "tanh",
        "variational_loss": 0.,
        "ridge"  :  0,
    },

    "forced_test" : {

        # Standardization + prediction component
        "target_component": "forced",
        'save_predictions': True,

        # Network specs
        "seed"          : 0 ,
        "learn_rate" : 0.0001,
        "patience"      : 100,
        "batch_size"    : 64,
        "max_epochs"    : 15,
        
        # Architecture specs
        "encoding_nodes" : [100,100,],
        "code_nodes"     : 5,
        "activation"     : "tanh",
        "variational_loss": 0.,
        "ridge"  :  0.,
    },
}

### Add data_name to the experiments
for exp_name in experiments.copy():
    for data_name in data_dictionary:
        experiments[exp_name + '_' + data_name] = experiments[exp_name].copy()
        experiments[exp_name + '_' + data_name]["data_name"] = data_name
        add_data_to_experiment(experiments[exp_name + '_' + data_name], data_name)
