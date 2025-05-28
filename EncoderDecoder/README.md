# Encoder-Decoder to Predict Forced Response (ForceSMIP)

***
Predicts the forced response for a given input using an encoder-decoder architecture trained on ESMs

## Code Versions
***
This code has been verified to work with the following package versions:
* numpy==1.24.3
* tensorflow==2.13.0
* xarray==2023.7.0

## Order of Operations
***
* Step 1: enter path to *ForceSMIP* data directory in ```preprocessing.py``` e.g. root_dir = "path/to/ForceSMIP/"
* Step 2: add experimental settings to experiment and data dictionaries in ```experiments.py```
* Step 3: run experiment as ```python run_experiment_new.py experiment_name``` where ```experiment_name``` is a combination of experiments and data_dictionary entries, e.g. ```python run_experiment_new.py internal_test_standard_tos_1``` for a prediction of **tos** in **January** using the **standard** data_dictionary and **internal_test** experiment

### Credits
This work is a collaborative effort between the members of TPG: [Jamin K. Rader](https://jaminrader.wordpress.com), [Charlotte J. Connolly](https://sites.google.com/view/connolly-climate/home), [Dr. Martin A. Fernandez](https://mafern.github.io/), and [Dr. Emily M. Gordon](https://sites.google.com/view/emilygordon). 
