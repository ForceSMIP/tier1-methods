This code implement estimation of the forced response with GPCA

Author: Gherardo Varando <gherardo.varando@uv.es> 


## prerequisites 


- We use R version 4.3.3 (2024-02-29)
- and additional packages all available from CRAN:
  - terra
  - zoo
  - optparse
  - cli
  - glue
  - glmnet

## content 

The code files are:

- `fix_forcings.R` script to interpolate and extrapolate the forcing time series 
- `util.R`, some functions to compute climatology and anomalies an other utility functions
- `gpca.R`, the main GPCA algorithm 
- `forced_gpca.R`, the main entry point to compute the forced response estimations 

We also include the interpolated and extrapolated forcing time series in data/

## how to 

To use the code

- install the required R packages
- include the input files in `data/` (e.g. `data/Evaluation-Tier1`)
- run the `forced_gpca.R` script with appropriate command line arguments (see below)

In particular:

- for `gpca` we run 
```
Rscript forced_gpca.R --out out-Tier1/gpca/ --input data/Evaluation-Tier1/ --maxlag 60 --pcarank 20 --degree 5 --expansion fourier --modelf glmnet --modelnf lm --glmnet.alpha 0.5 --level 0.001
```

- for `gpca-da` we additionally specify the directory with the psl files, the script will match them automatically to each input files. 
```
Rscript forced_gpca.R --out out-Tier1/gpca/ --input data/Evaluation-Tier1/ --psl data/Evaluation-Tier1/Amon/psl --maxlag 60 --pcarank 20 --pcarankpsl 4 --degree 5 --expansion fourier --modelf glmnet --modelnf lm --glmnet.alpha 0.5 --level 0.001
```

Some parameters are set to default values, check `Rscript forced_gpca.R --help` for an exhaustive list of arguments (for example the forcings file location is set to default to `data/interpolatedForcing.csv` but it could be changed). 
