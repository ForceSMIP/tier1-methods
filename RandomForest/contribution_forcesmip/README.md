The following is a contribution to the ForceSMIP project and the associated code needed to reproduce the result.

# To install the necessary packages, use

pip install -r requirements.txt

# Steps to reproduce the Tier 1 estimates

1. Download the ForceSMIP data with the following command:
wget -r -nc -nH --user="climdyn" --password="friendly" ftp://data.iac.ethz.ch/ForceSMIP/

2. Run the notebook "PCA.ipynb" which will construct the principal components for each model.

3. Next, run the notebook "forced_signal.ipynb" which will calculate the forced signal for each model in the training dataset.

4. Next, run the notebook "estimate.ipynb" which will calculate the estimates for each Tier 1 evaluation.

# Final Estimates

Available at https://drive.google.com/drive/folders/1m-JWWf9-g81vozu1SeHUUNE0gVkSDHd6?usp=sharing
