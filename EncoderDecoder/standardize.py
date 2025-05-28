import numpy as np

# standardize by mean and std from function inputs
def standardize(D, Dmean, Dstd):
    return np.nan_to_num((D - Dmean) / Dstd)

# standardize by lat, lon, variable
def self_standardize(D, climo=None, mean_only=False):
    Dmean = np.nanmean(D, axis=(1,2,3))[:, None, None, None]
    if mean_only:
        Dstd = 1
    else:
        Dstd = np.nanstd(D, axis=(1,2,3))[:, None, None, None]
    if climo is  None:
        return np.nan_to_num((D - Dmean) / Dstd)
    else:
        return np.nan_to_num((D - Dmean) / Dstd) - climo

# unstandardize by mean and std from function inputs
def unstandardize(D, Dmean, Dstd):
    return D * Dstd + Dmean

# standardize by year for each member
def standardize_for_each_member(D_orig):
    n_yrs = 73 # HARDCODE FIXME
    D = D_orig.reshape(D_orig.shape[0]//n_yrs, n_yrs, D_orig.shape[1], D_orig.shape[2], D_orig.shape[3])
    Dmean = D.mean(axis=(1,))[:, None, ...]
    Dstd = D.std(axis=(1,))[:, None, ...]
    D = np.nan_to_num((D - Dmean) / Dstd).reshape(D_orig.shape)
    return D, Dmean, Dstd

# unstandardize by year for each member
def unstandardize_for_each_member(D_orig, Dmean, Dstd):
    n_yrs = 73 # HARDCODE FIXME
    D = D_orig.reshape(D_orig.shape[0]//n_yrs, n_yrs, D_orig.shape[1], D_orig.shape[2], D_orig.shape[3])
    D = np.nan_to_num(D * Dstd + Dmean).reshape(D_orig.shape)
    return D

# unstandardize and return forced based on what is predicted (forced directly, or internal as full-internal)
def unstandardize_predictions(Atrain, Aval, Atest,
                              Ptr_stand, Pva_stand, Pte_stand,
                              Ttr_mean, Ttr_std, 
                              Tva_mean, Tva_std, 
                              Tte_mean, Tte_std,
                              settings):
                              
    # get the target index, in case there are multiple input variables
    target_index = np.where(np.array(settings['input_variable']) == np.array(settings['target_variable']))[0]
    # select 'all' for just the variable correlating to the target
    Atr = Atrain[..., target_index]
    Ava = Aval[..., target_index]
    Ate = Atest[..., target_index]

    # unstandardize the predictions, in this case forced
    # Ptr = unstandardize_for_each_member(Ptr_stand, Ttr_mean, Ttr_std)
    # Pva = unstandardize_for_each_member(Pva_stand, Tva_mean, Tva_std)
    # Pte = unstandardize_for_each_member(Pte_stand, Tte_mean, Tte_std)
    Ptr = unstandardize(Ptr_stand, Ttr_mean, Ttr_std)
    Pva = unstandardize(Pva_stand, Tva_mean, Tva_std)
    Pte = unstandardize(Pte_stand, Tte_mean, Tte_std)
    
    if settings['target_component'] == 'forced':
        # calculate internal as full - forced
        PItr = Atr - Ptr
        PIva = Ava - Pva
        PIte = Ate - Pte
        # return forced predictions
        PFtr = Ptr
        PFva = Pva
        PFte = Pte

    if settings['target_component'] == 'internal':
        # calculate forced as full - internal
        PFtr = Atr - Ptr
        PFva = Ava - Pva
        PFte = Ate - Pte
        # return internal predictions
        PItr = Ptr
        PIva = Pva
        PIte = Pte

    return PFtr, PFva, PFte, PItr, PIva, PIte

# standardize inputs and outputs (forced, internal) and assigning everything correctly depending
# on what is being predicted (forced or internal)
def standardize_all_data(Atr, Ava, Ate,
                    Ftr, Fva, Fte,
                    Itr, Iva, Ite,
                    settings):
    # Standardize the inputs
    if settings['target_component'] == 'internal':
        Xtr_stand = self_standardize(Atr)
        Xva_stand = self_standardize(Ava)
        Xte_stand = self_standardize(Ate)
    elif settings['target_component'] == 'forced':
        Xtr_stand = Atr.copy()
        Xva_stand = Ava.copy()
        Xte_stand = Ate.copy()

    # standardize by year
    Xtr_stand, __, __ = standardize_for_each_member(Xtr_stand)
    Xva_stand, __, __ = standardize_for_each_member(Xva_stand)
    Xte_stand, __, __ = standardize_for_each_member(Xte_stand)

    # standardize all the splits by the training mean and std for the forced
    fmean = Ftr.mean(axis=(0))
    fstd = Ftr.std(axis=(0))
    Ftr_stand = standardize(Ftr, fmean, fstd)
    Fva_stand = standardize(Fva, fmean, fstd)
    Fte_stand = standardize(Fte, fmean, fstd)

    # standardize all the splits by the training mean and std for internal
    imean = Itr.mean(axis=(0))
    istd = Itr.std(axis=(0))
    Itr_stand = standardize(Itr, imean, istd)
    Iva_stand = standardize(Iva, imean, istd)
    Ite_stand = standardize(Ite, imean, istd)

    # assign variables depending on predicted target (forced or internal)
    if settings['target_component'] == 'internal':
        Ttr_stand = Itr_stand
        Tva_stand = Iva_stand
        Tte_stand = Ite_stand
        Ttr_mean = imean
        Ttr_std = istd
        Tva_mean = imean
        Tva_std = istd
        Tte_mean = imean
        Tte_std = istd
    elif settings['target_component'] == 'forced':
        Ttr_stand = Ftr_stand
        Tva_stand = Fva_stand
        Tte_stand = Fte_stand
        Ttr_mean = fmean
        Ttr_std = fstd
        Tva_mean = fmean
        Tva_std = fstd
        Tte_mean = fmean
        Tte_std = fstd

    # Atr_stand, Atr_mean, Atr_std = standardize_for_each_member(Atr)
    # Ava_stand, Ava_mean, Ava_std  = standardize_for_each_member(Ava)
    # Ate_stand, Ate_mean, Ate_std  = standardize_for_each_member(Ate)

    # Ftr_stand, Ftr_mean, Ftr_std = standardize_for_each_member(Ftr)
    # Fva_stand, Fva_mean, Fva_std  = standardize_for_each_member(Fva)
    # Fte_stand, Fte_mean, Fte_std  = standardize_for_each_member(Fte)

    # Itr_stand, Itr_mean, Itr_std = standardize_for_each_member(Itr)
    # Iva_stand, Iva_mean, Iva_std  = standardize_for_each_member(Iva)
    # Ite_stand, Ite_mean, Ite_std  = standardize_for_each_member(Ite)

    # if settings['target_component'] == 'internal':
    #     Ttr_stand = Itr_stand
    #     Tva_stand = Iva_stand
    #     Tte_stand = Ite_stand
    #     Ttr_mean = Itr_mean
    #     Ttr_std = Itr_std
    #     Tva_mean = Iva_mean
    #     Tva_std = Iva_std
    #     Tte_mean = Ite_mean
    #     Tte_std = Ite_std
    # elif settings['target_component'] == 'forced':
    #     Ttr_stand = Ftr_stand
    #     Tva_stand = Fva_stand
    #     Tte_stand = Fte_stand
    #     Ttr_mean = Ftr_mean
    #     Ttr_std = Ftr_std
    #     Tva_mean = Fva_mean
    #     Tva_std = Fva_std
    #     Tte_mean = Fte_mean
    #     Tte_std = Fte_std

    return  Xtr_stand, Xva_stand, Xte_stand, \
            Ttr_stand, Tva_stand, Tte_stand, \
            Ttr_mean, Ttr_std, Tva_mean, Tva_std, Tte_mean, Tte_std
