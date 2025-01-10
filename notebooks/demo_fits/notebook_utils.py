import numpy as np

def window_raster(raster, window=60):
    n_trials, n_time_bins = raster.shape
    n_windows = n_time_bins / window 
    n_windows = int(n_windows)

    def _custom_nansum(arr):
        is_all_nan = np.all(np.isnan(arr), axis=1)
        nansum = np.nansum(arr, axis=1)
        nansum = np.where(is_all_nan, np.nan, nansum)
        return nansum

    # check nan in raster
    windowed_raster = raster[:, :n_windows * window].reshape(n_trials, n_windows, window)
    # nansum = np.nansum(windowed_raster, axis=2)

    windowed_raster = np.vmap(_custom_nansum)(windowed_raster)
    mask = np.isnan(windowed_raster)
    mask = np.where(mask, 0, 1)

    return windowed_raster, mask
    
def window_constant_covariate(covariate, window=60):
    # window in ms
    # covariate is n_time_bins
    n_time_bins = covariate.shape[0]
    n_windows = n_time_bins / window
    n_windows = int(n_windows)
    
    windowed_covariate = covariate[:n_windows * window].reshape(n_windows, window)
    return np.nanmean(windowed_covariate, axis=1) # if constant covariate then mean is fine

def downsample_ramp_covariate(covariate, window=60):
    # window in ms
    # covariate is n_time_bins
    n_time_bins = covariate.shape[0]
    n_windows = n_time_bins / window
    n_windows = int(n_windows)
    windowed_covariate = covariate[:n_windows * window].reshape(n_windows, window)
    return np.nanmean(windowed_covariate, axis=1) # average covariate over window

def limit_to_window(arr, window, start_offset=None, end_offset=None):
    """ 
    arr = (n_trials, n_time_bins) e.g. 1000ms
    window = int in ms e.g. 10 ms
    start_offset = int in ms e.g. +150ms
    end_offset = int in ms e.g. -50
    """
    nmb_bins = arr.shape[1]
    window_size = window / 1000
    T = nmb_bins * window_size + window_size
    if start_offset is None:
        lower_bound = 0
    else:
        lower_bound = int((start_offset / 1000) / window_size)
    if end_offset is None:
        upper_bound = nmb_bins
    else:
        upper_bound = int((T + (end_offset / 1000)) / window_size)
    return arr[:, lower_bound:upper_bound] 


def read_Tall_epoch_npy(Y, bin_window, save_window, start_offset, end_offset):
    # shape is nested spikes in (N_neurons, N_choice, N_coh) 

    assert len(Y.shape) == 3
    N_coh_cond = Y.shape[2]

    t_count = 0
    T = None
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(N_coh_cond):
                t_count += Y[i, j, k].shape[0]
                if T is None:
                    T = Y[i, j, k].shape[1]
                else:
                    assert T == Y[i, j, k].shape[1]

    bin_size_seconds = bin_window / 1000.0
    max_time = T / 1000 
    t_ax = np.arange(0, max_time+bin_size_seconds, bin_size_seconds)
    cts_all = np.zeros((len(t_ax) - 1,)) 
    total_trial_count = 0  
    valid_trials = []
    Y_all = []
    Y_all_mask = []
    for i in range(Y.shape[0]):
        Y_all_unit = []
        Y_all_unit_mask = []
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                    
                trials = Y[i, j, k]

                if i == 0:
                    isnan = np.isnan(trials)
                    has_nan = np.any(isnan, axis=1)
                    firsttrue = np.argmax(isnan, axis=1)
                    firsttrue = np.where(has_nan, firsttrue, trials.shape[1])
                    firsttrue_time = firsttrue * (save_window / 1000.0)
                    cts, _ = np.histogram(firsttrue_time, bins=t_ax)
                    cts_all += cts
                    # LIMIT TRIALS THAT MATCH FULL WINDOW
                    toadd = np.where(firsttrue_time >= t_ax[-2])[0]
                    if len(valid_trials) == 0:
                        valid_trials = toadd
                    else:
                        toadd += total_trial_count
                        valid_trials = np.hstack((valid_trials, toadd))

                # """"""""" REMOVE THIS """""""""
                # # For each trial, replace the nan values with the avg value
                # trial_avgs = np.nanmean(trials, axis=1, keepdims=True)
                # trials = np.where(np.isnan(trials), trial_avgs, trials)
                # """"""""" REMOVE THIS """""""""
                        
                # LIMIT TO WINDOW
                # trials = limit_to_window(trials, save_window, start_offset, end_offset)
                # WINDOW
                if bin_window != 1 or bin_window != save_window:
                    trials, trials_mask = window_raster(trials, window=bin_window)
                else:
                    trials_mask = np.ones_like(trials)
                # NORMALIZE
                # trials = (trials - np.nanmean(trials, axis=1, keepdims=True)) / np.nanstd(trials, axis=1, keepdims=True)
                # STACK TRIALS
                if len(Y_all_unit) == 0:
                    Y_all_unit = trials
                    Y_all_unit_mask = trials_mask
                else:
                    Y_all_unit = np.vstack((Y_all_unit, trials))
                    Y_all_unit_mask = np.vstack((Y_all_unit_mask, trials_mask))

                total_trial_count += trials.shape[0]

        Y_all_unit = np.array(Y_all_unit)
        Y_all.append(Y_all_unit.T)
        Y_all_mask.append(Y_all_unit_mask.T)

    Y_all = np.array(Y_all)
    Y_all_mask = np.array(Y_all_mask)
    Y = Y_all.T
    Y_mask = Y_all_mask.T
    Y_mask = Y_mask[:,:,0]

    assert len(valid_trials) == cts_all[-1]

    T = Y.shape[1]
    Y = (Y - np.nanmean(Y, axis=(0,1), keepdims=True)) / np.nanstd(Y, axis=(0,1), keepdims=True)
    unshuffled_Y = np.copy(Y)

    return Y, Y_mask, unshuffled_Y, T, valid_trials

def read_Tsingle_epoch_npy(Y, bin_window, save_window, start_offset, end_offset):
    # shape is nested spikes in (N_neurons, N_coh)

    assert len(Y.shape) == 2

    N_coh_cond = Y.shape[1]
    t_count = 0
    T = None
    for i in range(Y.shape[0]):
        for j in range(N_coh_cond):
            t_count += Y[i, j].shape[0]
            if T is None:
                T = Y[i, j].shape[1]
            else:
                assert T == Y[i, j].shape[1]

    bin_size_seconds = bin_window / 1000.0
    max_time = T / 1000 
    t_ax = np.arange(0, max_time+bin_size_seconds, bin_size_seconds)
    cts_all = np.zeros((len(t_ax) - 1,)) 
    total_trial_count = 0  
    valid_trials = []
    Y_all = []
    Y_all_mask = []
    for i in range(Y.shape[0]):
        Y_all_unit = []
        for j in range(Y.shape[1]):

            trials = Y[i, j]

            if i == 0:
                isnan = np.isnan(trials)
                has_nan = np.any(isnan, axis=1)
                firsttrue = np.argmax(isnan, axis=1)
                firsttrue = np.where(has_nan, firsttrue, trials.shape[1])
                firsttrue_time = firsttrue * (save_window / 1000.0)
                cts, _ = np.histogram(firsttrue_time, bins=t_ax)
                cts_all += cts
                # LIMIT TRIALS THAT MATCH FULL WINDOW
                toadd = np.where(firsttrue_time >= t_ax[-2])[0]
                if len(valid_trials) == 0:
                    valid_trials = toadd
                else:
                    toadd += total_trial_count
                    valid_trials = np.hstack((valid_trials, toadd))

            # """"""""" REMOVE THIS """""""""
            # # For each trial, replace the nan values with the avg value
            # trial_avgs = jnp.nanmean(trials, axis=1, keepdims=True)
            # trials = jnp.where(jnp.isnan(trials), trial_avgs, trials)
            # """"""""" REMOVE THIS """""""""
                        
            # LIMIT TO WINDOW
            # trials = limit_to_window(trials, save_window, start_offset, end_offset)

            # WINDOW
            if bin_window != 1 or bin_window != save_window:
                trials, trials_mask = window_raster(trials, window=bin_window)
            else:
                trials_mask = np.ones_like(trials)

            # NORMALIZE
            # trials = (trials - jnp.nanmean(trials, axis=1, keepdims=True)) / jnp.nanstd(trials, axis=1, keepdims=True)
            # STACK TRIALS

            if len(Y_all_unit) == 0:
                Y_all_unit = Y[i, j]
                Y_all_unit_mask = trials_mask
            else:
                Y_all_unit = np.vstack((Y_all_unit, Y[i, j]))
                Y_all_unit_mask = np.vstack((Y_all_unit_mask, trials_mask))

            total_trial_count += trials.shape[0]

        Y_all_unit = np.array(Y_all_unit)
        Y_all.append(Y_all_unit.T)
        Y_all_mask.append(Y_all_unit_mask.T)

    Y_all = np.array(Y_all)
    Y_all_mask = np.array(Y_all_mask)
    Y = Y_all.T
    Y_mask = Y_all_mask.T
    Y_mask = Y_mask[:,:,0]

    assert len(valid_trials) == cts_all[-1]

    T = Y.shape[1]
    Y = (Y - np.nanmean(Y, axis=(0,1), keepdims=True)) / np.nanstd(Y, axis=(0,1), keepdims=True)
    unshuffled_Y = np.copy(Y)

    return Y, Y_mask, unshuffled_Y, T, valid_trials

def read_Tall_constant_covariate(U1, bin_window, save_window, start_offset, end_offset):
    U1_all_c1 = []
    U1_all_c2 = []
    for j in range(U1.shape[0]):
        for k in range(U1.shape[1]):
            for t in range(U1[j, k].shape[0]):
                coh_pulse = U1[j, k][t].reshape(1, -1)
                # LIMIT TO WINDOW
                # coh_pulse = limit_to_window(coh_pulse, save_window, start_offset, end_offset)
                # WINDOW
                if bin_window != 1 or bin_window != save_window:
                    coh_pulse = window_constant_covariate(coh_pulse.squeeze(), window=bin_window)

                if j == 0:
                    if len(U1_all_c1) == 0:
                        U1_all_c1 = coh_pulse
                    else:
                        U1_all_c1 = np.vstack((U1_all_c1, coh_pulse))
                else:
                    assert j == 1
                    if len(U1_all_c2) == 0:
                        U1_all_c2 = coh_pulse
                    else:
                        U1_all_c2 = np.vstack((U1_all_c2, coh_pulse))

    U1_all_c1 = U1_all_c1[::-1]
    U1_all = np.vstack((U1_all_c1, U1_all_c2))
    U1_all = np.array(U1_all)
    U1 = U1_all/1000 # convert to percentage (-1 to 1)

    coherence_indices = []
    prev_coh = None
    for trial_idx in range(U1.shape[0]):
        trial = U1[trial_idx]
        trial_coh = np.nanmax(trial)
        trial_coh_arg = np.where(trial == trial_coh)[0][0]
        trial_coh = float(trial[trial_coh_arg])
        trial_coh = round(trial_coh, 3)
        if prev_coh != trial_coh:
            # coherence_indices.append(float(trial_coh))
            coherence_indices.append(trial_idx)

        prev_coh = trial_coh

    U1 = np.nan_to_num(U1)

    return U1, coherence_indices


def read_Tsingle_constant_covariate(U1, bin_window, save_window, start_offset, end_offset):
    U1_all = []
    # Loop over coherence conditions
    for j in range(U1.shape[0]):
        # For each trial 
        for t in range(U1[j].shape[0]):
            coh_pulse = U1[j][t].reshape(1, -1)
            
            # coh_pulse = limit_to_window(coh_pulse, save_window, start_offset, end_offset)
            
            if bin_window != 1 or bin_window != save_window:
                coh_pulse = window_constant_covariate(coh_pulse.squeeze(), window=bin_window)
            
            if len(U1_all) == 0:
                U1_all = coh_pulse
            else:
                U1_all = np.vstack((U1_all, coh_pulse))
    
    U1_all = np.array(U1_all)
    U1 = U1_all / 1000.0
    
    coherence_indices = []
    prev_coh = None
    for trial_idx in range(U1.shape[0]):
        trial = U1[trial_idx]
        # get the maximum in that trial
        trial_coh = np.nanmax(trial)
        trial_coh_arg = np.where(trial == trial_coh)[0][0]
        trial_coh = float(trial[trial_coh_arg])
        trial_coh = round(trial_coh, 3)
        if prev_coh != trial_coh:
            coherence_indices.append(trial_idx)
        prev_coh = trial_coh

    U1 = np.nan_to_num(U1)

    return U1, coherence_indices
