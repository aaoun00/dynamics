import numpy as np

def custom_histc(data, bins):
    """
    Custom implementation of histc function in MATLAB. Difference is to do with
    how the inclusion/exclusion of bins is handled. This method results in spike 
    counts that are consistent with MATLAB's histc function as tested by extracting
    spike counts from matlab and testing using ==/np.allclose. 
    """
    data = data[~np.isnan(data)] # remove nans
    
    indices = np.digitize(data, bins, right=False) # right=False means that the right bin edge is not included

    counts = np.zeros(len(bins) - 1, dtype=int) # bins are bin edges so e.g. 3 edges gives 2 bins

    for idx in indices: # indices returns idx i for bin [bins[i-1], bins[i])
        if 1 <= idx < len(counts) + 1:  
            counts[idx - 1] += 1

    return counts

def window_raster(raster, window=60):
    # window in ms
    # raster is n_trials x n_time_bins
    n_trials, n_time_bins = raster.shape
    n_windows = n_time_bins / window 
    if n_windows % 1 != 0:
        n_windows = int(np.ceil(n_windows))
        n_time_bins = n_windows * window
        pad_width = int(n_time_bins - raster.shape[1])
        raster = np.pad(raster, ((0, 0), (0, pad_width)), mode='constant', constant_values = np.nan)
    else:
        n_windows = int(n_windows)
    windowed_raster = raster[:, :n_windows * window].reshape(n_trials, n_windows, window)
    return windowed_raster.squeeze()
