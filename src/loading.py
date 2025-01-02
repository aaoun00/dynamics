import numpy as np
import scipy.io as sio
from src.utils import get_index, format_event
from src.format import get_raster_array, get_raster_aligned_covariates

def load_fira_mat_file(fpath, align_struct):
    data = sio.loadmat(fpath)

    # print(data.keys())

    FIRA = data['FIRA']

    trg_cho_index = get_index('targ_cho', FIRA) # choice target
    trg_cor_index = get_index('targ_cor', FIRA) # correct choice
    coh_index = get_index('dot_coh', FIRA) # coherence level

    trg_cho = FIRA[0, 1][:, trg_cho_index]
    trg_cho = format_event(trg_cho)

    trg_cor = FIRA[0, 1][:, trg_cor_index]
    trg_cor = format_event(trg_cor)

    coh = FIRA[0, 1][:, coh_index]
    coh = format_event(coh)

    cor = trg_cho == trg_cor # correct trials
    coh_set = np.unique(coh) # all coherence levels
    valid_trials = ~np.isnan(trg_cho) # valid trials

    trg_right = 1 
    coh_group = [np.array([0, 16], dtype=float), np.array([32, 64], dtype=float), np.array([128, 256, 512], dtype=float)]

    alignto = align_struct # can load for multiple alignment windows

    trials_cor = [
        np.where(valid_trials & (cor | (coh == 0)) & (trg_cho == trg_right))[0], # right
        np.where(valid_trials & (cor | (coh == 0)) & (trg_cho != trg_right))[0] # left
    ] # list of correct trials for right and left choices 

    raster = np.empty(len(alignto), dtype=object)
    unit_id = np.empty(len(alignto), dtype=object)

    for i in range(len(alignto)):
        output = get_raster_array(FIRA, align_struct[i], trials_cor, alignto[i]['limits'][0], align_struct[i]['limits'][1])

        raster[i] = output[0]
        unit_id[i] = output[1]

        ramps, choice_pulse = get_raster_aligned_covariates(FIRA, align_struct[i], trials_cor, alignto[i]['limits'][0], align_struct[i]['limits'][1])

    raster = np.array([raster[0]], dtype=object) 
    ramps = np.array([ramps], dtype=object)
    choice_pulse = np.array([choice_pulse], dtype=object)
    # coherence level is not here because it is not time varying, it is constant for each trial

    return raster, ramps, choice_pulse, alignto, unit_id, coh, trials_cor, coh_group, coh_set
