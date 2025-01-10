import numpy as np
import scipy.io as sio
from src.utils import get_index, format_event
from src.format import get_raster_array, get_raster_aligned_covariates

def get_supp_events(FIRA):
    event_list = ['fp_on', 'fp_off', 'targ_on', 'targ_off', 'dots_on', 'dots_off', 'feedback', 'in_fp_wd', 'out_fp_wd', 
                  'in_targ_wd', 'out_targ_wd', 'task', 'trial', 'touch_fixation', 'touch_response', 'eye_fixation', 
                  'eye_response', 'key_response', 'rt_fp', 'rt_dots', 'dots_on_screen'] # , 'sacc']

    event_dict = {}

    for event in event_list:
        idx = get_index(event, FIRA)
        event_data = FIRA[0, 1][:, idx]
        event_data = format_event(event_data)
        event_dict[event] = event_data

    return event_dict

def load_fira_mat_file(fpath, align_struct):
    data = sio.loadmat(fpath)

    # print(data.keys())

    FIRA = data['FIRA']

    trg_cho_index = get_index('targ_cho', FIRA) # choice target
    trg_cor_index = get_index('targ_cor', FIRA) # correct choice
    coh_index = get_index('dot_coh', FIRA) # coherence level

    event_dict = get_supp_events(FIRA)

    trg_cho = FIRA[0, 1][:, trg_cho_index]
    trg_cho = format_event(trg_cho)
    trg_cor = FIRA[0, 1][:, trg_cor_index]
    trg_cor = format_event(trg_cor)
    coh = FIRA[0, 1][:, coh_index]
    coh = format_event(coh)

    event_dict['targ_cho'] = trg_cho
    event_dict['targ_cor'] = trg_cor
    event_dict['dot_coh'] = coh

    cor = trg_cho == trg_cor # correct trials
    coh_set = np.unique(coh) # all coherence levels
    valid_trials = ~np.isnan(trg_cho) # valid trials

    trg_right = 1 
    coh_group = [np.array([0, 16], dtype=float), np.array([32, 64], dtype=float), np.array([128, 256, 512], dtype=float)]

    # alignto = align_struct # can load for multiple alignment windows

    trials_cor = [
        np.where(valid_trials & (cor | (coh == 0)) & (trg_cho == trg_right))[0], # right
        np.where(valid_trials & (cor | (coh == 0)) & (trg_cho != trg_right))[0] # left
    ] # list of correct trials for right and left choices 

    raster = np.empty(len(align_struct), dtype=object)
    unit_id = np.empty(len(align_struct), dtype=object)
    ramps = np.empty(len(align_struct), dtype=object)
    choice_pulse = np.empty(len(align_struct), dtype=object)

    for i in range(len(align_struct)):
        output = get_raster_array(FIRA, align_struct[i], trials_cor, align_struct[i]['limits'][0], align_struct[i]['limits'][1])

        raster[i] = output[0]
        unit_id[i] = output[1]

        output = get_raster_aligned_covariates(FIRA, align_struct[i], trials_cor, align_struct[i]['limits'][0], align_struct[i]['limits'][1])

        ramps[i] = output[0]
        choice_pulse[i] = output[1]
    # raster = np.array([raster[0]], dtype=object) 
    raster = np.array([raster[i] for i in range(len(align_struct))], dtype=object)
    ramps = np.array([ramps[i] for i in range(len(align_struct))], dtype=object)
    choice_pulse = np.array([choice_pulse[i] for i in range(len(align_struct))], dtype=object)
    # coherence level is not here because it is not time varying, it is constant for each trial

    return raster, ramps, choice_pulse, align_struct, unit_id, coh, trials_cor, coh_group, coh_set, event_dict
