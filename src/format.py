import numpy as np
from src.utils import get_index, format_event
from src.stats import custom_histc, window_raster

def get_raster_array(FIRA, alignto, trials=None, lowcut=None, highcut=None):
    """
    Returns rasters of all neurons for arbitrary groups of trials.
    
    Parameters:
        FIRA: The FIRA structure that contains the data.
        alignto: What the rasters should be aligned to.
        trials: A list of trial indices.
        lowcut: Defines the lower time limit of the rasters.
        highcut: Defines the upper time limit of the rasters.
    
    Returns:
        RASTER: A list where each element is an array of shape [units, trials, time].
        unit_id: The ID of units in RASTER.
        times: Valid time windows for rasters of each trial.
    """
    
    # If no trials are specified, use all trials
    if trials is None:
        trials = np.arange(FIRA[0,1].shape[0]).reshape(1, -1)

    # If no lowcut or highcut is specified, set them to have no effect
    if lowcut is None:
        lowcut = {'event': None, 'offset': 0}
    else:
        lowcut = {'event': lowcut[0], 'offset': lowcut[1]}
    
    if highcut is None:
        highcut = {'event': None, 'offset': None}
    else:
        highcut = {'event': highcut[0], 'offset': highcut[1]}

    raster = np.empty(len(trials), dtype=object) # len trials is 2 for Tin and Tout, change name when u come back to this

    # Get the spike data
    c = 0
    for item in FIRA[0,0][0,0]:
        if c == 4:
            spikecd = item
            break
        c += 1

    unit_num = spikecd.shape[0] # number of neurons
    unit_id = spikecd[:,0] - 1 # neuron ids
    unit_index = np.ravel_multi_index((spikecd[:,0], spikecd[:,1]), FIRA[0,2][0,0].shape) - 1 # neuron indices
    times = [{} for _ in range(len(trials))] # array for valid time windows

    # Loop to get the valid time windows
    for g in range(len(trials)):

        event_index = get_index(alignto['event'], FIRA)
        wrt = FIRA[0,1][trials[g], event_index].astype(np.int32) # wrt = with respect to? 
        begin_times = wrt + alignto['start_offset']
        end_times = wrt + alignto['end_offset']

        # If lowcut or highcut is specified, adjust the time windows (e.g. go to end offset except if lowcut is reached)
        if lowcut['event']:
            lowcut_index = get_index(lowcut['event'], FIRA)
            lowcut_event = FIRA[0,1][trials[g], lowcut_index]
            lowcut_event = format_event(lowcut_event)

            """ To not pad with nans """
            # begin_times_cap = np.maximum(begin_times, lowcut_event + lowcut['offset'])
            # begin_times = begin_times
            """ To not pad with nans """

            begin_times = np.maximum(begin_times, lowcut_event + lowcut['offset'])

        if highcut['event']:
            highcut_index = get_index(highcut['event'], FIRA)
            highcut_event = FIRA[0,1][trials[g], highcut_index]
            highcut_event = format_event(highcut_event)

            """ To not pad with nans """
            # end_times_cap = np.minimum(end_times, highcut_event + highcut['offset'])
            # end_times = end_times
            """ To not pad with nans """

            end_times = np.minimum(end_times, highcut_event + highcut['offset'])

        # Store valid time windows
        times[g]['wrt'] = wrt
        times[g]['begin_times'] = begin_times
        times[g]['end_times'] = end_times
        
        raster_len = len(np.arange(int(alignto['start_offset']), int(alignto['end_offset']) + 1))
        raster[g] = np.full((unit_num, len(trials[g]), int(raster_len)), np.nan)

        # Loop to get the rasters
        for i in range(len(trials[g])):
            if (np.isnan(end_times[i] - begin_times[i]) or 
                np.isnan(wrt[i]) or 
                not np.isreal(wrt[i]) or 
                not np.isreal(end_times[i]) or 
                not np.isreal(begin_times[i]) or 
                end_times[i] == 0):
                continue
            
            """ To not pad with nans add the commented out code below """
            if int(end_times[i]) + 1 <= int(begin_times[i]): # or int(end_times_cap[i]) + 1 < int(begin_times[i]):
                continue
            else:
                t = np.arange(int(begin_times[i]), int(end_times[i]) + 1) # different to t_, this is the time bin edges to use for histogram
                t_ = t - wrt[i] - alignto['start_offset'] # drop + 1 that was in matlab code
                t = np.hstack((t, t[-1] + 1)) # add one more bin edge to get the last bin, DO NOT MOVE THIS BEFORE t_

                for u in range(unit_num):
                    sp = FIRA[0,2][trials[g][i],0][unit_index[u]][0] # spike times
                    spike_counts = custom_histc(sp, t) 
                    raster[g][u, i, t_] = spike_counts 

    return raster, unit_id, times

def get_raster_aligned_covariates(FIRA, alignto, trials=None, lowcut=None, highcut=None):
    """
    Returns covariates of all neurons for arbitrary groups of trials. Very similar to get_raster_array
    so less comments are provided.
    
    Parameters:
        FIRA: The FIRA structure that contains the data.
        alignto: What the rasters should be aligned to.
        trials: A list of trial indices.
        lowcut: Defines the lower time limit of the rasters.
        highcut: Defines the upper time limit of the rasters.
    
    Returns:
        time_time_ramps: A list where each element is an array of shape [trials, time].
        choice_pulse: A list where each element is an array of shape [trials, time].
    """
    
    if trials is None:
        trials = np.arange(FIRA[0,1].shape[0]).reshape(1, -1)

    if lowcut is None:
        lowcut = {'event': None, 'offset': 0}
    else:
        lowcut = {'event': lowcut[0], 'offset': lowcut[1]}
    
    if highcut is None:
        highcut = {'event': None, 'offset': None}
    else:
        highcut = {'event': highcut[0], 'offset': highcut[1]}

    time_ramps = np.empty(len(trials), dtype=object)
    choice_pulse = np.empty(len(trials), dtype=object)
    c = 0
    for item in FIRA[0,0][0,0]:
        if c == 4:
            spikecd = item
            break
        c += 1

    for g in range(len(trials)):
        event_index = get_index(alignto['event'], FIRA)
        wrt = FIRA[0,1][trials[g], event_index].astype(np.int32) 
        begin_times = wrt + alignto['start_offset']
        end_times = wrt + alignto['end_offset']

        if lowcut['event']:
            lowcut_index = get_index(lowcut['event'], FIRA)
            lowcut_event = FIRA[0,1][trials[g], lowcut_index]
            lowcut_event = format_event(lowcut_event)

            # begin_times_cap = np.maximum(begin_times, lowcut_event + lowcut['offset'])
            # begin_times = begin_times

            begin_times = np.maximum(begin_times, lowcut_event + lowcut['offset'])

        if highcut['event']:
            highcut_index = get_index(highcut['event'], FIRA)
            highcut_event = FIRA[0,1][trials[g], highcut_index]
            highcut_event = format_event(highcut_event)

            # end_times_cap = np.minimum(end_times, highcut_event + highcut['offset'])
            # end_times = end_times

            end_times = np.minimum(end_times, highcut_event + highcut['offset'])

        raster_len = len(np.arange(int(alignto['start_offset']), int(alignto['end_offset']) + 1))

        time_ramps[g] = np.full((len(trials[g]), int(raster_len)), np.nan)
        choice_pulse[g] = np.full((len(trials[g]), int(raster_len)), np.nan)

        for i in range(len(trials[g])):
            if (np.isnan(end_times[i] - begin_times[i]) or 
                np.isnan(wrt[i]) or 
                not np.isreal(wrt[i]) or 
                not np.isreal(end_times[i]) or 
                not np.isreal(begin_times[i]) or 
                end_times[i] == 0):
                continue
            
            if int(end_times[i]) + 1 <= int(begin_times[i]): # or int(end_times_cap[i]) + 1 <= int(begin_times[i]):
                continue
            else:
                """ To not pad with nans need to swap the commented and uncommented code below """
                # ramp_t_ = np.arange(int(begin_times_cap[i]), int(end_times_cap[i]) + 1)
                # ramp_t_ = ramp_t_ - wrt[i] - alignto['start_offset']

                t = np.arange(int(begin_times[i]), int(end_times[i]) + 1) # different to t_, this is the time bin edges to use for histogram
                t_ = t - wrt[i] - alignto['start_offset'] # drop + 1 that was in matlab code
                ramp_t_ = t_
                """ To not pad with nans need to swap the commented and uncommented code above """

                time_ramps[g][i, ramp_t_] = np.linspace(0, 1 - (1 / len(ramp_t_)), len(ramp_t_))

                if g == 0:
                    choice_pulse[g][i, ramp_t_] = np.ones(len(ramp_t_)) 
                else:
                    choice_pulse[g][i, ramp_t_] = np.ones(len(ramp_t_)) * -1
                
                # Dont want nans in the covariates
                first_non_nan = np.where(~np.isnan(time_ramps[g][i, :]))[0][0]
                last_non_nan = np.where(~np.isnan(time_ramps[g][i, :]))[0][-1]

                # make time ramp 0 before ramp and 1 after ramp
                time_ramps[g][i, :first_non_nan] = 0
                time_ramps[g][i, last_non_nan + 1:] = 1

                # make choice pulse 0 before ramp and 0 after ramp
                choice_pulse[g][i, :first_non_nan] = 0
                choice_pulse[g][i, last_non_nan + 1:] = 0

    return time_ramps, choice_pulse

def organize_raster_by_condition(raster, ramps, choice_pulse, coh, trials_cor, coh_set=None, window_size=1):
    """
    Organizes data for saving into 3 groups: Tin, Tout, Tall (= Tin + Tout).

    window_size in ms for binning spikes. Default is 1 ms so no binning is done.

    Example coh_set: coh_group = [np.array([0, 16], dtype=float), np.array([32, 64], dtype=float), np.array([128, 256, 512], dtype=float)]
    """

    n_units = raster[0][0].shape[0]

    # if coh_set is None, then use all unique coherences in the data
    if coh_set is None:
        coh_set = np.unique(coh)
        coh_set = coh_set[coh_set == coh_set]
        use_coh_groups = False
    # otherwise can restrict to a subset of coherences
    else:
        use_coh_groups = True

    # This is generally what is used by the model
    raster_by_cond = np.empty((len(raster), n_units, 2, len(coh_set)), dtype=object) # spikes
    inputs_by_cond = np.empty((len(raster), 2, len(coh_set)), dtype=object) # coherence level 
    trial_ramp_by_cond = np.empty((len(raster), 2, len(coh_set)), dtype=object) # time ramp
    trial_choice_by_cond = np.empty((len(raster), 2, len(coh_set)), dtype=object) # choice pulse

    # Tin
    raster_by_cond_t1 = np.empty((len(raster), n_units, len(coh_set)), dtype=object)
    inputs_by_cond_t1 = np.empty((len(raster), len(coh_set)), dtype=object)
    trial_ramp_by_cond_t1 = np.empty((len(raster), len(coh_set)), dtype=object)
    trial_choice_by_cond_t1 = np.empty((len(raster), len(coh_set)), dtype=object)

    # Tout
    raster_by_cond_t2 = np.empty((len(raster), n_units, len(coh_set)), dtype=object)
    inputs_by_cond_t2 = np.empty((len(raster), len(coh_set)), dtype=object)
    trial_ramp_by_cond_t2 = np.empty((len(raster), len(coh_set)), dtype=object)
    trial_choice_by_cond_t2 = np.empty((len(raster), len(coh_set)), dtype=object)

    for g in range(len(raster)):

        interval_window = raster[g] 

        for c in range(interval_window.shape[0]):

            condition = interval_window[c] # Tin or Tout
            coherence_cond = coh[trials_cor[c]] # coherence for each trial in this condition

            for u in range(n_units):

                unit_raster = condition[u]

                for t in range(unit_raster.shape[0]):

                    trial_coh = coherence_cond[t] 

                    coh_cond = None
                    if not use_coh_groups:
                        coh_cond = np.where(coh_set == trial_coh)[0][0]
                    else:
                        # Look at example coh_set above
                        # This is to group coherences together when using coh_set and/or restrict to specific coherences
                        for coh_grp in range(len(coh_set)):
                            if trial_coh in coh_set[coh_grp]:
                                coh_cond = coh_grp
                                break

                    if coh_cond is None or np.isnan(unit_raster[t, :]).all():
                        pass
                    else:
                        trial_spikes = unit_raster[t, :]
                        trial_spikes = trial_spikes.reshape(1, -1)
                        trial_ramp = ramps[g,c][t, :] 
                        trial_choice = choice_pulse[g,c][t, :]

                        """ If you want coherence pulse instead of coherence level (single line) """
                        # trial_inputs = trial_choice * trial_coh 
                        """ If you want coherence pulse instead of coherence level (single line) """

                        """ If you want nans """
                        # trial_inputs[trial_choice == 0] = np.nan
                        # trial_ramp[trial_choice == 0] = np.nan
                        # # trial_choice[trial_choice == 0] = np.nan

                        # # trial_inputs[trial_inputs == -0] = 0 # -0 vs 0 shouldn't matter
                        
                        # This won't do anything if window_size = 1
                        trial_spikes = window_raster(trial_spikes, window_size)
                        trial_ramp = window_raster(trial_ramp.reshape(1,-1), window_size)
                        trial_choice = window_raster(trial_choice.reshape(1,-1), window_size)

                        if len(trial_spikes.shape) == 1:
                            pass 
                        else:
                            # If bin window was 1, then nothing happens because we still have it with extra dimension at the end
                            trial_spikes = np.nansum(trial_spikes, axis=1)
                            trial_ramp = np.nanmean(trial_ramp, axis=1)
                            trial_choice = np.nanmean(trial_choice, axis=1)
                            trial_inputs = np.nanmean(trial_inputs, axis=1)

                        # Set trial coherence level to +/-
                        if c == 0:
                            trial_inputs = np.ones(trial_spikes.shape) * trial_coh 
                        else:
                            assert c == 1
                            trial_inputs = np.ones(trial_spikes.shape) * trial_coh * -1

                        # This is both Tin and Tout
                        # Add if first otherwise stack
                        if raster_by_cond[g, u, c, coh_cond] is None:
                            raster_by_cond[g, u, c, coh_cond] = trial_spikes

                            # Only need to do this once since its the same across neurons
                            if u == 0: 
                                inputs_by_cond[g, c, coh_cond] = trial_inputs
                                trial_ramp_by_cond[g, c, coh_cond] = trial_ramp
                                trial_choice_by_cond[g, c, coh_cond] = trial_choice
                        else:
                            raster_by_cond[g, u, c, coh_cond] = np.vstack((raster_by_cond[g, u, c, coh_cond], trial_spikes))

                            if u == 0: 
                                inputs_by_cond[g, c, coh_cond] = np.vstack((inputs_by_cond[g, c, coh_cond], trial_inputs))
                                trial_ramp_by_cond[g, c, coh_cond] = np.vstack((trial_ramp_by_cond[g, c, coh_cond], trial_ramp))
                                trial_choice_by_cond[g, c, coh_cond] = np.vstack((trial_choice_by_cond[g, c, coh_cond], trial_choice))
                        
                        if c == 0:
                            if raster_by_cond_t1[g, u, coh_cond] is None:
                                raster_by_cond_t1[g, u, coh_cond] = trial_spikes

                                if u == 0:
                                    inputs_by_cond_t1[g, coh_cond] = trial_inputs
                                    trial_ramp_by_cond_t1[g, coh_cond] = trial_ramp
                                    trial_choice_by_cond_t1[g, coh_cond] = trial_choice
                            else:
                                raster_by_cond_t1[g, u, coh_cond] = np.vstack((raster_by_cond_t1[g, u, coh_cond], trial_spikes))
                                
                                if u == 0:
                                    inputs_by_cond_t1[g, coh_cond] = np.vstack((inputs_by_cond_t1[g, coh_cond], trial_inputs))
                                    trial_ramp_by_cond_t1[g, coh_cond] = np.vstack((trial_ramp_by_cond_t1[g, coh_cond], trial_ramp))
                                    trial_choice_by_cond_t1[g, coh_cond] = np.vstack((trial_choice_by_cond_t1[g, coh_cond], trial_choice))
                        else:
                            assert c == 1, "Condition index should be 0 or 1."
                            if raster_by_cond_t2[g, u, coh_cond] is None:
                                raster_by_cond_t2[g, u, coh_cond] = trial_spikes

                                if u == 0:
                                    inputs_by_cond_t2[g, coh_cond] = trial_inputs
                                    trial_ramp_by_cond_t2[g, coh_cond] = trial_ramp
                                    trial_choice_by_cond_t2[g, coh_cond] = trial_choice
                            else:
                                raster_by_cond_t2[g, u, coh_cond] = np.vstack((raster_by_cond_t2[g, u, coh_cond], trial_spikes))
                                
                                if u == 0:
                                    inputs_by_cond_t2[g, coh_cond] = np.vstack((inputs_by_cond_t2[g, coh_cond], trial_inputs))
                                    trial_ramp_by_cond_t2[g, coh_cond] = np.vstack((trial_ramp_by_cond_t2[g, coh_cond], trial_ramp))
                                    trial_choice_by_cond_t2[g, coh_cond] = np.vstack((trial_choice_by_cond_t2[g, coh_cond], trial_choice))

    by_cond = {'raster_by_cond': raster_by_cond, 'inputs_by_cond': inputs_by_cond, 'trial_ramp_by_cond': trial_ramp_by_cond, 'trial_choice_by_cond': trial_choice_by_cond}
    by_cond_t1 = {'raster_by_cond_t1': raster_by_cond_t1, 'inputs_by_cond_t1': inputs_by_cond_t1, 'trial_ramp_by_cond_t1': trial_ramp_by_cond_t1, 'trial_choice_by_cond_t1': trial_choice_by_cond_t1}
    by_cond_t2 = {'raster_by_cond_t2': raster_by_cond_t2, 'inputs_by_cond_t2': inputs_by_cond_t2, 'trial_ramp_by_cond_t2': trial_ramp_by_cond_t2, 'trial_choice_by_cond_t2': trial_choice_by_cond_t2}
               
    return by_cond, by_cond_t1, by_cond_t2
        
