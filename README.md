# Dynamics
Change repo name asap

# Notebooks

## explore.ipynb

Exploratory analysis of the data. So far this notebooks:

    - Extracts specific time points (e.g. dot onset, saccade, etc.)
    - Plots and computes median reaction time
    - Loads in spike window intervals computed from the median reaction time
        - Window 1: 150 ms from dot onset to median RT
        - Window 2: -150 ms from median RT to -50 ms from median RT
        - Window 3: -150 ms from median RT to -50 ms from saccade ???
    - Uses only correct trials and separates by Tin/Tout and coh level
    - Computes 1 psth per coherence level and per Tin/Tout (left/right)
    - Smooths PSTH for each neuron (and condition) separately (truncated kernel when kernel extends beyond edge)
    - Concatenates and fits PCA (no z-score, no detrending)
    - Plots PSTH, PSTH smooth, PSTH concat, PCA colored by coh (all time points), PCA colored by time (all coherences)

    - 2 points to clarify:
        - Currently can only use one animal, each animal has different median RT so need to change main.py to use a list of RTs 
        unique to each animal instead of a single number (valid for only one animal)
        - FIRA data structure already has rt_dots event but that is aligned to dots_on and not dots_on_screen. Matlab demo
        uses dots_on_screen not dots_on. Here we use rt = out_fp_wd - dots_on_screen instead of rt_dots (which was
        computed as rt_dots = out_fp_wd - dots_on). out_fp_wd = out fixation point window, corresponds to saccade time on
        valid trials.

    - Current obs:
        - Window 1: Tin/Tout trajectories end in different subspaces? PCA trajectories curve away at later data points
        - Window 2: Tin and Tout trajectories for different coherence levels lie along a roughly straight line? The 
        trajectories for Tin and Tout seem roughly perpendicular/lie in opposite directions
        - Window 3: pending

## rnn_gplds_pca_fit.ipynb

This notebook uses the outputs of the RNN model implemented in the Bredenberg et al. paper (hidden units, figure 1), fits a PCA
and generates single trial projections. These are then passed into the GPLDS model and extracted as latent states + fixed points
which are compared to PCA. Current version in demo_fits is all trials (correct/incorrect and Tin/Tout). 

For now GPLDS seems to recover underlying structure (fixed points on straight line and latent trajectories curve around them).

Caveats: Fixed points code still needs to change such that we have points per coh + time bin instead of per coh and across all time bins as is currently displayed

## rnn_gplds_fit.ipynb

Similar to above but fit GPLDS directly to hidden units (as opposed to single trial PC projections)
