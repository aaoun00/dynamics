import os
import numpy as np
from src.loading import load_fira_mat_file
from src.format import organize_raster_by_condition

if __name__ == "__main__":
    folder_path = 'data/prearcuate_gyrus'
    model_output_path = 'model_output'
    
    if not os.path.exists(model_output_path):
        os.mkdir(model_output_path)

    emissions_save_path = model_output_path + '/emissions'
    inputs_save_path = model_output_path + '/inputs'

    if not os.path.exists(emissions_save_path):
        os.mkdir(emissions_save_path)
    if not os.path.exists(inputs_save_path):
        os.mkdir(inputs_save_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            print(file_name)
            fpath = os.path.join(folder_path, file_name)
            animal_name = file_name.split('.')[0]
            animal_emissions_save_path = emissions_save_path + '/' + str(animal_name)
            animal_inputs_save_path = inputs_save_path + '/' + str(animal_name)

            if not os.path.exists(animal_emissions_save_path):
                os.mkdir(animal_emissions_save_path)
            if not os.path.exists(animal_inputs_save_path):
                os.mkdir(animal_inputs_save_path)

            # full_align_struct = [
            #     # {'event': 'targ_on', 'start_offset': -300, 'end_offset': 500, 'limits': [['in_fp_wd', 0], ['dots_on_screen', 0]]},
            #     # {'event': 'dots_on_screen', 'start_offset': -300, 'end_offset': 1000, 'limits': [['targ_on', 0], ['out_fp_wd', -50]]},
            #     {'event': 'out_fp_wd', 'start_offset': -1000, 'end_offset': 300, 'limits': [['dots_on_screen', 200], ['feedback', 0]]},
            #     # {'event': 'feedback', 'start_offset': -250, 'end_offset': 500, 'limits': [['out_fp_wd', 100], ['feedback', 500]]},
            #     # # {'event': 'targ_on', 'start_offset': -300, 'end_offset': 9999, 'limits': [['in_fp_wd', 0], ['feedback', 500]]}
            # ]

            # full_align_struct = [
            #     # {'event': 'dots_on_screen', 'start_offset': -300, 'end_offset': 1000, 'limits': [['dots_on_screen', 150], ['out_fp_wd', -50]]},
            #     {'event': 'dots_on_screen', 'start_offset': +150, 'end_offset': 1000, 'limits': [['dots_on_screen', 150], ['out_fp_wd', -50]]},
            #     # {'event': 'dots_on_screen', 'start_offset': -200, 'end_offset': 100, 'limits': [[None, None], [None, None]]},
            #     # {'event': 'dots_on_screen', 'start_offset': +150, 'end_offset': 1000, 'limits': [['targ_on', 0], ['out_fp_wd', -50]]},
            # ]

            full_align_struct = [
                # {'event': 'dots_on_screen', 'start_offset': 0, 'end_offset': 1000, 'limits': [['dots_on_screen', +150], ['out_fp_wd', -50]]},
                {'event': 'dots_on_screen', 'start_offset': 150, 'end_offset': 1000, 'limits': [['dots_on_screen', 150], ['out_fp_wd', -50]]},
                # {'event': 'dots_on_screen', 'start_offset': 250, 'end_offset': 600, 'limits': [['dots_on_screen', 250], ['out_fp_wd', -50]]},
            ]

            # full_align_struct = [
            #     {'event': 'dots_on_screen', 'start_offset': +150, 'end_offset': 1000, 'limits': [[None, None], [None, None]]},
            # ]

            window_size = 1 # ms
                                
            raster, ramps, choice_pulse, alignto, unit_id, coh, trials_cor, coh_group, coh_set = load_fira_mat_file(fpath, full_align_struct)

            coh_dims = len(coh_group)

            by_cond, by_cond_t1, by_cond_t2 = organize_raster_by_condition(raster, ramps, choice_pulse, coh, trials_cor, coh_set=None, window_size=window_size)

            raster_by_cond = by_cond['raster_by_cond']
            raster_by_cond_t1 = by_cond_t1['raster_by_cond_t1']
            raster_by_cond_t2 = by_cond_t2['raster_by_cond_t2']

            inputs_by_cond = by_cond['inputs_by_cond']
            inputs_by_cond_t1 = by_cond_t1['inputs_by_cond_t1']
            inputs_by_cond_t2 = by_cond_t2['inputs_by_cond_t2']

            trial_ramp_by_cond = by_cond['trial_ramp_by_cond']
            trial_ramp_by_cond_t1 = by_cond_t1['trial_ramp_by_cond_t1']
            trial_ramp_by_cond_t2 = by_cond_t2['trial_ramp_by_cond_t2']

            trial_choice_by_cond = by_cond['trial_choice_by_cond']
            trial_choice_by_cond_t1 = by_cond_t1['trial_choice_by_cond_t1']
            trial_choice_by_cond_t2 = by_cond_t2['trial_choice_by_cond_t2']

            epoch1_t1_call = raster_by_cond_t1[0]
            epoch1_t2_call = raster_by_cond_t2[0]
            
            # epoch2_t1_call = raster_by_cond_t1[1]
            # epoch2_t2_call = raster_by_cond_t2[1]

            # epoch3_t1_call = raster_by_cond_t1[2]
            # epoch3_t2_call = raster_by_cond_t2[2]

            # epoch4_t1_call = raster_by_cond_t1[3]
            # epoch4_t2_call = raster_by_cond_t2[3]

            # fullepoch_t1_call = raster_by_cond_t1[4]
            # fullepoch_t2_call = raster_by_cond_t2[4]

            epoch1_call = raster_by_cond[0]
            # epoch2_call = raster_by_cond[1]
            # epoch3_call = raster_by_cond[2]
            # epoch4_call = raster_by_cond[3]
            # fullepoch_call = raster_by_cond[4]

            to_save_emission_names = ['epoch1_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            'epoch1_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch2_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch2_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch3_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch3_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch4_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch4_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'fullepoch_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'fullepoch_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            'epoch1_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy'] # ,
                            # 'epoch2_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch3_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'epoch4_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # 'fullepoch_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy'],
                            # 'epoch1_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',]
                            # # 'epoch2_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # # 'epoch3_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # # 'epoch4_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy',
                            # # 'fullepoch_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_Y.npy']
            
            to_save_emissions = [epoch1_t1_call, epoch1_t2_call, 
                                #  epoch2_t1_call, epoch2_t2_call, epoch3_t1_call, epoch3_t2_call, 
            #                 epoch4_t1_call, epoch4_t2_call, fullepoch_t1_call, fullepoch_t2_call,
                            epoch1_call] # , 
                            # # epoch2_call, epoch3_call, epoch4_call, fullepoch_call,
                            # epoch1_tall_call]
                            # # epoch2_tall_call, epoch3_tall_call, epoch4_tall_call, fullepoch_tall_call]
            
            assert len(to_save_emission_names) == len(to_save_emissions), "Mismatch in emission names and emissions"

            for i in range(len(to_save_emission_names)):
                save_name = to_save_emission_names[i]
                save_var = to_save_emissions[i]
                np.save(os.path.join(animal_emissions_save_path, save_name), save_var)
                print(f'Saved {save_name} to {animal_emissions_save_path}')

            epoch1_t1_call = inputs_by_cond_t1[0]
            epoch1_t2_call = inputs_by_cond_t2[0]

            # epoch2_t1_call = inputs_by_cond_t1[1]
            # epoch2_t2_call = inputs_by_cond_t2[1]

            # epoch3_t1_call = inputs_by_cond_t1[2]
            # epoch3_t2_call = inputs_by_cond_t2[2]

            # epoch4_t1_call = inputs_by_cond_t1[3]
            # epoch4_t2_call = inputs_by_cond_t2[3]

            # fullepoch_t1_call = inputs_by_cond_t1[4]
            # fullepoch_t2_call = inputs_by_cond_t2[4]

            epoch1_call = inputs_by_cond[0]
            # epoch2_call = inputs_by_cond[1]
            # epoch3_call = inputs_by_cond[2]
            # epoch4_call = inputs_by_cond[3]
            # fullepoch_call = inputs_by_cond[4]

            to_save_input_names = ['epoch1_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            'epoch1_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch2_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch2_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch3_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch3_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch4_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch4_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'fullepoch_Tin_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'fullepoch_Tout_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            'epoch1_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy'] # ,
                            # 'epoch2_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch3_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch4_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'fullepoch_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # 'epoch1_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy']
                            # # 'epoch2_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # # 'epoch3_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # # 'epoch4_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy',
                            # # 'fullepoch_Tall_Coh' + str(coh_dims) + '_' + str(window_size) + '_U.npy']
                        
            to_save_inputs = [epoch1_t1_call, epoch1_t2_call, 
                            #   epoch2_t1_call, epoch2_t2_call, epoch3_t1_call, epoch3_t2_call, 
                            # epoch4_t1_call, epoch4_t2_call, fullepoch_t1_call, fullepoch_t2_call,
                            epoch1_call] # , 
                            # # epoch2_call, epoch3_call, epoch4_call, fullepoch_call,
                            # epoch1_tall_call]
                            # # epoch2_tall_call, epoch3_tall_call, epoch4_tall_call, fullepoch_tall_call]

            assert len(to_save_input_names) == len(to_save_inputs), "Mismatch in input names and inputs"

            for i in range(len(to_save_input_names)):
                save_name = to_save_input_names[i]
                save_var = to_save_inputs[i]
                np.save(os.path.join(animal_inputs_save_path, save_name), save_var)
                print(f'Saved {save_name} to {animal_inputs_save_path}')

            epoch1_ramp = trial_ramp_by_cond[0]
            np.save(os.path.join(animal_inputs_save_path, 'epoch1_ramp_' + str(window_size) + '.npy'), epoch1_ramp)
            print(f'Saved epoch1_ramp to {animal_inputs_save_path}')

            epoch1_choice = trial_choice_by_cond[0]
            np.save(os.path.join(animal_inputs_save_path, 'epoch1_choice_' + str(window_size) + '.npy'), epoch1_choice)
            print(f'Saved epoch1_choice to {animal_inputs_save_path}')

            # save coh_set
            np.save(os.path.join(animal_inputs_save_path, 'coh_set_' + str(window_size) + '.npy'), coh_set)
            print(f'Saved coh_set to {animal_inputs_save_path}')