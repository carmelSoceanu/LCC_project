import Utils
import os
import numpy as np
import pandas as pd
import h5py
import bisect

participants_df = pd.read_csv('Unformatted Data/Toffolo/participants.tsv', sep='\t')
# Get participant IDs and count
participant_ids = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-16', 'sub-17', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24']
# Excluding sub-02 for formatting reasons
participant_ids.remove('sub-02')

num_participants = len(participant_ids)

files_df = pd.read_csv('Unformatted Data/Toffolo/N400Stimset_stimuli_parameters.tsv', sep='\t')

# Get example file names
example_files = ['NPI_cup(flag).wav', 'NPC_cake.wav']

# Get file names of valid sentences according to them
valid_files = files_df[files_df['linguistic-group_div'] != 5]['stim_file'].tolist()

# Remove examples
valid_files = [file for file in valid_files if file not in example_files]

# Get the time axis
sub = participant_ids[0]
with h5py.File( f'Unformatted Data/Toffolo/derivatives/erps/{sub}/{sub}_task-N400Stimset_erp-GA.mat', 'r') as mat_data:
    # Load the time data
    time_axis = np.array([time[0] for time in mat_data['t']])

# Types for the EEG struct
stimuli_types = [1, 253]

def struct_to_dict(mat_data, struct):
    result = dict()
    for key in struct.keys():
        vals = struct[key][:,0]
        result[key] = [mat_data[obj][()][0,0] for obj in vals]
    return result

def struct_to_dataframe(mat_data, struct, with_index=False):
    df = pd.DataFrame(struct_to_dict(mat_data, struct))
    
    if with_index:
        df['epoch'] = range(1, len(df) + 1)

    return df

def get_EEG(mat_data, participant, condition, avg_window=(0, len(time_axis))):
    # Load the relevant ERP cell
    erps = mat_data['ERPs']
    condition_index = 0 if condition == 'NPC' else 1
    erp = mat_data[erps[0][condition_index]]

    # Load the stimuli data in order
    relevant_cols = ['stim_file', '1', '2', '3', '4', '5', '6', '7', '8', 'linguistic-group_div']
    stimuli_df = pd.read_csv('Unformatted Data/Toffolo/N400Stimset_stimuli_parameters.tsv', sep='\t')
    stimuli_df = stimuli_df[relevant_cols].reset_index(drop=True)
    # stimuli_df = stimuli_df[stimuli_df['linguistic-group_div'] != 5].reset_index(drop=True)
    stimuli_df = pd.concat([stimuli_df.iloc[-2:], stimuli_df.iloc[:-2]]).reset_index(drop=True) # Moving the examples from the bottom to the top

    # Load the EEG event data
    event = struct_to_dataframe(mat_data, erp['event'])
    urevent = struct_to_dataframe(mat_data, erp['urevent'], with_index=True)
    
    # Keep only the stimuli in urevent
    urevent = urevent[urevent['type'].isin(stimuli_types)].reset_index(drop=True)
    
    # Attach the stimuli data to urevent
    urevent = pd.concat([urevent, stimuli_df], axis=1)
    
    # Keep only the relevant stimuli
    stimuli_data = urevent[urevent['epoch'].isin(event['urevent'])].reset_index(drop=True)
    
    # Load the EEG data
    EEG = np.array(erp['data']).transpose(0, 2, 1) # (S, 128, T) matrix

    # Average over the given time window
    if avg_window is not None:
        # Average the EEG vectors over the selected time window
        t_start, t_end = avg_window
        EEG = np.mean(EEG[:,:,t_start:t_end], axis=2) # (S, 128) matrix
    
    # Attach sentence info
    word_cols = ['1', '2', '3', '4', '5', '6', '7', '8']
    stimuli_data['sentence'] = stimuli_data[word_cols].apply(lambda x: ' '.join(x.dropna()), axis=1)
    
    # Attach the EEG data to the stimuli data
    EEG_list = [EEG[i] for i in range(EEG.shape[0])]
    stimuli_data['EEG_data'] = EEG_list

    stimuli_data = stimuli_data[stimuli_data['stim_file'].isin(valid_files)]
    stimuli_data = stimuli_data.reset_index(drop=True)
    
    return stimuli_data

def save_data():
    # Averaging over the N400 interval (200ms to 600ms)
    idx_200ms = bisect.bisect_left(time_axis, 200)
    idx_600ms = bisect.bisect_left(time_axis, 600)
    avg_window = (idx_200ms, idx_600ms)

    # Go over each condition we want to save and each participant
    for condition in ['NPC', 'NPI']:
        print(f'Saving Toffolo {condition}:')
        all_sentences = set()
        EEG_data = dict()
        
        for participant in participant_ids:
            EEG_data[participant] = dict()
            file_path = f'Unformatted Data/Toffolo/derivatives/erps/{participant}/{participant}_task-N400Stimset_erp-GA.mat'
            # Get the EEG data
            with h5py.File(file_path, 'r') as mat_data:
                stimuli_data = get_EEG(mat_data, participant, condition, avg_window=avg_window)
                
            for index, row in stimuli_data.iterrows():
                # Clean sentence
                sentence = row['sentence']
                sentence = sentence.replace('  ', ' ')

                EEG_data[participant][sentence] = row['EEG_data']
                all_sentences |= {sentence}
                
        all_sentences = np.array(list(all_sentences))
        
        print(f'Saving Toffolo {condition} to .pkl files...')
        folder = f'Data/Toffolo {condition}'
        os.makedirs(folder, exist_ok=True)
        
        # Save the list of all the concepts
        Utils.dump_pickle(all_sentences, f'{folder}/Toffolo {condition} concepts.pkl')

        # Save the EEG data of each participant
        Utils.dump_pickle(EEG_data, f'{folder}/Toffolo {condition}.pkl')

        print(f'Finished Toffolo {condition}.')
        print(f'{num_participants} participants in total.')
        print('*'*50)
        
    print('Finished Toffolo.')
    

                
                





