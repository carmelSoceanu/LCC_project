import Utils
import os
import random
import scipy
import numpy as np

num_participants = 9
participants = [f'P{p+1}' for p in range(num_participants)]

def get_random_voxels(num=512, seed=42):
     # Find all coordinates sharedd by all participants
    all_coords = None
    for participant in participants:
        filename = f'Unformatted Data/Mitchell/data-science-{participant}.mat'
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        meta = mat['meta'].item()
        colToCoord = meta[7]
        # Find all coordinates used by the participant
        participant_coords = {tuple(coord) for coord in colToCoord}
        if all_coords is None:
            all_coords = participant_coords
        else:
            all_coords &= participant_coords

    print(f'Mitchell has {len(all_coords)} shared voxels.')

    # Select random coordinates
    print(f'Selecting {num} random voxels using seed={seed}...')
    random.seed(seed)
    random_coords = np.array(random.sample(list(all_coords), num))
    
    random_voxels = dict()
    for participant in participants:
        filename = f'Unformatted Data/Mitchell/data-science-{participant}.mat'
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        meta = mat['meta'].item()
        coordToCol = meta[8]
        # Get the voxel columns
        x, y, z = random_coords.T
        random_voxels[participant] = coordToCol[x-1,y-1,z-1]-1

    return random_voxels

def save_data(num=512, seed=42):
    print(f'Saving Mitchell:')
    
    print('Getting random voxels...')
    random_voxels = get_random_voxels(num=num, seed=seed)

    print('Getting fMRI data...')
    concepts_data = dict()
    all_concepts = None
    for participant in participants:
        filename = f'Unformatted Data/Mitchell/data-science-{participant}.mat'
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        # Get all concepts (the same list for each participant)
        if all_concepts is None:
            all_concepts = list({info[2] for info in mat['info']})
    
        # Get fMRI data for the random voxels (minus the average for the participant) 
        fMRI_data = np.array(list(mat['data']))
        fMRI_data = fMRI_data[:,random_voxels[participant]]
        mean_fMRI = np.mean(fMRI_data, axis=0)

        # Average over all epochs (each concept is presented 6 times) and subtract the average fMRI state 
        concepts_data[participant] = {concept:[] for concept in all_concepts}
        for info, data in zip(mat['info'], fMRI_data):
            concept = info[2]
            concepts_data[participant][concept] += [data]
            
        for concept in all_concepts:
            concepts_data[participant][concept] = np.mean(np.array(concepts_data[participant][concept]), axis=0) - mean_fMRI

    print(f'Saving Mitchell to .pkl files...')
    folder = f'Data/Mitchell'
    os.makedirs(folder, exist_ok=True)
    
    # Save the columns of the random voxels for each participant
    Utils.dump_pickle(random_voxels, f'{folder}/Mitchell voxels.pkl')

    # Save the list of all the concepts
    Utils.dump_pickle(all_concepts, f'{folder}/Mitchell concepts.pkl')

    # Save the fMRI data of each participant
    Utils.dump_pickle(concepts_data, f'{folder}/Mitchell.pkl')
    
    print(f'Finished Mitchell.')
    print(f'{num_participants} participants in total.')
    print('*'*50)

    print('Finished Mitchell.')
    









    