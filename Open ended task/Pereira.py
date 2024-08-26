import Utils
import os
import random
import scipy
import numpy as np

# Mute warnings which appear whem we try to read the .mat files
import warnings
warnings.filterwarnings("ignore")

EXPs = ['EXP1', 'EXP2', 'EXP3']

data_file_name = {
    'EXP1': 'data_180concepts_sentences',
    'EXP2': 'data_384sentences',
    'EXP3': 'data_243sentences',
}

num_concepts = {
    'EXP1': 180,
    'EXP2': 384,
    'EXP3': 243,
}

key_concepts = {
    'EXP1': 'keyConcept',
    'EXP2': 'keySentences',
    'EXP3': 'keySentences',
}

key_fMRI = {
    'EXP1': 'examples',
    'EXP2': 'examples_passagesentences',
    'EXP3': 'examples_passagesentences',
}

def list_folders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and 'ipynb_checkpoints' not in name]

participants = {EXP: list_folders(f'Unformatted Data/Pereira {EXP}') for EXP in EXPs}
num_participants = {EXP: len(participants[EXP]) for EXP in EXPs}

def get_random_voxels(EXP, num=512, seed=42):
    # Find all coordinates sharedd by all participants
    all_coords = None
    for participant in participants[EXP]:
        filename = f'Unformatted Data/Pereira {EXP}/{participant}/{data_file_name[EXP]}.mat'
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        meta = mat['meta'].item()
        colToCoord = meta[5]
        # Find all coordinates used by the participant
        participant_coords = {tuple(coord) for coord in colToCoord}
        if all_coords is None:
            all_coords = participant_coords
        else:
            all_coords &= participant_coords

    print(f'{EXP} has {len(all_coords)} shared voxels.')

    # Select random coordinates
    print(f'Selecting {num} random voxels using seed={seed}...')
    random.seed(seed)
    random_coords = np.array(random.sample(list(all_coords), num))
    
    random_voxels = dict()
    for participant in participants[EXP]:
        filename = f'Unformatted Data/Pereira {EXP}/{participant}/{data_file_name[EXP]}.mat'
        mat = scipy.io.loadmat(filename, squeeze_me=True)
        meta = mat['meta'].item()
        coordToCol = meta[6]
        # Get the voxel columns
        x, y, z = random_coords.T
        random_voxels[participant] = coordToCol[x-1,y-1,z-1]-1

    return random_voxels
    

def save_data(num=512, seed=42):
    for EXP in EXPs:
        print(f'Saving Pereira {EXP}:')
        
        print('Getting random voxels...')
        random_voxels = get_random_voxels(EXP, num=num, seed=seed)
        
        print('Getting fMRI data...')
        concepts_data = dict()
        all_concepts = None
        for participant in participants[EXP]:
            filename = f'Unformatted Data/Pereira {EXP}/{participant}/{data_file_name[EXP]}.mat'
            mat = scipy.io.loadmat(filename, squeeze_me=True)
            # Get all concepts (the same list for each participant)
            if all_concepts is None:
                all_concepts = mat[key_concepts[EXP]]
            # Get fMRI data for the random voxels (minus the average for the participant) 
            concept_fMRI = mat[key_fMRI[EXP]][:,random_voxels[participant]]
            mean_fMRI = np.mean(concept_fMRI, axis=0)
            concept_fMRI = concept_fMRI - mean_fMRI
            # Store the fMRI data
            concepts_data[participant] = {concept: concept_fMRI[concept_idx] for concept_idx, concept in enumerate(all_concepts)}

        print(f'Saving Pereira {EXP} to .pkl files...')
        folder = f'Data/Pereira {EXP}'
        os.makedirs(folder, exist_ok=True)
        
        # Save the columns of the random voxels for each participant
        Utils.dump_pickle(random_voxels, f'{folder}/Pereira {EXP} voxels.pkl')

        # Save the list of all the concepts
        Utils.dump_pickle(all_concepts, f'{folder}/Pereira {EXP} concepts.pkl')

        # Save the fMRI data of each participant
        Utils.dump_pickle(concepts_data, f'{folder}/Pereira {EXP}.pkl')

        print(f'Finished Pereira {EXP}.')
        print(f'{num_participants[EXP]} participants in total.')
        print('*'*50)

    print('Finished Pereira.')





        