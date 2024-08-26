import pickle
import numpy as np

dataset_names = ['Mitchell', 'Pereira EXP1', 'Pereira EXP2', 'Pereira EXP3', 'Toffolo NPC', 'Toffolo NPI']

model_names = ['bert', 'roberta', 'distilbert', 'xlnet', 'albert', 'gpt2', 't5', 'bart', 'electra',
               'deberta', 'elmo', 'sbert', 'inferSent', 'word2vec', 'glove-300', 'glove-200', 'glove-100', 'glove-50']

hidden_sizes = {
    50: [20],
    100: [40],
    200: [75],
    300: [100] ,
    384: [150],
    768: [300],
    850: [300],
    1024: [400]
}

# Save object to a .pkl file
def dump_pickle(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

# Load object from a .pkl file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Get all of the datasets
def get_datasets():
    # The names of the datasets    
    datasets = dict()
    for dataset_name in dataset_names:
        dataset = dict()
        # Load the dataset's neural data
        dataset['data'] = load_pickle(f'Data/{dataset_name}/{dataset_name}.pkl')
        # Load the dataset's additional info
        for info in ['cleaner', 'embeddings']:
            dataset[info] = load_pickle(f'Data/{dataset_name}/{dataset_name} {info}.pkl')
        # Clean and rescale the dataset
        dataset['data'] = get_clean_data(dataset)
        dataset['concepts'] = list(dataset['cleaner'].values())
        # Add and update info to the dataset
        dataset['participants'] = list(dataset['data'].keys())
        dataset['num_participants'] = len(dataset['participants'])
        dataset['modality'] = 'EEG' if 'Toffolo' in dataset_name else 'fMRI'
        dataset['type'] = 'word' if 'Mitchell' in dataset_name or 'EXP1' in dataset_name else 'sentence'
        # Add the dataset
        datasets[dataset_name] = dataset
    return datasets

# Get min and max of participant data
def get_data_range(participant_data):
    all_data = np.concatenate(list(participant_data.values()))
    return all_data.min(), all_data.max()

# Cleans the concepts and rescales the data of the participant
def get_clean_participant_data(dataset, participant):
    participant_data = dataset['data'][participant]
    data_min, data_max = get_data_range(participant_data)
    participant_data = {dataset['cleaner'][concept]: (data-data_min)/(data_max-data_min) for concept, data in participant_data.items()}
    return participant_data

# Clean entire dataset
def get_clean_data(dataset):
    cleaned_dataset = {participant:get_clean_participant_data(dataset, participant) for participant in dataset['data']}
    return cleaned_dataset

# Get the embedding vector size of an embedding model
def get_model_embedding_size(model):
    # Returns the size of the first vector, since all embedding vectors are of the same size
    return len(list(model.values())[0])