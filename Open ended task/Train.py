import Utils
import Test
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed in order to replicate results
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Generate a random baseline of the same size as the embedding vectors
def generate_random_baseline(model):
    embedding_size = Utils.get_model_embedding_size(model)
    random_baseline_name = f'random_{embedding_size}'
    random_baseline = {concept: np.random.uniform(-1, 1, embedding_size) for concept in model}
    return random_baseline_name, random_baseline

# Train a neural network on a model and participant data, returns squared error of every concept
def train_neural_network(input_vectors, target_vectors, n_splits=5, learning_rate=0.001, momentum=0.9, epochs=100):
    concepts = list(target_vectors.keys())
    
    # Convert dictionaries to lists of tensors, and match their order
    inputs = [torch.tensor(input_vectors[concept], dtype=torch.float32).to(device) for concept in concepts]
    targets = [torch.tensor(target_vectors[concept], dtype=torch.float32).to(device) for concept in concepts]
    
    # Convert to tensor datasets
    X = torch.stack(inputs)
    y = torch.stack(targets)

    assert X.shape[0] == y.shape[0]
        
    # Get dimensions
    num_words, input_size = X.shape
    num_words, output_size = y.shape

    if input_size in Utils.hidden_sizes:
        hidden_sizes = Utils.hidden_sizes[input_size]
    else:
        hidden_sizes = [100]
    
    # Neural network architexture
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size):
            super(NeuralNetwork, self).__init__()
            
            # Create a list to store the layers
            layers = []
            
            # Add the first hidden layer
            layers.append(nn.Linear(input_size, hidden_sizes[0]))
            layers.append(nn.Tanh())
            
            # Add the remaining hidden layers
            for i in range(1, len(hidden_sizes)):
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                layers.append(nn.Tanh())
            
            # Add the output layer
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            
            # Combine all layers into a sequential module
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)

    scores = dict()
    training_data = [[0 for epoch in range(epochs)] for fold in range(n_splits)]
    
    # Cross-validation setup with a random seed
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(0, 10000))
    fold = 0
    for train_index, test_index in kf.split(X):
        model = NeuralNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(momentum, 0.999))
        criterion = nn.MSELoss()

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Store the loss on the test data
            model.eval()
            with torch.no_grad():
                loss = criterion(model(X_test), y_test)
                training_data[fold][epoch] = loss.item()
        
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            predictions_batch = model(X_test)
            for i, idx in enumerate(test_index):
                concept = concepts[idx]
                # Move to CPU before converting to numpy
                predicted_vector = predictions_batch[i]
                target_vector = y_test[i]
                # Compute MSE
                mse = ((target_vector - predicted_vector) ** 2).mean().item()
                scores[concept] = mse

        fold += 1
    
    # print(len(prediction_vectors), len(target_vectors))
    assert set(scores.keys()) == set(concepts)

    return np.array([scores[concept] for concept in concepts]), np.array(training_data)

# Train all models
def train_models(save=True, seed=42):
    # Use seed before training
    seed_all(seed)
    
    print(f'{'DATASET':^12}{'PARTICIPANT':^16}{'CONCEPTS':^8}{'MODEL':^16}VS{'BASELINE':^16}<=>{'MODEL_MSE':^16}VS{'BASELINE_MSE':^16}~{'P-VALUE':^14}{'ALPHA':^12}{'SIGNIFICANT':^12}')
    
    # Get all datasets
    datasets = Utils.get_datasets()
    results = dict()
    print('*'*150)
    for dataset_name in Utils.dataset_names:
        results[dataset_name] = dict()
    
        dataset = datasets[dataset_name]
        # Calculate the alpha with Bonferroni correction to counteract the multiple hypotheses problem
        num_hypotheses = dataset['num_participants']
        corrected_alpha = Test.bonferroni_correction(Test.alpha, num_hypotheses)
        
        for participant in dataset['participants']:
            results[dataset_name][participant] = dict()
            
            participant_data = dataset['data'][participant]
            num_concepts = len(participant_data)
            
            for model_name in Utils.model_names:
                
                model = dataset['embeddings'][model_name]
    
                # Create random baseline
                random_baseline_name, random_baseline = generate_random_baseline(model)
                
                # Train neural network on the model and the random baseline
                model_scores, model_training_data = train_neural_network(model, participant_data)
                baseline_scores, baseline_training_data = train_neural_network(random_baseline, participant_data)
    
                # Use the statistical test
                statistic, p_value = Test.test(model_scores, baseline_scores)
                significant = int(p_value < corrected_alpha)
    
                # Calculate average MSE over all concepts
                model_mean_score = np.mean(model_scores)
                baseline_mean_score = np.mean(baseline_scores)
    
                # Record result
                results[dataset_name][participant][model_name] = {
                    'num_concepts': num_concepts,
                    'model_mean_score': model_mean_score,
                    'model_training_data': model_training_data,
                    'baseline_mean_score': baseline_mean_score,
                    'baseline_training_data': baseline_training_data,
                    'significant': significant
                }
                print(f'{dataset_name:^12}{participant:^16}{num_concepts:^8}{model_name:^16}VS{random_baseline_name:^16}<=>{model_mean_score:^16.3e}VS{baseline_mean_score:^16.3e}~{p_value:^14.3e}{corrected_alpha:^12.3e}{significant:^12}')
            print('-'*150)
    # Save results
    if save:
        filename = 'Data/results.pkl'
        print(f'Saving results to {filename}')
        Utils.dump_pickle(results, filename)
    
    print('Finished.')