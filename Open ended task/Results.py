import Utils
import numpy as np

# Load the results
def load_results():
    return Utils.load_pickle('Data/results.pkl')

# Get the average performance of a certain model aggregated over all dataset participants
def get_result(results, datasets, dataset_name, model_name):
    dataset = datasets[dataset_name]

    total_model_error = 0
    total_baseline_error = 0    
    total_concepts = 0

    total_significant = 0
    num_hypotheses = dataset['num_participants']
    
    # Aggregate over all participants
    for participant in dataset['participants']:
        result = results[dataset_name][participant][model_name]
        
        num_concepts = result['num_concepts']
        
        total_model_error += result['model_mean_score'] * num_concepts
        total_baseline_error += result['baseline_mean_score'] * num_concepts
        total_concepts += num_concepts
        
        total_significant += result['significant']
    
    # Average
    model_result = total_model_error / total_concepts
    baseline_result = total_baseline_error / total_concepts
    return model_result, baseline_result, total_significant, num_hypotheses