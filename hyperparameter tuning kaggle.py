import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import json
import pickle
import numpy as np

# Read data
train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')

# Algorithms curated for the Titanic dataset
algorithms = ['xgb', 'rf', 'logreg', 'adaboost', 'naive_bayes']


# Generate common TPOT parameters
def generate_tpot_params():
    return {
        'generations': 50,
        'population_size': 50,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 5,
        'cv': 10,
        'mutation_rate': 0.8,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 15,
        'max_time_mins': 5,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    }

# Generate TPOT parameters for each algorithm
tpot_params = {algo: generate_tpot_params() for algo in algorithms}

# Dictionary to store the best models
best_models = {}

# Iterate over algorithms and perform tuning
for algo in algorithms:
    print(f"Training model for {algo}...")
    model = TPOTClassifier(**tpot_params[algo])

    X = train_df.drop(['Survived', 'PassengerId'], axis=1)
    y = train_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model.fit(X_train, y_train)

# Save full pipeline and hyperparameters to files
for algo, model in best_models.items():
    hyperparams = model.fitted_pipeline_[-1].get_params()
    hyperparams = {key.replace('classifier__', ''): value for key, value in hyperparams.items()}
    
    json_filename = f'best_{algo}_params.json'
    with open(json_filename, 'w') as f:
        json.dump(hyperparams, f, indent=4)

    pickle_filename = f'best_{algo}_pipeline.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(model.fitted_pipeline_, f)

    print(f"Best {algo} Hyperparameters saved to {json_filename}.")
    print(f"Best {algo} Pipeline saved to {pickle_filename}.")
