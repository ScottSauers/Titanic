import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import json
import pickle

# Try to load feature-engineered training datasets
try:
    train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')
    train_no_int_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered_no_interaction.csv')
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")

# Prepare datasets
datasets = {
    'xgb': train_df,
    'rf': train_df,
    'svc': train_df,
    'knn': train_df,
    'logreg': train_no_int_df
}

# Define TPOTClassifier parameters for each algorithm with early_stop
tpot_params = {
    'xgb': {
        'generations': 50,
        'population_size': 150,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 4,
        'cv': 10,
        'mutation_rate': 0.7,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 10,
        'max_time_mins': 20,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    },
    'rf': {
        'generations': 50,
        'population_size': 150,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 4,
        'cv': 10,
        'mutation_rate': 0.7,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 10,
        'max_time_mins': 20,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    },
    'svc': {
        'generations': 50,
        'population_size': 150,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 4,
        'cv': 10,
        'mutation_rate': 0.7,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 10,
        'max_time_mins': 20,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    },
    'knn': {
        'generations': 50,
        'population_size': 150,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 4,
        'cv': 10,
        'mutation_rate': 0.7,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 10,
        'max_time_mins': 20,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    },
    'logreg': {
        'generations': 50,
        'population_size': 150,
        'scoring': 'accuracy',
        'verbosity': 2,
        'early_stop': 4,
        'cv': 10,
        'mutation_rate': 0.7,
        'crossover_rate': 0.2,
        'warm_start': False,
        'memory': 'auto',
        'n_jobs': -1,
        'max_eval_time_mins': 10,
        'max_time_mins': 20,
        'periodic_checkpoint_folder': '/Users/scott/Downloads/titanic',
    }
}

# Dictionary to store the best models for each algorithm
best_models = {}

# Iterate over algorithms and perform automatic hyperparameter tuning using TPOT
for algo, data in datasets.items():
    model = TPOTClassifier(**tpot_params[algo])
    
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)

    best_models[algo] = model


# Save full pipeline and hyperparameters to files
for algo, model in best_models.items():
    hyperparams = model.fitted_pipeline_[-1].get_params()  
    hyperparams = {key.replace('classifier__', ''): value for key, value in hyperparams.items()}
    json_filename = f'best_{algo}_params.json'
    with open(json_filename, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Save the entire pipeline as well
    pickle_filename = f'best_{algo}_pipeline.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump(model.fitted_pipeline_, f)

    print(f"Best {algo} Hyperparameters saved to {json_filename}.")
    print(f"Best {algo} Pipeline saved to {pickle_filename}.")
