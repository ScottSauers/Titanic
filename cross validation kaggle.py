from sklearn.model_selection import cross_val_score
import pandas as pd
import json
import pickle
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# Custom class to allow setting parameters in pipeline
class ClassifierWrapper(BaseEstimator):
    def __init__(self, classifier, params):
        self.classifier = classifier
        self.params = params

    def fit(self, X, y):
        self.classifier.set_params(**self.params)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

# Load datasets
try:
    train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")

# Add additional algorithms to datasets and algorithms list
datasets = {
    'xgb': train_df,
    'rf': train_df,
    'svc': train_df,
    'logreg': train_df,
    'adaboost': train_df,}

algorithms = ['xgb', 'rf', 'svc', 'logreg', 'adaboost']

# Load saved hyperparameters and pipelines, and perform CV
for algo in algorithms:
    # Load hyperparameters
    json_filename = f'best_{algo}_params.json'
    try:
        with open(json_filename, 'r') as f:
            hyperparams = json.load(f)
    except FileNotFoundError:
        print(f"JSON file {json_filename} not found. Skipping {algo}.")
        continue

    # Load pipeline
    pickle_filename = f'best_{algo}_pipeline.pkl'
    try:
        with open(pickle_filename, 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        print(f"Pickle file {pickle_filename} not found. Skipping {algo}.")
        continue

    # Update pipeline with saved hyperparameters
    classifier = pipeline.steps[-1][1]
    wrapper = ClassifierWrapper(classifier, hyperparams)
    pipeline.steps[-1] = (pipeline.steps[-1][0], wrapper)

    # Prepare dataset
    data = datasets[algo]
    X = data.drop(['Survived', 'PassengerId'], axis=1)
    y = data['Survived']

    # Perform CV
    cv_scores = cross_val_score(pipeline, X, y, cv=20, scoring='accuracy')
    mean_score = round(cv_scores.mean(), 3)
    std_dev = cv_scores.std()
    std_error = std_dev / np.sqrt(len(cv_scores))

    # 95% CI calculation
    ci = round(1.96 * std_error, 3)
    
    print(f"Mean CV Accuracy for {algo}: {mean_score} ± {ci}")
