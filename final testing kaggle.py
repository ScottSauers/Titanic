import pandas as pd
import json
import pickle
from sklearn.base import BaseEstimator

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
    train_no_int_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered_no_interaction.csv')
    test_df = pd.read_csv('/Users/scott/Downloads/titanic/test_engineered.csv').drop('Survived', axis=1)
    test_df_X2 = pd.read_csv('/Users/scott/Downloads/titanic/test_engineered_no_interaction.csv').drop('Survived', axis=1)
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")

datasets = {
    'xgb': train_df.drop('PassengerId', axis=1),
    'rf': train_df.drop('PassengerId', axis=1),
    'svc': train_df.drop('PassengerId', axis=1),
    'knn': train_df.drop('PassengerId', axis=1),
    'logreg': train_no_int_df.drop('PassengerId', axis=1)
}

test_datasets = {
    'xgb': test_df,
    'rf': test_df,
    'svc': test_df,
    'knn': test_df,
    'logreg': test_df_X2
}

algorithms = ['xgb', 'rf', 'svc', 'knn', 'logreg']

# Train models using entire datasets and make predictions
for algo in algorithms:
    # Load hyperparameters
    json_filename = f'best_{algo}_params.json'
    with open(json_filename, 'r') as f:
        hyperparams = json.load(f)

    # Load pipeline
    pickle_filename = f'best_{algo}_pipeline.pkl'
    with open(pickle_filename, 'rb') as f:
        pipeline = pickle.load(f)

    # Update pipeline with saved hyperparameters
    classifier = pipeline.steps[-1][1]
    wrapper = ClassifierWrapper(classifier, hyperparams)
    pipeline.steps[-1] = (pipeline.steps[-1][0], wrapper)

    # Prepare training dataset
    data = datasets[algo]
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Fit model on entire dataset
    pipeline.fit(X, y)

    # Prepare test dataset
    test_data = test_datasets[algo]
    test_X = test_data.drop('PassengerId', axis=1)

    # Make predictions
    predictions = pipeline.predict(test_X)

    # Create Kaggle submission file
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'].astype('int32'),
        'Survived': predictions.astype('int32')
    })
    
    submission_file = f'{algo}_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"Kaggle submission file for {algo} saved as {submission_file}.")
