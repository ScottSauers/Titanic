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

# Function to load file safely
def safe_file_load(file_path, file_type):
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_type == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        print(f"{file_type.upper()} file not found: {file_path}")
        return None

# Load datasets
train_df = safe_file_load('/Users/scott/Downloads/titanic/train_engineered.csv', 'csv')
test_df = safe_file_load('/Users/scott/Downloads/titanic/test_engineered.csv', 'csv')

# Exit if datasets are not loaded
if train_df is None or test_df is None:
    exit()

# Drop the 'Survived' column if exists in the test dataset
test_df.drop('Survived', axis=1, errors='ignore', inplace=True)

# Prepare data
train_data = train_df.drop('PassengerId', axis=1)
test_data = test_df.drop('PassengerId', axis=1)
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']

# Add additional algorithms to the list
algorithms = ['xgb', 'rf', 'svc', 'logreg', 'adaboost']

# Train models using entire datasets and make predictions
for algo in algorithms:
    # Load hyperparameters
    hyperparams = safe_file_load(f'best_{algo}_params.json', 'json')
    if hyperparams is None:
        continue

    # Load pipeline
    pipeline = safe_file_load(f'best_{algo}_pipeline.pkl', 'pickle')
    if pipeline is None:
        continue

    # Update pipeline with saved hyperparameters
    classifier = pipeline.named_steps.get('classifier', pipeline.steps[-1][1])
    wrapper = ClassifierWrapper(classifier, hyperparams)
    pipeline.steps[-1] = (pipeline.steps[-1][0], wrapper)

    # Fit model on entire dataset
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(test_data)

    # Create Kaggle submission file
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'].astype('int32'),
        'Survived': predictions.astype('int32')
    })

    submission_file = f'{algo}_submission.csv'
    submission.to_csv(submission_file, index=False)

    print(f"Kaggle submission file for {algo} saved as {submission_file}.")
