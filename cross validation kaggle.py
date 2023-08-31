# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
from sklearn.model_selection import cross_val_score, StratifiedKFold  # For k-fold cross-validation and stratification
from xgboost import XGBClassifier  # Extreme Gradient Boosting Classifier
from sklearn.linear_model import LogisticRegression  # Logistic Regression Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.svm import SVC  # Support Vector Machine Classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors Classifier
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # For creating a pipeline of transformations
from sklearn.compose import ColumnTransformer  # For applying transformers to columns
from scipy import stats  # For hypothesis testing
import json  # For reading and writing JSON files

# Function to remove 'classifier__' prefix from hyperparameter keys
# This is useful for ensuring compatibility when we pass parameters to a model in a pipeline
def remove_prefix(params):
    return {key.replace('classifier__', ''): value for key, value in params.items()}

# Load the feature-engineered training dataset
# Catch FileNotFoundError in case the file path is incorrect
try:
    train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")

# Check if 'Survived' column exists, as it is our target variable
if 'Survived' not in train_df.columns:
    print("Column 'Survived' not found in the dataset.")

# Separate features and target variable
# X contains all columns except 'Survived', y contains 'Survived'
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Create a pipeline for numerical features
# 1. SimpleImputer replaces missing values with the mean
# 2. StandardScaler scales the features to have zero mean and unit variance
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handling missing values
    ('scaler', StandardScaler())  # Scaling the feature values
])

# Assuming all features are numerical, apply the numerical pipeline to all columns
preprocessor = ColumnTransformer([
    ('num', num_pipeline, list(X.columns))
])

# Set up k-fold cross-validation
# StratifiedKFold ensures that each fold has the same proportion of classes as the entire dataset
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize dictionaries to store model performance and k-fold CV scores
model_performance = {}
model_scores = {}

# List of model names
model_names = ['xgb', 'logreg', 'rf', 'svc', 'knn']

# Loop through each model
# 1. Load best hyperparameters from a JSON file
# 2. Perform k-fold CV and store performance
for name in model_names:
    # Try to open the JSON file containing the best parameters for the model
    try:
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue
    
    # Remove the prefix 'classifier__' from the parameter names
    best_params = remove_prefix(best_params)
    
    # Create a pipeline for each model type, incorporating the best parameters
    # Initialize the chosen classifier with the best hyperparameters
    if name == 'xgb':
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(**best_params))])
    elif name == 'logreg':
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(**best_params))])
    elif name == 'rf':
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**best_params))])
    elif name == 'svc':
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', SVC(**best_params))])
    elif name == 'knn':
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(**best_params))])
    
    # Perform k-fold cross-validation using the pipeline
    # Scoring is based on accuracy
    cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')
    
    # Calculate the mean score across all k folds
    mean_score = np.mean(cv_scores)
    model_performance[name] = mean_score
    model_scores[name] = cv_scores
    print(f"{name.upper()} k-fold CV Mean Score: {mean_score}")

# Determine the best model based on mean k-fold cross-validation score
best_model = max(model_performance, key=model_performance.get)
print(f"The best model is {best_model.upper()} with a k-fold CV mean score of {model_performance[best_model]}")

# Perform hypothesis testing to compare the best model with other models
# A paired t-test is performed between the k-fold CV scores of the best model and each other model
# t-statistic and p-value are calculated
for name, scores in model_scores.items():
    if name == best_model:
        continue
    t_stat, p_val = stats.ttest_rel(model_scores[best_model], scores)
    print(f"Paired t-test between {best_model.upper()} and {name.upper()}: t-statistic = {t_stat}, p-value = {p_val}")