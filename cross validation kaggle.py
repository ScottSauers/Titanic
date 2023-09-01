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

# Load the feature-engineered training datasets
# Catch FileNotFoundError in case the file path is incorrect
try:
    train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')
    train_df_X2 = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered_no_interaction.csv')  # Reading the CSV file into a DataFrame
except FileNotFoundError:
    print("CSV file(s) not found. Please check the file path.")

# Check if 'Survived' column exists in both datasets, as it is our target variable
if 'Survived' not in train_df.columns or 'Survived' not in train_df_X2.columns:
    print("Column 'Survived' not found in one or both of the datasets.")

# Separate features and target variable for both datasets
# X contains all columns except 'Survived', y contains 'Survived'
# X2 and y2 are for the non-interaction terms dataset
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X2 = train_df_X2.drop('Survived', axis=1)
y2 = train_df_X2['Survived']

# Create a pipeline for numerical features
# 1. SimpleImputer replaces missing values with the mean
# 2. StandardScaler scales the features to have zero mean and unit variance
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handling missing values
    ('scaler', StandardScaler())  # Scaling the feature values
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, list(X.columns))
])

# Create another preprocessor for X2
preprocessor_X2 = ColumnTransformer([
    ('num', num_pipeline, list(X2.columns))
])

# Set up k-fold cross-validation
# StratifiedKFold ensures that each fold has the same proportion of classes as the entire dataset
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize dictionaries to store model performance and k-fold CV scores
model_performance = {}
model_scores = {}

# List of model names for interaction and non-interaction terms
model_names_with_interaction = ['logreg']
model_names_without_interaction = ['xgb', 'rf', 'svc', 'knn']

# Loop through each model with interaction terms
for name in model_names_with_interaction:
    try:
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue

    best_params = remove_prefix(best_params)

    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(**best_params))])

    cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy')

    mean_score = np.mean(cv_scores)
    model_performance[name] = mean_score
    model_scores[name] = cv_scores
    print(f"{name.upper()} k-fold CV Mean Score: {mean_score}")

# Loop through each model without interaction terms
for name in model_names_without_interaction:
    try:
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue

    best_params = remove_prefix(best_params)

    if name == 'xgb':
        pipeline = Pipeline([('preprocessor', preprocessor_X2), ('classifier', XGBClassifier(**best_params))])
    elif name == 'rf':
        pipeline = Pipeline([('preprocessor', preprocessor_X2), ('classifier', RandomForestClassifier(**best_params))])
    elif name == 'svc':
        pipeline = Pipeline([('preprocessor', preprocessor_X2), ('classifier', SVC(**best_params))])
    elif name == 'knn':
        pipeline = Pipeline([('preprocessor', preprocessor_X2), ('classifier', KNeighborsClassifier(**best_params))])


    cv_scores = cross_val_score(pipeline, X2, y2, cv=kfold, scoring='accuracy')

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
