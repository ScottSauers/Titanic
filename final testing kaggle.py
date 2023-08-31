# Importing necessary libraries and modules
import pandas as pd  # For data manipulation and analysis
import json  # For reading JSON files
from xgboost import XGBClassifier  # XGBoost classifier model
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier model
from sklearn.svm import SVC  # Support Vector Machine model
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors model
from sklearn.impute import SimpleImputer  # For missing value imputation
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.pipeline import Pipeline  # For creating pipelines
from sklearn.compose import ColumnTransformer  # For applying multiple transformers

# Function to remove the 'classifier__' prefix from hyperparameters
# This is useful when hyperparameters are nested inside a pipeline
def remove_prefix(params):
    return {key.replace('classifier__', ''): value for key, value in params.items()}

# Load the training dataset from the specified path
# This CSV file contains the training data for the Titanic dataset
train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')

# Load the test dataset from the specified path
# This CSV file contains the test data for the Titanic dataset
test_df = pd.read_csv('/Users/scott/Downloads/titanic/test_engineered.csv')

# Separate features and target variable from the training dataset
# 'Survived' is the target variable, rest are features
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df  # Test dataset only contains features

# Create a pipeline for numerical features
# First, missing values are imputed using the mean of the column
# Then, features are scaled to have mean=0 and variance=1
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Assuming all features are numerical, applying the pipeline to all columns
preprocessor = ColumnTransformer([
    ('num', num_pipeline, list(X_train.columns))
])

# List of model names for iteration
model_names = ['xgb', 'logreg', 'rf', 'svc', 'knn']

# Loop through each model, read best hyperparameters, train and generate predictions
for name in model_names:
    try:
        # Try to load the best hyperparameters for each model from a JSON file
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = remove_prefix(json.load(f))
    except FileNotFoundError:
        # If the JSON file is not found, skip this model
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue

    # Initialize the correct model type based on the name and use the best hyperparameters
    # Each model is wrapped in a pipeline that first applies preprocessor
    if name == 'xgb':
        model = Pipeline([('preprocessor', preprocessor), ('classifier', XGBClassifier(**best_params))])
    elif name == 'logreg':
        model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(**best_params))])
    elif name == 'rf':
        model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**best_params))])
    elif name == 'svc':
        model = Pipeline([('preprocessor', preprocessor), ('classifier', SVC(**best_params))])
    elif name == 'knn':
        model = Pipeline([('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(**best_params))])

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Generate predictions on the test dataset
    # Converting predictions to integer type for Kaggle submission
    predictions = model.predict(X_test).astype(int)

    # Create Kaggle submission file
    # Saving the Passenger ID and predictions to a new CSV file
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'].astype(int), 'Survived': predictions})
    submission.to_csv(f'/Users/scott/Downloads/titanic/{name}_submission.csv', index=False)