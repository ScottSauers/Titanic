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

# Function to remove 'classifier__' prefix from hyperparameters
def remove_prefix(params):
    return {key.replace('classifier__', ''): value for key, value in params.items()}

# Load both interaction and non-interaction training and test datasets
train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')
test_df = pd.read_csv('/Users/scott/Downloads/titanic/test_engineered.csv')
train_df_X2 = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered_no_interaction.csv')
test_df_X2 = pd.read_csv('/Users/scott/Downloads/titanic/test_engineered_no_interaction.csv')

# Separate features and target variable
X_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
X_test = test_df

X2_train = train_df_X2.drop('Survived', axis=1)
y2_train = train_df_X2['Survived']
X2_test = test_df_X2

# Create pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create preprocessors for both interaction and non-interaction datasets
preprocessor = ColumnTransformer([
    ('num', num_pipeline, list(X_train.columns))
])

preprocessor_X2 = ColumnTransformer([
    ('num', num_pipeline, list(X2_train.columns))
])

# List of model names for interaction and non-interaction datasets
model_names_with_interaction = ['logreg']
model_names_without_interaction = ['xgb', 'rf', 'svc', 'knn']

# Loop through each model that uses interaction terms
for name in model_names_with_interaction:
    try:
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = remove_prefix(json.load(f))
    except FileNotFoundError:
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue
    model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(**best_params))])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).astype(int)
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'].astype(int), 'Survived': predictions})
    submission.to_csv(f'/Users/scott/Downloads/titanic/{name}_submission.csv', index=False)

# Loop through each model that does not use interaction terms
for name in model_names_without_interaction:
    try:
        with open(f'best_{name}_params.json', 'r') as f:
            best_params = remove_prefix(json.load(f))
    except FileNotFoundError:
        print(f"JSON file for {name.upper()} not found. Skipping this model.")
        continue
    if name == 'xgb':
        model = Pipeline([('preprocessor', preprocessor_X2), ('classifier', XGBClassifier(**best_params))])
    elif name == 'rf':
        model = Pipeline([('preprocessor', preprocessor_X2), ('classifier', RandomForestClassifier(**best_params))])
    elif name == 'svc':
        model = Pipeline([('preprocessor', preprocessor_X2), ('classifier', SVC(**best_params))])
    elif name == 'knn':
        model = Pipeline([('preprocessor', preprocessor_X2), ('classifier', KNeighborsClassifier(**best_params))])
    model.fit(X2_train, y2_train)
    predictions = model.predict(X2_test).astype(int)
    submission = pd.DataFrame({'PassengerId': test_df_X2['PassengerId'].astype(int), 'Survived': predictions})
    submission.to_csv(f'/Users/scott/Downloads/titanic/{name}_submission.csv', index=False)
