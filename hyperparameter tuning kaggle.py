import pandas as pd  # Importing the Pandas library for data manipulation
import numpy as np  # Importing the NumPy library for numerical operations
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # Importing classes for hyperparameter tuning and k-fold cross-validation
from xgboost import XGBClassifier  # Importing the XGBoost classifier
from sklearn.linear_model import LogisticRegression  # Importing the Logistic Regression classifier
from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest classifier
from sklearn.svm import SVC  # Importing the Support Vector Machines classifier
from sklearn.neighbors import KNeighborsClassifier  # Importing the k-Nearest Neighbors classifier
from sklearn.impute import SimpleImputer  # Importing the SimpleImputer for handling missing values
from sklearn.preprocessing import StandardScaler  # Importing the StandardScaler for feature scaling
from sklearn.pipeline import Pipeline  # Importing the Pipeline class for creating data preprocessing pipelines
from sklearn.compose import ColumnTransformer  # Importing the ColumnTransformer for applying transformers to specific columns
import json  # Importing the JSON library to save and load data in JSON format

# Load the feature-engineered training dataset
try:
    train_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered.csv')  # Reading the CSV file into a DataFrame
    train_no_int_df = pd.read_csv('/Users/scott/Downloads/titanic/train_engineered_no_interaction.csv')  # Reading the CSV file into a DataFrame
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")  # Error message if the file is not found

# Check if 'Survived' column exists
if 'Survived' not in train_df.columns:
    print("Column 'Survived' not found in the dataset.")  # Error message if the 'Survived' column is missing

# Separate features and target variable
X = train_df.drop('Survived', axis=1)  # Dropping the 'Survived' column to obtain features
y = train_df['Survived']  # Target variable 'Survived'

X2 = train_no_int_df.drop('Survived', axis=1)  # Dropping the 'Survived' column to obtain features
y2 = train_no_int_df['Survived']  # Target variable 'Survived'

# Create a pipeline for numerical features
# Imputer fills missing values, and StandardScaler standardizes the features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Replacing missing values with mean
    ('scaler', StandardScaler())  # Scaling features to have mean=0 and variance=1
])

# Assuming all features are numerical
preprocessor = ColumnTransformer([
    ('num', num_pipeline, list(X.columns))  # Applying numerical pipeline to all columns
])

# Dynamically set the columns for the ColumnTransformer based on the DataFrame in use
preprocessor_X = ColumnTransformer([
    ('num', num_pipeline, list(X.columns))  # Applying numerical pipeline to all columns in X
])

preprocessor_X2 = ColumnTransformer([
    ('num', num_pipeline, list(X2.columns))  # Applying numerical pipeline to all columns in X2
])

# Adjust the pipelines to use the respective preprocessor
xgb_pipeline_X = Pipeline([('preprocessor', preprocessor_X), ('classifier', XGBClassifier())])
logreg_pipeline_X = Pipeline([('preprocessor', preprocessor_X), ('classifier', LogisticRegression())])
rf_pipeline_X = Pipeline([('preprocessor', preprocessor_X), ('classifier', RandomForestClassifier())])

xgb_pipeline_X2 = Pipeline([('preprocessor', preprocessor_X2), ('classifier', XGBClassifier())])
rf_pipeline_X2 = Pipeline([('preprocessor', preprocessor_X2), ('classifier', RandomForestClassifier())])




# Create pipelines for different classifiers along with preprocessing steps
# Each pipeline first applies preprocessing and then fits the model
#xgb_pipeline = Pipeline([('preprocessor', preprocessor2), ('classifier', XGBClassifier())])
# XGBoost (eXtreme Gradient Boosting)
# What it Does: XGBoost is an ensemble learning method based on decision trees.
# It's particularly good for large datasets and high-dimensional feature spaces.
# How it Works: It builds trees one at a time, where each new tree corrects errors made by a previously trained tree.
# Each tree is built in a greedy manner to minimize a loss function.
# Spatial/Visual Explanation: Imagine you're trying to classify whether an email is spam or not.
# Each decision tree acts like a filter that sorts emails based on features like "contains the word 'lottery'"
# or "sender is unknown". XGBoost combines the decisions of multiple such filters (trees) to make a final decision
# that is more accurate than any single tree.
# Hyperparameters in Use: Learning rate (how quickly the model adapts), number of trees (n_estimators),
# depth of the trees (max_depth), subsample ratio of the training instance (subsample),
# and feature sampling ratio (colsample_bytree).
#logreg_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression())])
# Logistic Regression
# What it Does: It's a statistical method for modeling the relationship between a binary dependent variable
# and one or more independent variables.
# How it Works: It uses the logistic function to squeeze the output of a linear equation between 0 and 1,
# which can then be interpreted as a probability.
# Spatial/Visual Explanation: Imagine plotting your binary classes (0 and 1) on a 2D plane according to two features.
# The logistic regression model will try to find the "best-fitting" line that separates the two classes.
# For multi-dimensional features, this becomes a hyperplane.
# Hyperparameters in Use: C controls the inverse of regularization strength, penalty specifies the norm used
# in penalization (l1 or l2), solver specifies the algorithm to use for optimization,
# and max_iter specifies the maximum number of iterations for the solver to converge.
#rf_pipeline = Pipeline([('preprocessor', preprocessor2), ('classifier', RandomForestClassifier())])
# Random Forest
# What it Does: It's an ensemble learning method that fits multiple decision trees on various sub-samples of the dataset
# and uses averaging to improve the predictive accuracy.
# How it Works: It combines the predictions from multiple decision trees that are trained on different random subsets
# of the training data.
# Spatial/Visual Explanation: Each decision tree in the forest makes a "vote" for classifying a new object.
# The forest chooses the classification that has the most votes over all the trees in the forest.
# Hyperparameters in Use: Number of trees in the forest (n_estimators), the maximum depth of the tree (max_depth),
# the minimum number of samples required to split an internal node (min_samples_split),
# and the minimum number of samples required to be at a leaf node (min_samples_leaf).
#svc_pipeline = Pipeline([('preprocessor', preprocessor2), ('classifier', SVC())])
# Support Vector Machines (SVC)
# What it Does: It's a classification method that finds the hyperplane that best divides a dataset into classes.
# How it Works: The algorithm selects the hyperplane that has the maximum margin, i.e., the maximum distance
# between data points of both classes.
# Spatial/Visual Explanation: Imagine a 2D plane where you have two classes of points scattered.
# The SVC would find a line that not only separates the two classes but also stays as far away from the closest points
# to it from both classes.
# Hyperparameters in Use: C is the regularization parameter, kernel specifies the kernel type to be used (linear, rbf),
# and gamma is the kernel coefficient for 'rbf'.
#knn_pipeline = Pipeline([('preprocessor', preprocessor2), ('classifier', KNeighborsClassifier())])
# k-Nearest Neighbors (KNN)
# What it Does: KNN is a type of instance-based learning, used for classification and regression tasks.
# How it Works: Given a new observation, KNN identifies k closest instances (neighbors) in the training data
# and makes a prediction based on the majority class among these neighbors.
# Spatial/Visual Explanation: Imagine you scatter plot your dataset in a 2D plane.
# For any new point, you would measure its distance to all other points.
# The k closest points "vote" to decide the class of the new point.
# Hyperparameters in Use: Number of neighbors to use (n_neighbors), weight function used in prediction (weights),
# and the algorithm used to compute the nearest neighbors (algorithm).


# Set up k-fold cross-validation
# StratifiedKFold ensures balanced class distribution in each fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost hyperparameters
xgb_params = {
    # learning_rate: Controls the contribution of each tree to the final prediction.
    # Lower values make the model robust but require more trees (n_estimators) to be effective.
    'classifier__learning_rate': [0.02, 0.03, 0.04, 0.05],
    
    # n_estimators: Number of boosting rounds, or the number of trees added to the model.
    # More trees make the model more expressive, but too many can lead to overfitting.
    'classifier__n_estimators': [20, 30, 40, 50],
    
    # max_depth: The maximum depth of the decision trees being trained.
    # Higher depth will allow more complex models but may overfit.
    'classifier__max_depth': [5, 9, 10, 12, 15],
    
    # subsample: The fraction of the training data randomly sampled for each boosting round.
    # Can prevent overfitting. 1 means use all data, less than 1 means sub-sample.
    'classifier__subsample': [0.5, 0.6, 0.7, 0.8],
    
    # colsample_bytree: Fraction of features to be randomly sampled for each boosting round.
    # Helps prevent overfitting.
    'classifier__colsample_bytree': [0.2, 0.3, 0.4]
}

# Logistic Regression hyperparameters
logreg_params = {
    # C: Inverse of regularization strength. Smaller values increase regularization, reducing overfit.
    'classifier__C': [0.003, 0.005, 0.008],
    
    # penalty: Type of regularization applied ('l1', 'l2', or 'None').
    # Regularization can help in reducing overfit.
    'classifier__penalty': ['None', 'l1', 'l2'],
    
    # solver: Algorithm used for optimization.
    # Various algorithms are suited for different types of data and requirements.
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    
    # max_iter: Maximum number of iterations taken for the solvers to converge.
    'classifier__max_iter': [60, 70, 75, 80, 200]
}

# Random Forest hyperparameters
rf_params = {
    # n_estimators: Number of trees in the forest.
    # Increasing number of trees increases robustness but can be computationally expensive.
    'classifier__n_estimators': [80, 100, 120],
    
    # max_depth: Maximum depth of each decision tree.
    # 'None' means nodes expand until they contain fewer than min_samples_split samples.
    'classifier__max_depth': [None, 15, 20, 25],
    
    # min_samples_split: The minimum number of samples required to split an internal node.
    'classifier__min_samples_split': [7, 10, 15, 20],
    
    # min_samples_leaf: Minimum number of samples required to be at a leaf node.
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Support Vector Classifier hyperparameters
svc_params = {
    # C: Penalty parameter, controls the trade-off between maximizing the margin and minimizing classification error.
    'classifier__C': [0.5, 1, 3, 5],
    
    # kernel: Specifies the type of hyperplane used to separate the data.
    'classifier__kernel': ['linear', 'rbf'],
    
    # gamma: Kernel coefficient. Only used for 'rbf', 'poly', and 'sigmoid'.
    'classifier__gamma': ['auto', 'scale']
}

# k-Nearest Neighbors hyperparameters
knn_params = {
    # n_neighbors: Number of neighbors to consider for each data point.
    'classifier__n_neighbors': [1, 2, 3, 4],
    
    # weights: Function to weight the importance of neighbors ('uniform' or 'distance').
    'classifier__weights': ['uniform', 'distance'],
    
    # algorithm: Algorithm to compute the nearest neighbors.
    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Perform GridSearchCV for each model
# GridSearchCV performs k-fold cross-validation on each hyperparameter combination
# For data with interaction terms (uses X)
pipelines_with_interaction = {'logreg': logreg_pipeline_X, 'xgb': xgb_pipeline_X, 'rf': rf_pipeline_X}
params_with_interaction = {'logreg': logreg_params, 'xgb': xgb_params, 'rf': rf_params}

# For data without interaction terms (uses X2)
pipelines_without_interaction = {'xgb': xgb_pipeline_X2, 'rf': rf_pipeline_X2}
params_without_interaction = {'xgb': xgb_params, 'rf': rf_params}

# GridSearchCV (Grid Search Cross-Validation)
# What it Does: GridSearchCV is a hyperparameter tuning technique that exhaustively searches through a specified hyperparameter grid
# to determine the optimal parameter settings for a given model. It is also used for finding the model that performs best through cross-validation.

# How it Works: For each combination of hyperparameters specified in the grid:
#  1. The algorithm performs k-fold cross-validation, dividing the original training set into 'k' subsets.
#     One of the 'k' subsets is used as the validation set, and the rest are used as the actual training set.
#  2. The model is trained on the training subset and evaluated on the validation subset.
#     This is repeated 'k' times, each time with a different validation subset.
#  3. The cross-validation scores for each hyperparameter combination are averaged to get a single measure of predictive performance.

# Spatial/Visual Explanation: Imagine a 3D space where each axis represents one hyperparameter
# (let's say we have only three hyperparameters for simplicity).
# Each point in this 3D space represents a specific combination of hyperparameters.
# GridSearchCV evaluates the model at each of these points (hyperparameter combinations)
# to find the point where the model performs the best (highest accuracy, lowest loss, etc.).

# Cross-Validation Technique Used: The code uses StratifiedKFold, which is a variation of k-fold cross-validation.
# In StratifiedKFold, the folds are made by preserving the percentage of samples for each class.
# This ensures that each fold is a good representative of the overall dataset, especially useful for imbalanced classes.
# For DataFrame with interaction terms (uses X)
for name, pipeline in {'logreg': logreg_pipeline_X}.items():
    grid = GridSearchCV(pipeline, params_with_interaction[name], cv=kfold, scoring='accuracy', verbose=1)
    grid.fit(X, y)
    print(f"Best {name.upper()} Parameters: {grid.best_params_}")  # Displaying the best hyperparameters
    print(f"Best {name.upper()} Score: {grid.best_score_}")  # Displaying the highest accuracy achieved
    
    # Save best hyperparameters to a JSON file
    with open(f'best_{name}_params.json', 'w') as f:
        json.dump(grid.best_params_, f)  # Saving the best hyperparameters as a JSON file

# For DataFrame without interaction terms (uses X2)
for name, pipeline2 in {'xgb': xgb_pipeline_X2, 'rf': rf_pipeline_X2}.items():
    grid = GridSearchCV(pipeline2, params_without_interaction[name], cv=kfold, scoring='accuracy', verbose=1)
    grid.fit(X2, y2)
    print(f"Best {name.upper()} Parameters: {grid.best_params_}")  # Displaying the best hyperparameters
    print(f"Best {name.upper()} Score: {grid.best_score_}")  # Displaying the highest accuracy achieved
    
    # Save best hyperparameters to a JSON file
    with open(f'best_{name}_params.json', 'w') as f:
        json.dump(grid.best_params_, f)  # Saving the best hyperparameters as a JSON file
