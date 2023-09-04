import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


def load_and_combine_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)
    return combined_df

def add_title_feature(df):
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Rev",
        "Col": "Officer",
        "Major": "Officer",
        "Mlle": "Miss",
        "Mme": "Mrs",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Jonkheer": "Royalty",
        "Sir": "Royalty",
        "Capt": "Officer",
        "Ms": "Miss"
    }
    df['Title'] = df['Title'].map(title_mapping)
    return df

def add_family_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['LastName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
    df['FamilyID'] = df['LastName'] + "_" + df['FamilySize'].astype(str)
    return df

def add_missing_age_indicator(df):
    df['Age_Missing'] = df['Age'].isna().astype(int)
    return df

def add_new_features(df):
    df['IsChild'] = df['Age'] < 18
    df['IsElderly'] = df['Age'] > 60
    df['IsAlone'] = df['FamilySize'] == 1
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: 'PC' if len(x.split()) > 1 and x.split()[0] == 'PC' else 'None')
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['NameLength'] = df['Name'].apply(len)
    df['Age_Fare'] = df['Age'] * df['Fare']
    df['Title_Age'] = df['Title'].astype(str) + "_" + df['Age'].astype(str)
    df['Sex_Class'] = df['Sex'].astype(str) + "_" + df['Pclass'].astype(str)
    
    # Adding interaction terms
    df['Age_Class'] = df['Age'] * df['Pclass']
    
    # Create Age bins
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100], labels=['Child', 'YoungAdult', 'Adult', 'Elderly'])
    
    # Create binned interaction terms
    df['AgeBin_Class'] = df['Age_Bin'].astype(str) + "_" + df['Pclass'].astype(str)
    df['AgeBin_Sex'] = df['Age_Bin'].astype(str) + "_" + df['Sex'].astype(str)
    
    return df

def one_hot_encoding(df, cols, drop_first=True):
    one_hot_df = pd.get_dummies(df[cols], columns=cols, drop_first=drop_first)
    df = pd.concat([df, one_hot_df], axis=1)
    return df

def remove_single_member_family_columns(df):
    for i in range(1, 6):  # Looping through numbers 1 to 5
        family_columns = [col for col in df.columns if 'FamilyID_' in col and col.endswith(f'_{i}')]
        df.drop(family_columns, axis=1, inplace=True)
    return df


def feature_scaling(df, scaler=None):
    # Store the "PassengerId" column temporarily if it exists in the DataFrame
    passenger_id_column = None
    if "PassengerId" in df.columns:
        passenger_id_column = df["PassengerId"].copy()
        
    # Store the "Survived" column temporarily if it exists in the DataFrame
    survived_column = None
    if "Survived" in df.columns:
        survived_column = df["Survived"].copy()

    # Drop the "PassengerId" and "Survived" columns if they exist
    if "PassengerId" in df.columns:
        df.drop("PassengerId", axis=1, inplace=True)
    if "Survived" in df.columns:
        df.drop("Survived", axis=1, inplace=True)

    # Apply scaling
    if scaler is None:
        scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if len(df[col].unique()) > 2]
    df.loc[:, numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Add back the "PassengerId" and "Survived" columns in a way that avoids fragmentation
    if passenger_id_column is not None:
        df = pd.concat([passenger_id_column, df], axis=1)
    if survived_column is not None:
        df = pd.concat([survived_column, df], axis=1)

    return df, scaler


def convert_to_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'bool':
            df.loc[:, col] = df[col].astype(int)
    return df

def split_and_save(train_df, test_df, train_path, test_path):
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)



def drop_constant_columns(df):
    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) == 2 and (df[col].value_counts().min() == 1):
            df.drop(col, axis=1, inplace=True)
    return df

def drop_non_numeric_text_columns(df):
    dropped_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str)).any():
            dropped_cols.append(col)
            df.drop(col, axis=1, inplace=True)
    print(f"Dropped Columns: {dropped_cols}")
    return df

def knn_imputation(df):
    # Store the 'Survived' column separately
    survived_column = df['Survived'].copy()
    
    # Drop the 'Survived' column from the DataFrame
    df = df.drop('Survived', axis=1)
    
    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=3)
    
    # Perform KNN imputation on the DataFrame without 'Survived'
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Re-add the 'Survived' column to the DataFrame
    df_imputed['Survived'] = survived_column
    
    return df_imputed

def main():
    # Load and preprocess the data
    combined_df = load_and_combine_data('/Users/scott/Downloads/titanic/train.csv', '/Users/scott/Downloads/titanic/test.csv')
    combined_df = add_title_feature(combined_df)
    combined_df = add_family_features(combined_df)
    combined_df = add_new_features(combined_df)
    combined_df = add_missing_age_indicator(combined_df)
    
    # One-hot encoding
    combined_df = one_hot_encoding(combined_df, ['Sex', 'Embarked', 'Title', 'FamilyID', 'Deck', 'TicketPrefix', 'AgeBin_Class', 'AgeBin_Sex'])
    
    # Drop constant columns
    combined_df = drop_constant_columns(combined_df)
    
    # Drop non-numeric text columns
    combined_df = drop_non_numeric_text_columns(combined_df)
    
    # KNN Imputation
    combined_df = knn_imputation(combined_df)
    
    # Convert to numeric (optional, for boolean values)
    combined_df = convert_to_numeric(combined_df)

    combined_df = remove_single_member_family_columns(combined_df)

    # Feature scaling
    combined_df, _ = feature_scaling(combined_df)
    
    # Separate into train and test data
    train_df = combined_df.loc[combined_df['Survived'].notna()].copy()
    test_df = combined_df.loc[combined_df['Survived'].isna()].copy()

    # Save the processed datasets
    split_and_save(train_df, test_df, '/Users/scott/Downloads/titanic/train_engineered.csv', '/Users/scott/Downloads/titanic/test_engineered.csv')

if __name__ == '__main__':
    main()
