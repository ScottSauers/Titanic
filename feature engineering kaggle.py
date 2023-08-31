import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load datasets
train_df = pd.read_csv('/Users/scott/Downloads/titanic/train.csv')
test_df = pd.read_csv('/Users/scott/Downloads/titanic/test.csv')

# Combine DataFrames
combined_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

# Extract 'Title'
combined_df['Title'] = combined_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Group titles
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
combined_df['Title'] = combined_df['Title'].map(title_mapping)

# Create 'FamilySize'
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1

# Create 'IsAlone'
combined_df['IsAlone'] = 0
combined_df.loc[combined_df['FamilySize'] == 1, 'IsAlone'] = 1

# Create 'FamilyID'
combined_df['FamilyID'] = combined_df['Name'].apply(lambda x: x.split(',')[0]) + "_" + combined_df['FamilySize'].astype(str)

# Count the occurrences of each FamilyID
family_counts = combined_df['FamilyID'].value_counts()

# Set FamilyID to 'Single' for those with a unique FamilyID or FamilySize of 1
combined_df['FamilyID'] = combined_df.apply(lambda row: 'Single' if family_counts[row['FamilyID']] == 1 or row['FamilySize'] == 1 else row['FamilyID'], axis=1)

# One-hot encode specific categorical columns, avoiding the dummy variable trap by using drop_first=True
categorical_cols = ['Sex', 'Embarked', 'Title', 'FamilyID']
one_hot_df = pd.get_dummies(combined_df[categorical_cols], columns=categorical_cols, drop_first=True)
combined_df = pd.concat([combined_df, one_hot_df], axis=1)

# Create interaction term between 'Sex' and 'Pclass'
combined_df['Sex_Pclass'] = combined_df['Sex'] + "_" + combined_df['Pclass'].astype(str)

# One-hot encode the interaction term, avoiding the dummy variable trap
one_hot_interaction = pd.get_dummies(combined_df['Sex_Pclass'], drop_first=True)
combined_df = pd.concat([combined_df, one_hot_interaction], axis=1)

# Drop redundant or unnecessary columns
drop_columns = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'Title', 'FamilyID', 'Sex_Pclass']
combined_df.drop(drop_columns, axis=1, inplace=True)

# Impute missing values with KNN
temp_df = combined_df.drop('Survived', axis=1)
imputer = KNNImputer(n_neighbors=5)
temp_df_imputed = imputer.fit_transform(temp_df)
temp_df_imputed = pd.DataFrame(temp_df_imputed, columns=temp_df.columns)

# Add 'Survived' back
temp_df_imputed['Survived'] = combined_df['Survived']

# Create 'AgeBand' and 'FareBand'
temp_df_imputed['AgeBand'] = pd.cut(temp_df_imputed['Age'], 5, labels=False)
temp_df_imputed['FareBand'] = pd.qcut(temp_df_imputed['Fare'], 5, labels=False)

# Split back into train and test
train_df = temp_df_imputed.loc[temp_df_imputed['Survived'].notna()]
test_df = temp_df_imputed.loc[temp_df_imputed['Survived'].isna()]

# Save to CSV
train_df.to_csv('/Users/scott/Downloads/titanic/train_engineered.csv', index=False)
test_df.to_csv('/Users/scott/Downloads/titanic/test_engineered.csv', index=False)