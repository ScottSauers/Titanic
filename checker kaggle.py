import pandas as pd

# Load the preprocessed data
train_path = '/Users/scott/Downloads/titanic/train_engineered.csv'
train_df = pd.read_csv(train_path)

# Calculate Pearson correlation matrix
correlation_matrix = train_df.corr()

# Create a list of unique pairs of feature names and their correlations
feature_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feature_i = correlation_matrix.columns[i]
        feature_j = correlation_matrix.columns[j]
        correlation = correlation_matrix.loc[feature_i, feature_j]
        feature_pairs.append((feature_i, feature_j, correlation))

# Sort feature pairs by correlation in descending order
feature_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# Print the top 30 sorted feature pairs with correlations
top_n = 30
for feature_pair in feature_pairs[:top_n]:
    feature_i, feature_j, correlation = feature_pair
    print(f"Pair: {feature_i} and {feature_j}, Correlation: {correlation:.4f}")
