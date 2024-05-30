import pandas as pd

# Load the Iris dataset
iris_df = pd.read_csv('iris_dataset.csv')

# Print the first few rows to inspect the data
print(iris_df.head())

# Check for non-numeric data
print(iris_df.info())

# Exclude the 'Species' column for correlation calculation
numeric_df = iris_df.drop(columns=['Species'])

# Ensure all remaining columns are numeric
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

# Drop any rows with NaN values
numeric_df = numeric_df.dropna()

# Print the cleaned numeric data
print(numeric_df.head())

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Save the correlation matrix to a CSV file 
correlation_matrix.to_csv('correlation_matrix.csv')

print("Correlation matrix saved to 'correlation_matrix.csv'")

