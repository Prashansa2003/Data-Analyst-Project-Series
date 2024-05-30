# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Basic data exploration
print("Head of the dataset:")
print(iris_df.head())

print("\nStatistical summary of the dataset:")
print(iris_df.describe())

# Visualizing the distribution of features
plt.figure(figsize=(10, 6))
for i, feature in enumerate(iris['feature_names']):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=iris_df, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Visualizing pairwise relationships between features
sns.pairplot(iris_df, hue='species')
plt.show()
