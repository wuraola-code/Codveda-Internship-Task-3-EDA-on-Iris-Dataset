# Exploratory Data Analysis (EDA) on Iris Dataset

Introduction

The Iris dataset is a classic dataset in machine learning and statistics, widely used for pattern recognition and classification problems. It consists of 150 samples of iris flowers from three species (setosa, versicolor, and virginica). Each sample contains four numerical features:

sepal length (cm)

sepal width (cm)

petal length (cm)

petal width (cm)


The goal of this analysis is to perform exploratory data analysis (EDA) to better understand the distribution, relationships, and patterns within the dataset.

Step 1: Load the Dataset

I loaded the Iris dataset directly using Seaborn’s built-in load_dataset() function, which provides a clean DataFrame containing all features along with the target column (species).

import seaborn as sns
import pandas as pd

# Load dataset
df = sns.load_dataset("iris")
df.head()

Step 2: Dataset Overview

Shape of the dataset (rows and columns).
Column names and data types.
Checking for missing values.

print(df.shape)
print(df.info())
print(df.isnull().sum())

Step 3: Summary Statistics

I generated descriptive statistics to understand the central tendency and spread of the features.

df.describe()
This gives insights such as mean, median, standard deviation, min, and max for each numerical feature.

Step 4: Univariate Analysis

I visualized the distribution of each feature to identify patterns, skewness, and spread.

Histograms for each numerical column.
Boxplots to detect potential outliers.

import matplotlib.pyplot as plt
df.hist(figsize=(10, 8))
plt.show()
sns.boxplot(data=df)

Step 5: Bivariate Analysis

I explored the relationships between features and how they differ across species.

Pairplot for feature relationships.
Scatter plots grouped by species.
Correlation heatmap to check for multicollinearity.


sns.pairplot(df, hue="species")
plt.show()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

Step 6: Grouped Analysis by Species

I compared mean values of features across the three species to detect distinguishing traits.

df.groupby("species").mean()


Conclusion

From the EDA, I can draw the following key insights:

Petal length and petal width are strong distinguishing features for differentiating species.

Setosa species is clearly separated from the other two in most visualizations.

Versicolor and Virginica show some overlap but still display measurable differences in petal features.

No missing values were found, and the dataset is clean and ready for further modeling.


This exploratory analysis provides a strong foundation for building classification models on the Iris dataset.

