
# TASK 1: LOAD AND EXPLORE DATASET

from sklearn.datasets  import load_iris # pyright: ignore[reportMissingModuleSource]

iris = load_iris()

import pandas as pd

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target


print(df.head())

# Display the first 5 rows of the DataFrame
print("First 5 rows:")
print(df.head())

# Display basic information about the DataFrame (column types, non-null counts)
print("\nDataFrame info:")
print(df.info())


# Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())


# TASK 2: BASIC DATA ANALYSIS

#  Display summary statistics of numerical columns
print("\nSummary statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped_means = df.groupby('species').mean()
print("\nMean values by species:\n", grouped_means)


# TASK 3: DATA VISUALISATION


import matplotlib.pyplot as plt

import numpy as np

import calendar


months = list(calendar.month_name)[1:]  # ['January', ..., 'December']

#Creating a line Chart

#Example sales data for 12 months (random values for demonstration)
sales = np.array([1300, 1600, 1650, 1600, 1800, 1900, 2200, 2100, 2150, 2300, 2450, 2600])

plt.plot(months, sales, marker='o', linestyle='-', color='g', label='Monthly Sales')
plt.title('Monthly Sales Trend for 2025')
plt.xlabel('Month')
plt.ylabel('Sales (units)')
plt.grid(True)
plt.legend(title='Functions', loc='upper left')
plt.show()


#Creating a bar chart

species = ['rose', 'sunflower', 'carnation']
avg_petal_length = [1.46, 4.26, 5.55]  # example averages for each species

plt.bar(species, avg_petal_length, color=['red', 'yellow', 'pink'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()


#Creating a Histogram

 #Example numerical data
data = np.random.randn(900)  # 900 random numbers from a normal distribution

plt.hist(data, bins=30, color='skyblue', edgecolor='green')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Numerical Data')

# Show the plot
plt.show()


#Scattered plot


# Example data: sepal length and petal length for 5 samples
sepal_length = np.array([5.1, 4.9, 4.7, 4.6, 5.0])
petal_length = np.array([1.4, 1.4, 1.3, 1.5, 1.4])

plt.scatter(sepal_length, petal_length, color='brown', label='Samples')

# Adding titles and labels
plt.title('Scatter Plot of Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# Adding a legend
plt.legend()

# Show plot
plt.show()
