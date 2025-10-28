# 3a. Load the dataset from a CSV file and display the first 10 rows along with basic statistics.
import pandas as  pd

df = pd.read_csv('data.csv')
print(" Dataset loaded successfully!")
print("\n Top 10 rows:")
print(df.head(10))
print("\n Basic statistics:")
print(df.describe(include='all'))