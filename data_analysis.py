import pandas as pd

df = pd.read_csv("student_data.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nInformation:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())