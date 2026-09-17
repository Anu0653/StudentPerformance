import pandas as pd

# Load dataset
df = pd.read_csv("student_data.csv")

# Check data types
print("Data types before cleaning:")
print(df.dtypes)

# Remove duplicate rows
df = df.drop_duplicates()

# Convert categorical columns to category type
categorical_columns = df.select_dtypes(include=["str"]).columns

for column in categorical_columns:
    df[column] = df[column].astype("category")

print("\nData types after cleaning:")
print(df.dtypes)

print("\nFinal shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())