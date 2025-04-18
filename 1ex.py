import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/sesha/Downloads/archive.zip")  # Use raw string if needed, e.g., r"data.csv"

# Print full DataFrame (optional for preview)
print(df)

# Column-wise display
print("\nTo print columns 1 to 4 (second to fifth column) of the DataFrame:")
print("**************")
print(df.iloc[:, 1:5])

print("\nTo print column 5 (sixth column) of the DataFrame:")
print("**************")
print(df.iloc[:, 5:6])

# Check for missing values in the entire DataFrame
print("\nMissing values in the entire DataFrame:")
print(df.isnull().sum())

# Check for missing values in a specific column
print("\nMissing values in 'income' column:")
if 'income' in df.columns:
    print("Boolean check:\n", df['income'].isnull())
    print("Total missing values in 'income':", df['income'].isnull().sum())
else:
    print("Column 'income' does not exist in the dataset.")
