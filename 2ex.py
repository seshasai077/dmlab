import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'rollno': [1, 2, 3, 4],
    'branch': ['CSE', 'ECE', 'MECH', 'CSE'],
    'address': ['Addr1', 'Addr2', 'Addr3', 'Addr4']
}

df = pd.DataFrame(data)

# Dealing with Categorical Data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[['branch']])

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['branch']))

# Concatenate encoded columns with original DataFrame
df = pd.concat([df, encoded_df], axis=1)
df = df.drop(['branch'], axis=1)

# Print categorical data
print("Categorical Data (One-Hot Encoded):\n", df[['name', 'branch_CSE', 'branch_ECE', 'branch_MECH']])

# Scaling the Feature
scaler = MinMaxScaler()
df['rollno_scaled'] = scaler.fit_transform(df[['rollno']])

# Print scaled features
print("\nScaled Features:\n", df[['rollno', 'rollno_scaled']])

# Splitting Dataset into Training and Testing Sets
X = df[['rollno_scaled', 'branch_CSE', 'branch_ECE', 'branch_MECH']]
y = df['rollno_scaled']  # Assuming 'rollno_scaled' is the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print train and test datasets
print("\nTraining Data (X_train):\n", X_train)
print("\nTesting Data (X_test):\n", X_test)
print("\nTraining Target (y_train):\n", y_train)
print("\nTesting Target (y_test):\n", y_test)
