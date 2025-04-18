import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv('C:/Users/sesha/Downloads/archive.zip')

# Display first few rows to understand structure (optional)
print(df.head(3))

# Handle categorical variables using one-hot encoding
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
x = df_encoded.drop(['price'], axis=1)
y = df_encoded['price']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Predict price for new data
# Create new data matching the encoded features (manually mapped example)
# Assume: area=3000, bedrooms=3, bathrooms=2, stories=2, parking=1,
# mainroad=yes, guestroom=no, basement=yes, hotwaterheating=no,
# airconditioning=yes, prefarea=yes, furnishingstatus
