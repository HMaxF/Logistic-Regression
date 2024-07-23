import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Create the training dataset
train_data = {
    'gender': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'age': [20, 20, 35, 35, 50, 50, 20, 20, 35, 35, 50, 50],
    'city': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
    'income': [5000, 7000, 7000, 9000, 3000, 5000, 3000, 5000, 4000, 5000, 3000, 5000],
    'want_to_get_health_insurance': [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
}

# Convert to DataFrame
df_train = pd.DataFrame(train_data)

# Separate the input features (X) and the target variable (y)
X_train = df_train.drop('want_to_get_health_insurance', axis=1)
y_train = df_train['want_to_get_health_insurance'].values.ravel()

# Create and train the logistic regression model with max_iter parameter
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# New data to be predicted
new_data = {
    'gender': [1, 1, 1, 0, 0, 0],
    'age': [40, 20, 35, 20, 20, 35],
    'city': [1, 2, 3, 1, 2, 3],
    'income': [8000, 3000, 6000, 6000, 4000, 8000]
}

# Convert to DataFrame
df_new = pd.DataFrame(new_data)

# Predict probabilities using the trained model
predictions_prob = model.predict_proba(df_new)

# Set a custom threshold
threshold = 0.7 # default 0.5
predictions = (predictions_prob[:, 1] >= threshold).astype(int)

# Combine predictions with new data for output
df_new['want_to_get_health_insurance'] = predictions
df_new['probability'] = predictions_prob[:, 1]

# Display the results
print(df_new)

# Retrieve the learned weights and bias
weights = model.coef_[0]
bias = model.intercept_[0]

# Calculate z values
def calculate_z(row, weights, bias):
    return np.dot(weights, row) + bias

z_values = []
for index, row in df_new.drop(['want_to_get_health_insurance', 'probability'], axis=1).iterrows():
    z = calculate_z(row.values, weights, bias)
    z_values.append(z)
    print(f"Row {index+1} - x: {row.values}, z: {z}, Probability: {1 / (1 + np.exp(-z))}")

# Display z values
df_new['z_value'] = z_values
print(df_new)
