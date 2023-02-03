import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# Load the data into a pandas DataFrame
data = pd.read_csv("C:/Users/HI/Downloads/sensor.csv/sensor.csv")

# Drop any columns with missing values
data.dropna(axis=1, inplace=True)
imputer = SimpleImputer(strategy='mean')
data = data.select_dtypes(include=[np.number])
data = data.drop(["id"], axis=1)  # remove id column
data = data[data.status != "NORMAL"]  # remove "NORMAL" status
data = data.drop(["status"], axis=1)  # remove status column
# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(data.values)

# Train the one-class SVM classifier
model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model.fit(X)

# Predict the anomalies
y_pred = model.predict(X)

# Keep only the rows from the data DataFrame that correspond to the samples used for training the one-class SVM classifier
if X.shape[0] != data.shape[0]:
    data = data.iloc[model.support_, :]

# Add the predictions to the original data
data['anomaly'] = y_pred

# Save the results to a new file
data.to_csv("results.csv", index=False)


