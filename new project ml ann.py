import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv(r"C:\Users\Raghav Prabanth\OneDrive\Desktop\PROJECTS RELATED FILES\python learner files\Data_for_UCI_named.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for any missing values
print(df.isnull().sum())

# Features and target variable
X = df.drop(['stabf'], axis=1)  # 'stabf' is the target column
y = df['stabf'].map({'stable': 0, 'unstable': 1})  # Mapping 'stable' to 0 and 'unstable' to 1

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train_scaled)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Convert the prediction output to binary (0 for normal, 1 for anomaly)
y_pred = np.where(y_pred == -1, 1, 0)

# Evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a DataFrame for plotting
df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
df_test['True_Label'] = y_test.reset_index(drop=True)
df_test['Predicted_Anomaly'] = y_pred

# Plot the results using the correct column names
plt.figure(figsize=(10, 6))
plt.scatter(df_test['tau1'], df_test['p1'], c=df_test['Predicted_Anomaly'], cmap='coolwarm', label='Predicted Anomalies')
plt.title('Anomaly Detection in Electrical Grid Stability')
plt.xlabel('tau1 (Feature 1)')
plt.ylabel('p1 (Feature 2)')
plt.colorbar(label='Anomaly Score')
plt.show()

''''''

# Boxplot to detect anomalies in feature 'tau1'
plt.figure(figsize=(10, 6))
plt.boxplot(df_test['tau1'], vert=False)
plt.title('Boxplot for Feature tau1 (Anomalies)')
plt.show()


''''''


# Add the predicted anomalies to the dataframe
df_test['Anomaly'] = y_pred

# Pair plot to visualize relationships between features and anomalies
sns.pairplot(df_test, hue='Anomaly', diag_kind='kde')
plt.show()

''''''


''''''
# Plot histogram with anomalies highlighted
plt.figure(figsize=(10, 6))
plt.hist(df_test[df_test['Anomaly'] == 0]['tau1'], bins=30, alpha=0.6, label='Normal')
plt.hist(df_test[df_test['Anomaly'] == 1]['tau1'], bins=30, alpha=0.6, label='Anomaly', color='r')
plt.title('Histogram of tau1 (Normal vs Anomalies)')
plt.legend()
plt.show()


''''''
# Time series plot (Assuming we have a time index)
df_test['Time'] = range(len(df_test))  # Simulate a time column

plt.figure(figsize=(12, 6))
plt.plot(df_test['Time'], df_test['tau1'], label='Feature tau1', alpha=0.6)
plt.scatter(df_test['Time'][df_test['Anomaly'] == 1], df_test['tau1'][df_test['Anomaly'] == 1], color='r', label='Anomalies')
plt.title('Time Series Plot with Anomalies')
plt.xlabel('Time')
plt.ylabel('tau1')
plt.legend()
plt.show()
