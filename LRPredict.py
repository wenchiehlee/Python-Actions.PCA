import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # Use all features
y_original = iris.target  # Original categorical target labels

# 2. Perform PCA to determine feature importance
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
pca_weights = pca.components_[0]  # Use the first principal component as weights

# Print PCA weights
print("PCA Weights (First Principal Component):", pca_weights)

# 3. Scale the principal component to match the categorical target distribution
scaler = MinMaxScaler(feature_range=(0, 2))  # Scale between 0, 1, 2
y = scaler.fit_transform(X_pca[:, 0].reshape(-1, 1)).flatten()

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, y_orig_train, y_orig_test = train_test_split(
    X, y, y_original, test_size=0.2, random_state=42)

# 5. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Display regression coefficients and intercept
print("Regression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 7. Make predictions on the test set
y_pred = model.predict(X_test)

# 8. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# 9. Convert Predicted Numeric Values to Categorical Labels
thresholds = np.percentile(y_pred, [33, 67])  # Get percentile thresholds for splitting into 3 groups
y_pred_class = np.digitize(y_pred, bins=thresholds)  # Convert to classes (0,1,2)

# 10. Evaluate classification accuracy
accuracy = accuracy_score(y_orig_test, y_pred_class)
print(f"Classification Accuracy: {accuracy:.3f}")
print("Classification Report:\n", classification_report(y_orig_test, y_pred_class))

# 11. Display prediction results
df_result = pd.DataFrame({'Actual Numeric': y_test, 'Predicted': y_pred, 'Predicted Class': y_pred_class, 'Original Target': y_orig_test})
print(df_result.head(10))  # Display the first 10 predictions

# 12. Visualize predictions vs. actual values with color based on Original Target
plt.figure(figsize=(8,6))
colors = ['red', 'green', 'blue']  # Assign different colors for different categories
for i, target_class in enumerate(np.unique(y_orig_test)):
    mask_test = y_orig_test == target_class
    plt.scatter(y_test[mask_test], y_pred[mask_test], color=colors[i], alpha=0.5, label=f'Test Class {target_class}')

# 13. Visualize training data as 'x' markers
for i, target_class in enumerate(np.unique(y_orig_train)):
    mask_train = y_orig_train == target_class
    plt.scatter(y_train[mask_train], model.predict(X_train)[mask_train], color=colors[i], marker='x', alpha=0.5, label=f'Train Class {target_class}')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Ideal Fit')  # y = x reference line
plt.xlabel("Actual Numeric Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression Prediction (Colored by Original Target)")
plt.legend()
plt.grid(True)
plt.show()

# 14. Visualize how well Actual Numeric aligns with Original Target
plt.figure(figsize=(8,6))
plt.boxplot([y[y_original == i] for i in range(3)], labels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel("Original Target Class")
plt.ylabel("Numeric Target Value")
plt.title("Distribution of Numeric Target by Class")
plt.grid(True)
plt.show()

# 15. Check correlation between numeric target and original target
correlation = np.corrcoef(y_original, y)[0, 1]
print(f"Correlation between original categorical target and generated numeric target: {correlation:.3f}")
