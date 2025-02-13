import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 讀取 Iris 數據集
iris = load_iris()
X = iris.data  # 使用所有特徵
y = X @ np.array([0.5, -0.2, 0.8, -0.1]) + np.random.normal(0, 0.1, X.shape[0])  # 建立人造 target

# 2. 拆分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 訓練隨機森林回歸模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. 預測測試集
y_pred = rf_model.predict(X_test)

# 5. 評估模型表現
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.3f}")  # 越接近1越好
print(f"Mean Squared Error (MSE): {mse:.3f}")

# 6. 視覺化預測結果
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # y = x 參考線
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression Prediction")
plt.grid(True)
plt.show()
