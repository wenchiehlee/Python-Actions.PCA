import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine
### from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Wine dataset
wine_data = load_wine()
### iris_data = load_iris()
df_wine = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
### df_iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df_wine['target'] = wine_data.target_names[wine_data.target]
### df_iris['target'] = iris_data.target_names[iris_data.target]

# Visualize pairs of features
sns.pairplot(df_wine, hue='target', vars=['alcohol', 'malic_acid', 'color_intensity', 'proline', 'ash', 'magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'hue', 'od280/od315_of_diluted_wines', 'total_phenols'])
### sns.pairplot(df_iris, hue='target', vars=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

plt.show()