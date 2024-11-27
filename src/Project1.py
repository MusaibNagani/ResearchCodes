import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter


file_path = 'data/circles0.3.csv'
data = pd.read_csv(file_path)

# Visualize dataset
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:, 2], cmap='viridis', edgecolor='k')
plt.title("Circles Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_predict(X_train, y_train, test_point, k=3):
    distances = [(euclidean_distance(test_point, train_point), label) for train_point, label in zip(X_train, y_train)]
    k_neighbors = [label for _, label in sorted(distances)[:k]]
    return Counter(k_neighbors).most_common(1)[0][0]

k = 3
y_pred = [knn_predict(X_train, y_train, test_point, k) for test_point in X_test]
accuracy = np.mean(np.array(y_pred) == y_test)
print(f"Accuracy of k-NN with k={k}: {accuracy:.2f}")

def plot_decision_boundaries(X, y, knn_predict, k):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    predictions = [knn_predict(X_train, y_train, np.array([x, y]), k) for x, y in zip(xx.ravel(), yy.ravel())]
    plt.contourf(xx, yy, np.array(predictions).reshape(xx.shape), alpha=0.5, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f"k-NN Decision Boundaries (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundaries(X, y, knn_predict, k)
