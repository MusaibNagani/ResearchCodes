# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_predict(X_train, y_train, test_point, k=3):
    distances = []
    for idx, train_point in enumerate(X_train):
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, y_train[idx]))
    
    distances = sorted(distances, key=lambda x: x[0])
    k_neighbors = [label for _, label in distances[:k]]
    
    most_common = Counter(k_neighbors).most_common(1)
    return most_common[0][0]

datasets = ["circles0.3.csv", "twogaussians33.csv"]

for dataset in datasets:
    data = pd.read_csv(dataset)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ks = [1, 3, 5]
    for k in ks:
        y_pred = [knn_predict(X_train, y_train, test_point, k) for test_point in X_test]
        
        accuracy = np.mean(np.array(y_pred) == y_test)
        print(f"Dataset: {dataset.split('/')[-1]}, k={k}, Accuracy: {accuracy:.2f}")

        def plot_decision_boundaries(X, y, knn_predict, k):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            predictions = [knn_predict(X_train, y_train, point, k) for point in grid_points]
            predictions = np.array(predictions).reshape(xx.shape)
            
            plt.contourf(xx, yy, predictions, alpha=0.5, cmap='viridis')
            plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
            plt.title(f"k-NN Decision Boundaries (k={k}) for {dataset.split('/')[-1]}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.show()

        plot_decision_boundaries(X, y, knn_predict, k)
