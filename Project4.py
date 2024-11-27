
import numpy as np
import pandas as pd
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

file_path = 'twogaussians42.csv' 
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def flip_labels(y, flip_fraction):
    n_flip = int(len(y) * flip_fraction)
    flip_indices = np.random.choice(len(y), n_flip, replace=False)
    y_flipped = y.copy()
    for idx in flip_indices:
        y_flipped[idx] = 1 - y_flipped[idx]  
    return y_flipped

flip_fractions = [0.1, 0.2, 0.3]
for flip_fraction in flip_fractions:
    y_train_flipped = flip_labels(y_train, flip_fraction)
    
    k = 3  
    y_pred = [knn_predict(X_train, y_train_flipped, test_point, k) for test_point in X_test]
    
    accuracy = np.mean(np.array(y_pred) == y_test)
    print(f"Flip fraction: {flip_fraction}, Accuracy: {accuracy:.2f}")

import matplotlib.pyplot as plt

accuracies = []
for flip_fraction in flip_fractions:
    y_train_flipped = flip_labels(y_train, flip_fraction)
    y_pred = [knn_predict(X_train, y_train_flipped, test_point, k) for test_point in X_test]
    accuracy = np.mean(np.array(y_pred) == y_test)
    accuracies.append(accuracy)

plt.figure(figsize=(8, 6))
plt.plot(flip_fractions, accuracies, marker='o', linestyle='--', color='b')
plt.title("Effect of Label Flipping on k-NN Accuracy")
plt.xlabel("Label Flip Fraction")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
