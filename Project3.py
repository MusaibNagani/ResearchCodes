# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

file_path = 'Breastcancer.csv'  
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

noise = np.random.normal(0, 0.1, X_train.shape)  
X_train_noisy = X_train + noise

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_noisy)
X_test_norm = scaler.transform(X_test)

best_k = 1
best_score = 0
for k in range(1, 11):  # Testing k from 1 to 10
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_norm, y_train, cv=5) 
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_k = k

print(f"Best k: {best_k}, Cross-validation accuracy: {best_score:.2f}")

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_norm, y_train)
accuracy = final_knn.score(X_test_norm, y_test)
print(f"Test set accuracy with k={best_k}: {accuracy:.2f}")

import matplotlib.pyplot as plt

k_values = list(range(1, 11))
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_norm, y_train, cv=5).mean() for k in k_values]

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='--')
plt.title("Effect of k on Cross-Validation Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid()
plt.show()
