import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('data/data1.xls')

data = data.drop(columns=['ID'], errors='ignore') 

numerical_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10','X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20','X21', 'X22', 'X23']

data = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

data = data.dropna()

X = data.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

distance_metrics = ['euclidean', 'manhattan', 'cosine']
k_values = [5, 10, 15]

for distance_metric in distance_metrics:
    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric)
        nbrs.fit(X_scaled)

        distances, indices = nbrs.kneighbors(X_scaled)
        anomaly_scores = distances.mean(axis=1)

        threshold_percentile = 95
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        outliers = np.where(anomaly_scores > threshold)[0]

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', label='Normal Data')
        plt.scatter(X_pca[outliers, 0], X_pca[outliers, 1], c='red', label='Anomalies')
        plt.title(f'Outlier Detection with {distance_metric} Metric (k={k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.hist(anomaly_scores, bins=50, alpha=0.7, color='blue', label='Anomaly Scores')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold_percentile}th percentile)')
        plt.title(f'Distribution of Anomaly Scores ({distance_metric}, k={k})')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        print(f"\nResults for {distance_metric} Metric with k={k}")
        print(f"Number of Outliers: {len(outliers)}")
        print(f"Indices of Outliers: {outliers}")
        print(f"Anomaly Scores of Outliers: {anomaly_scores[outliers]}")

        results = pd.DataFrame({'Index': range(len(anomaly_scores)), 'Anomaly_Score': anomaly_scores})
        results['Is_Outlier'] = 0
        results.loc[outliers, 'Is_Outlier'] = 1
        results.to_csv(f'results_{distance_metric}_k{k}.csv', index=False)
        print(f"Results saved to results_{distance_metric}_k{k}.csv")
