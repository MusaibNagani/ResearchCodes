# k-Nearest Neighbors (k-NN) Implementation and Experiments

## Overview
This project explores the fundamentals and applications of the k-Nearest Neighbors (k-NN) algorithm through a series of hands-on implementations, visualizations, and robustness experiments using synthetic and real-world datasets.

---

## Objectives
1. **Project 1: Understanding k-NN Fundamentals**
   - Learn the basics of k-NN by implementing it from scratch.
   - Visualize decision boundaries for a simple dataset.

2. **Project 2: Experimenting with Synthetic Datasets**
   - Investigate k-NN's behavior with various dataset geometries and configurations.

3. **Project 3: Improving Robustness**
   - Explore the effects of noise and preprocessing techniques on k-NN performance.

4. **Project 4: Evaluating Adversarial Attacks**
   - Test the resilience of k-NN to label flipping and adversarial attacks.

---

## Projects

### **Project 1: Understanding k-NN Fundamentals**
- **Objective**: Understand the fundamentals of k-NN by building it from scratch.
- **Steps**:
  1. Load the `circles0.3.csv` dataset and split it into training and testing sets.
  2. Implement k-NN:
     - Calculate distances between a test point and all training points.
     - Identify the k-nearest neighbors.
     - Predict the label using majority voting.
  3. Visualize decision boundaries.
  4. Evaluate classification accuracy for different values of k.
- **Output**:
  - Decision boundary plots.
  - Classification accuracy for varying values of k.

---

### **Project 2: Experimenting with Synthetic Datasets**
- **Objective**: Explore k-NN's behavior with different dataset geometries.
- **Datasets**:
  - `moons1.csv`: Overlapping classes.
  - `twogaussians33.csv` and `twogaussians42.csv`: Gaussian distributions.
- **Steps**:
  1. Train a k-NN classifier (manual implementation or scikit-learn).
  2. Experiment with different values of k (e.g., 1, 5, 10).
  3. Test various distance metrics (e.g., Euclidean, Manhattan, Cosine).
  4. Visualize decision boundaries.
- **Output**:
  - Decision boundary plots for each dataset and configuration.
  - Accuracy scores for all combinations of k and distance metrics.

---

### **Project 3: Improving Robustness**
- **Objective**: Test and enhance k-NN's robustness against noise and imbalanced data.
- **Datasets**:
  - `halfkernel.csv` (overlapping data).
  - `Breastcancer.csv` (real-world data).
- **Steps**:
  1. Add Gaussian noise to datasets and observe the effect on k-NN performance.
  2. Normalize or standardize the data.
  3. Optimize hyperparameters (k, distance metrics) using cross-validation.
- **Output**:
  - Accuracy comparison before and after noise reduction.
  - Hyperparameter tuning results.

---

### **Project 4: Evaluating Adversarial Attacks**
- **Objective**: Test k-NN's resilience to adversarial attacks, such as label flipping.
- **Datasets**:
  - `twogaussians42.csv`
  - `spiral1.csv` (challenging decision boundaries).
- **Steps**:
  1. Implement a label-flipping attack by randomly flipping a percentage of training labels.
  2. Train k-NN with and without the poisoned dataset.
  3. Measure the accuracy drop due to the attack.
- **Output**:
  - Accuracy vs. percentage of flipped labels.
  - Visualized poisoned decision boundaries.

---

## Additional Insights and Learning Outcomes
- **Project 1**: Develop a deep understanding of k-NN and implement it from scratch.
- **Project 2**: Visualize and analyze how k-NN adapts to various dataset geometries.
- **Project 3**: Learn preprocessing techniques and optimize k-NN for better robustness.
- **Project 4**: Explore adversarial learning and understand k-NN vulnerabilities.

---

## How to Run
1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
