# Task 5: Decision Trees and Random Forests

This repository contains the solution for Task 5 of the AI & ML Internship at Elevate Labs. The project focuses on implementing and evaluating tree-based models for a classification task using the Heart Disease dataset.

## Objective
The goal was to train, visualize, and compare Decision Tree and Random Forest classifiers. Key concepts explored include overfitting, pruning, ensemble learning, cross-validation, and feature importance analysis.

## Additional Features
* **Pruning Demonstration**: The project directly compares a full-depth decision tree with a pruned tree (`max_depth=4`) to demonstrate how limiting tree depth can improve generalization and prevent overfitting.  
* **Robust Evaluation with Cross-Validation**: Instead of relying on a single train-test split, 5-fold cross-validation was used to get a more stable and reliable measure of the Random Forest model's performance.  
* **Feature Importance Visualization**: A bar chart was created to visualize the importance of each feature as determined by the Random Forest, providing clear insights into the key predictors of heart disease in the dataset.  

## Project Visualizations

### 1. Pruned Decision Tree
To ensure interpretability and prevent overfitting, the decision tree was pruned to a maximum depth of 4. The resulting tree is visualized below.

### 2. Feature Importance
The Random Forest model calculates the importance of each feature in making its predictions. As shown below, features like 'ca', 'thalach', and 'cp' are the most significant predictors.

## How to Run
1.  Clone the repository and navigate to the project directory.
2.  Install Graphviz on your system.
3.  Create and activate a virtual environment and install dependencies from `requirements.txt`.
4.  Run the main script:  
    ```bash
    python main.py
    ```  
    The output plots will be saved in the `output/` directory.
