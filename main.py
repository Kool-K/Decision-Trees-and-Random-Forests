import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz
import os

# Create an output directory for visualizations
os.makedirs("output", exist_ok=True)

# --- 1. Load and Prepare the Data ---
df = pd.read_csv('data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Train a Full-Depth Decision Tree (Prone to Overfitting) ---
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
preds_full = dt_full.predict(X_test)
print("--- Full-Depth Decision Tree Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, preds_full):.4f}\n")

# --- 3. Train a Pruned Decision Tree (To Control Overfitting) ---
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)
preds_pruned = dt_pruned.predict(X_test)
print("--- Pruned Decision Tree (max_depth=4) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, preds_pruned):.4f}\n")

# --- 4. Visualize the Pruned Decision Tree ---
dot_data = export_graphviz(dt_pruned, out_file=None,
                           feature_names=X.columns,
                           class_names=['No Disease', 'Disease'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("output/decision_tree_pruned", format='png', cleanup=True)
print("Pruned decision tree visualization saved to 'output/decision_tree_pruned.png'")

# --- 5. Train a Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
print("\n--- Random Forest Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, preds_rf):.4f}")
print(classification_report(y_test, preds_rf))

# --- 6. (Additional Feature) Cross-Validation ---
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\n--- Random Forest Cross-Validation Scores ---")
print(f"Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.2f})\n")

# --- 7. (Additional Feature) Feature Importance ---
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('output/feature_importance.png')
print("Feature importance plot saved to 'output/feature_importance.png'")

