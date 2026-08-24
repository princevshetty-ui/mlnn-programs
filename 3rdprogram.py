import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {cancer.target_names}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Logistic Regression model
# Increased max_iter to allow the optimizer more steps to converge
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Confusion Matrix:\n{cm}")

print(
    f"\nClassification Report:\n"
    f"{classification_report(y_test, y_pred, target_names=cancer.target_names)}"
)

# Plot confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names
)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.show()

# The ConvergenceWarning is a signal that your model might not be as good as it could be. 
# Increasing max_iter is often the first step to resolve it, allowing the model to fully learn from the data. 
# If increasing max_iter still doesn't resolve the warning, it's often recommended to scale your data (e.g., using StandardScaler from sklearn.preprocessing), as feature scaling can significantly help optimization algorithms converge faster.

#The code has executed successfully after increasing max_iter to 5000. This means the Logistic Regression model was able to converge and find a stable solution without hitting the iteration limit. As you can see from the output:
# Accuracy: The model achieved an accuracy of 0.9561, meaning it correctly classified about 95.6% of the test samples.
# Confusion Matrix:
# [[39 4] [ 1 70]]
# This matrix shows that out of 43 malignant cases (first row), the model correctly identified 39 as malignant (True Positives) and incorrectly identified 4 as benign (False Negatives).
# Out of 71 benign cases (second row), the model correctly identified 70 as benign (True Negatives) and incorrectly identified 1 as malignant (False Positive).
# Classification Report: Provides detailed metrics for each class:
# Malignant: High precision (0.97) means that when the model predicts malignant, it's usually correct. A recall of 0.91 means it correctly identifies 91% of actual malignant cases.
# Benign: High precision (0.95) and very high recall (0.99), indicating it's very good at identifying benign cases.
# The heatmap of the confusion matrix visually represents these results, making it easier to see the number of correct and incorrect predictions for each class. Overall, the model performs very well on this dataset.