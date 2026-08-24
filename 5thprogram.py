import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    classification_report
)

# Load Bank Marketing dataset
# Download and extract bank-additional-full.csv manually
df = pd.read_csv(
    "bank-additional-full.csv",
    sep=";"
)

# Convert target: 'yes'/'no' -> 1/0
df["y"] = df["y"].map({
    "yes": 1,
    "no": 0
})

# Select numeric columns only for simplicity
numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

# All numeric columns except target
X = df[numeric_cols[:-1]]
y = df["y"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Create Decision Tree classifier
dt = DecisionTreeClassifier(
    max_depth=5,
    criterion="entropy",
    random_state=42
)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feat_imp = sorted(
    zip(X.columns, dt.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 5 Features:")

for name, imp in feat_imp[:5]:
    print(f"  {name}: {imp:.4f}")

# Visualize the Decision Tree
plt.figure(figsize=(20, 12))

plot_tree(
    dt,
    filled=True,
    feature_names=X.columns.tolist(),
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree (Bank Marketing)")
plt.show()