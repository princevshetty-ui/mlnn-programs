
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

# Generate 2D linearly separable data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Train perceptron
p = Perceptron(max_iter=100, random_state=42)
p.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = p.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision regions
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=60
)

# Plot decision boundary line
w = p.coef_[0]
b = p.intercept_[0]

x_line = np.array([x_min, x_max])
y_line = -(w[0] * x_line + b) / w[1]

plt.plot(
    x_line,
    y_line,
    "k--",
    linewidth=2,
    label="Decision Boundary"
)

# Calculate decision boundary equation
equation = f"y = {(-w[0] / w[1]):.2f}x + {(-b / w[1]):.2f}"

print(f"Weights: {w}, Bias: {b}")
print(
    f"Decision boundary: "
    f"{w[0]:.2f}x1 + {w[1]:.2f}x2 + {b:.2f} = 0"
)
print(f"Line equation: {equation}")

# Add labels and title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()

# Display the plot
plt.show()
