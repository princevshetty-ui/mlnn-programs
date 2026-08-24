import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return 1 if x >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            errors = 0
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear)

                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0)
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                return True  # Converged
        return False # Did not converge

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

# XOR dataset (linearly inseparable)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR: same->0, different->1

print("XOR Problem - Single Layer Perceptron")
print("=" * 40)
print(f"Input:\n{X_xor}")
print(f"Expected: {y_xor}")

p = Perceptron(lr=0.1, epochs=200)
converged = p.fit(X_xor, y_xor)
predictions = p.predict(X_xor)

print(f"\nWeights: {p.weights}, Bias: {p.bias}")
print(f"Predictions: {predictions}")
print(f"Correct: {np.array_equal(predictions, y_xor)}")
print(f"Converged: {converged}")

print("\nSingle-layer perceptron CANNOT solve XOR because it's linearly inseparable.")
