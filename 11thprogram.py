
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build MLP
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),       # Hidden layer
    Dense(10, activation='relu'),       # Hidden layer
    Dense(3, activation='softmax')      # Output layer: 3 classes
])

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train (backpropagation happens here)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)

# Evaluate
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Plot loss and accuracy
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Loss (Backpropagation)')

# Plot accuracy
plt.subplot(122)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Accuracy')

plt.tight_layout()
plt.show()

# Display backpropagation steps
print("\nBackpropagation steps per epoch:")
print("1. Forward pass: compute predictions")
print("2. Compute loss: categorical cross-entropy")
print("3. Backward pass: compute gradients via chain rule")
print("4. Update weights: w = w - lr * gradient")
