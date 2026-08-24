import numpy as np
import matplotlib.pyplot as plt


# Step Activation Function
def step(x):
    return np.where(x >= 0, 1, 0)


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Tanh Activation Function
def tanh(x):
    return np.tanh(x)


# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)


# Generate input values
x = np.linspace(-10, 10, 1000)


# Compute outputs
y_step = step(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)


# Store functions and their outputs
functions = [
    ("Step", y_step),
    ("Sigmoid", y_sigmoid),
    ("Tanh", y_tanh),
    ("ReLU", y_relu)
]


# Create plots
plt.figure(figsize=(14, 10))


for i, (name, y_vals) in enumerate(functions, 1):

    plt.subplot(2, 2, i)

    plt.plot(
        x,
        y_vals,
        "b-",
        linewidth=2
    )

    # Reference lines
    plt.axhline(
        y=0,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    plt.axvline(
        x=0,
        color="gray",
        linestyle="--",
        alpha=0.5
    )

    # Labels and title
    plt.title(f"{name} Activation")
    plt.xlabel("Input")
    plt.ylabel("Output")

    plt.grid(
        True,
        alpha=0.3
    )

    # Set Y-axis limits
    if name != "ReLU":
        plt.ylim(-1.5, 1.5)
    else:
        plt.ylim(-1, 5)

    # Print properties
    print(f"\n{name}:")
    print(
        f"Output range: "
        f"[{y_vals.min():.2f}, {y_vals.max():.2f}]"
    )

    print(
        f"Differentiable: "
        f"{'No' if name == 'Step' else 'Yes'}"
    )

    if name != "Step":

        if name == "Sigmoid":
            derivative = "σ(x) * (1 - σ(x))"

        elif name == "Tanh":
            derivative = "1 - tanh²(x)"

        else:
            derivative = "0 if x < 0, 1 if x > 0"

        print(f"Derivative: {derivative}")


# Adjust layout
plt.tight_layout()

# Display plots
plt.show()