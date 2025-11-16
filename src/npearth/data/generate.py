import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def generate_high_dimensional_spline_data(n_samples=500, random_state=42):
    np.random.seed(random_state)
    X1 = np.linspace(0, 10, n_samples)
    X2 = np.random.uniform(0, 10, n_samples)
    X3 = np.random.normal(5, 1.5, n_samples)

    # Generate a spline effect with interactions
    Y = (
        np.piecewise(X1, [X1 < 5, X1 >= 5], [lambda x: 2 * x, lambda x: -0.5 * x + 10])
        + 3 * np.sin(X2)
        + 0.5 * X3**2
        + 0.3 * X1 * X2
        + np.random.normal(0, 0.5, n_samples)
    )

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Y": Y})
    return df


# Usage
spline_data = generate_high_dimensional_spline_data()
spline_data.to_csv("src/earth/data/high_dimensional_spline_data.csv", index=False)


# Original intermediate sine data function (no noise)
def generate_intermediate_sine_data(n_samples=200, random_state=42):
    np.random.seed(random_state)
    X1 = np.linspace(-np.pi, np.pi, n_samples)
    X2 = np.linspace(-np.pi, np.pi, n_samples)

    # Create a meshgrid for X1 and X2 values
    X1, X2 = np.meshgrid(X1, X2)

    # Generate a target variable with a moderate sinusoidal interaction term
    Y = np.sin(X1) * np.cos(X2) + 0.3 * np.sin(X1 + X2)

    # Return the data in a DataFrame for potential export or further use
    df = pd.DataFrame({"X1": X1.ravel(), "X2": X2.ravel(), "Y": Y.ravel()})
    return X1, X2, Y, df


# Modified function with added noise
def generate_intermediate_sine_data_with_noise(
    n_samples=200, noise_level=0.1, random_state=42
):
    np.random.seed(random_state)
    X1 = np.linspace(-np.pi, np.pi, n_samples)
    X2 = np.linspace(-np.pi, np.pi, n_samples)

    # Create a meshgrid for X1 and X2 values
    X1, X2 = np.meshgrid(X1, X2)

    # Generate the target variable with a moderate sinusoidal interaction term
    Y = np.sin(X1) * np.cos(X2) + 0.3 * np.sin(X1 + X2)

    # Add noise to Y
    noise = noise_level * np.random.randn(*Y.shape)
    Y_noisy = Y + noise

    # Return the data in a DataFrame for potential export or further use
    df = pd.DataFrame({"X1": X1.ravel(), "X2": X2.ravel(), "Y": Y_noisy.ravel()})
    return X1, X2, Y_noisy, df


# Generate the data
X1, X2, Y, sine_data = generate_intermediate_sine_data()
X1, X2, Y_noisy, sine_data_noisy = generate_intermediate_sine_data_with_noise()

sine_data.to_csv("src/earth/data/intermediate_sine_data.csv", index=False)
sine_data_noisy.to_csv("src/earth/data/intermediate_sine_data_noisy.csv", index=False)

# Plot both surfaces side by side
fig = plt.figure(figsize=(16, 8))

# Plot non-noisy data
ax1 = fig.add_subplot(121, projection="3d")
surf1 = ax1.plot_surface(X1, X2, Y, cmap=cm.viridis, edgecolor="k", alpha=0.8)
ax1.set_title("3D Surface Plot - No Noise")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("Y")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot noisy data
ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(X1, X2, Y_noisy, cmap=cm.viridis, edgecolor="k", alpha=0.8)
ax2.set_title("3D Surface Plot - With Noise")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.set_zlabel("Y")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.suptitle("Comparison of Noisy and Non-Noisy Sine Data")
plt.tight_layout()
plt.show()
