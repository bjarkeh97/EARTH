import os, sys

sys.path.append(os.path.abspath(".."))
from time import time as timer
import numpy as np
from matplotlib import pyplot as plt

from npearth.earth import EARTH
from npearth._knotsearcher_cholesky import KnotSearcherCholesky
from npearth._knotsearcher_cholesky_numba import KnotSearcherCholeskyNumba
from npearth._knotsearcher_svd import KnotSearcherSVD

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    def generate_weighted_data(n_samples=200, noise_level=0.1, random_state=42):
        np.random.seed(random_state)
        X = np.linspace(0, 3 * np.pi, n_samples)

        # 1. Define noise levels (sigma) for the four segments
        # The segments are: 0-25%, 25-50%, 50-75%, 75-100% of the samples.
        # Higher sigma means more noise (less reliable data), and thus smaller sample_weight.
        sigma_segments = [
            0.75,  # Segment 1 (Low noise, high weight)
            2,  # Segment 2 (High noise, low weight)
            1.5,  # Segment 3 (Medium noise, medium weight)
            0.25,  # Segment 4 (Very low noise, very high weight)
        ]

        # Calculate the size of each segment (n_samples // 4)
        q = n_samples // 4

        # Initialize arrays for sigma and sample_weights
        sigma = np.zeros(n_samples)
        sample_weights = np.zeros(n_samples)

        # Assign sigma values to each segment
        sigma[:q] = sigma_segments[0]
        sigma[q : 2 * q] = sigma_segments[1]
        sigma[2 * q : 3 * q] = sigma_segments[2]
        sigma[3 * q :] = sigma_segments[3]

        # 2. Calculate sample_weights (W = 1/sigma)
        sample_weights = 1.0 / sigma

        # 3. Define the non-linear function for Y (Sine wave)
        # y_true = 2 * sin(X) + 5
        y_true = 2 * np.sin(X) + 5

        # 4. Add noise to Y according to the segment-specific sigma
        y = y_true + noise_level * np.random.normal(0, sigma)

        return X.reshape(-1, 1), y, sample_weights

    # Step 2: Split Data
    n = 1000
    X, y, sample_weight = generate_weighted_data(n_samples=n, noise_level=0.2)
    X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = (
        train_test_split(X, y, sample_weight, test_size=0.4, random_state=42)
    )

    # Step 3: Fit the EARTH Model
    earth_model = EARTH(M_max=20, knot_searcher=KnotSearcherCholesky, ridge=1e-6)
    t0 = timer()
    earth_model.fit(X_train, y_train, sample_weight=sample_weight_train**2)
    t1 = timer()
    print(f"Cholesky took time {round(t1-t0,5)}")

    earth_model_svd = EARTH(M_max=20, knot_searcher=KnotSearcherSVD, ridge=1e-6)
    t2 = timer()
    earth_model_svd.fit(X_train, y_train, sample_weight=sample_weight_train**2)
    t3 = timer()
    print(f"SVD took time {round(t3-t2,5)}")

    earth_model_np = EARTH(M_max=20, ridge=1e-6)
    t2 = timer()
    earth_model_np.fit(X_train, y_train, sample_weight=sample_weight_train**2)
    t3 = timer()
    print(f"Numba took time {round(t3-t2,5)}")

    mlp = MLPRegressor(hidden_layer_sizes=(150, 50, 30), max_iter=2000, random_state=42)
    mlp.fit(X_train, y_train, sample_weight=sample_weight_train**2)
    y_pred_mlp = mlp.predict(X)
    y_pred_test_mlp = mlp.predict(X_test)

    dt = RandomForestRegressor(max_depth=6)
    dt.fit(X_train, y_train, sample_weight=sample_weight_train**2)
    y_pred_dt = dt.predict(X)
    y_pred_test_dt = dt.predict(X_test)

    # Step 4: Make Predictions
    y_pred = earth_model_np.predict(X)
    y_pred_test = earth_model_np.predict(X_test)
    y_pred_svd = earth_model_svd.predict(X)

    print(
        "sum of abs residuals EARTH",
        np.sqrt(sum(((y_test - y_pred_test) * sample_weight_test) ** 2)).round(4),
    )
    print(
        "sum of abs residuals RF",
        np.sqrt(sum(((y_test - y_pred_test_dt) * sample_weight_test) ** 2)).round(4),
    )
    print(
        "sum of abs residuals MLP",
        np.sqrt(sum(((y_test - y_pred_test_mlp) * sample_weight_test) ** 2)).round(4),
    )

    # Step 5: Visualize Results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.5)
    plt.scatter(X_test, y_test, color="orange", label="Test Data", alpha=0.5)
    plt.plot(
        X,
        y_pred,
        color="red",
        label="EARTH Numba with Cholesky Predictions",
        linewidth=2,
    )
    plt.plot(
        X,
        y_pred_mlp,
        color="green",
        label="MLP Predictions",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        X,
        y_pred_svd,
        color="blue",
        label="EARTH SVD Predictions",
        linewidth=2,
        linestyle="--",
    )
    plt.plot(
        X,
        y_pred_dt,
        color="purple",
        label="Random Forest Predictions",
        linewidth=2,
        linestyle="-.",
    )
    plt.title("EARTH Model Fit on Sine Curve with heteroscedastic Noise")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
