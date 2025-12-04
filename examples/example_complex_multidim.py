import logging
from time import time as timer
import numpy as np
from matplotlib import pyplot as plt

from npearth.earth import EARTH

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Step 1: Generate Sine Curve Data
    def generate_data(
        n_samples=200, noisy_features: int = 2, noise_level=0.1, random_state=42
    ):
        np.random.seed(random_state)
        X1 = np.linspace(0, 10, n_samples)
        X2 = np.random.uniform(0, 10, n_samples)
        X3 = np.random.normal(5, 1.5, n_samples)

        # Generate a spline effect with interactions
        Y = (
            1
            + np.piecewise(
                X1, [X1 < 5, X1 >= 5], [lambda x: 2 * x, lambda x: -0.5 * x + 10]
            )
            + 3 * np.sin(X2)
            + 2 * np.log(abs(X1) + 1) / (X3 + X2)
            + 0.5 * X3**2
            + 0.3 * X1 * X2
            + np.random.normal(0, noise_level, n_samples)
        )

        X = np.column_stack([X1, X2, X3])
        if noisy_features > 0 and isinstance(noisy_features, int):
            noisy_features = np.random.normal(0, 10, (n_samples, noisy_features))
            X = np.hstack([X, noisy_features])

        return X, Y  # Reshape for sklearn compatibility

    # Step 2: Split Data
    n = 2000
    X, y = generate_data(n_samples=n, noise_level=0.6, noisy_features=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Step 3: Fit the EARTH Model
    earth_model = EARTH(M_max=15, ridge=1e-6)
    t2 = timer()
    earth_model.fit(X_train, y_train)
    t3 = timer()
    print(f"Numba took time {round(t3-t2,5)}")

    mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=5000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X)
    y_pred_test_mlp = mlp.predict(X_test)

    dt = RandomForestRegressor(n_estimators=100, max_depth=3)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X)
    y_pred_test_dt = dt.predict(X_test)

    # Step 4: Make Predictions
    y_pred = earth_model.predict(X)
    y_pred_test = earth_model.predict(X_test)
    # y_pred_svd = earth_model_svd.predict(X)

    print(
        "sum of abs residuals MARS", np.sqrt(sum((y_test - y_pred_test) ** 2)).round(4)
    )
    print(
        "sum of abs residuals DT", np.sqrt(sum((y_test - y_pred_test_dt) ** 2)).round(4)
    )
    print(
        "sum of abs residuals MLP",
        np.sqrt(sum((y_test - y_pred_test_mlp) ** 2)).round(4),
    )
