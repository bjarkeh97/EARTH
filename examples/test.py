import os, sys
from time import time as timer

sys.path.append(os.path.abspath(""))

import numpy as np
from src.earth.earth import EARTH
from src.earth.earth_slow import EARTH as EARTH_SLOW

if __name__ == "__main__":
    # Simple example for testing
    # Step 1: Generate sample data (100 samples with 3 features)
    np.random.seed(42)  # For reproducibility
    N = 1000
    X = np.random.rand(N, 3)  # 100 samples, 3 features
    y = (
        2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2] + 0.0 * np.random.randn(N)
    )  # Linear combination with noise

    # Step 2: Create an instance of EARTH and fit the model
    earth_model = EARTH_SLOW(M_max=10)
    t0 = timer()
    earth_model.fit(X, y)
    print("Took time ", round(timer() - t0, 3), "seconds")
    print("earth coefs ", earth_model.coeffs)
    # Step 3: Make predictions on the same input
    y_pred = earth_model.predict(X)

    # Step 4: Print the results (printing just the first 5 for brevity)
    print("Original y values (first 5):", y[:5])
    print("Predicted y values (first 5):", y_pred[:5])
    print("SSR", ((y - y_pred) ** 2).sum())
