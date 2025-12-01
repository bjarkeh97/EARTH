# npearth

A NumPy-based implementation of the Multivariate Adaptive Regression Splines (MARS) algorithm, following Friedman’s original paper, with support for weighted regression. The goal of this project is to provide a lightweight, efficient, and transparent implementation suitable for research, numerical experiments, and scientific computing.

## Overview

`npearth` implements the main components of the EARTH algorithm:

* Forward pass using hinge functions and interaction terms
* Backward pruning based on Generalized Cross Validation (GCV)
* Efficient regression fitting using Cholesky updates
* Weighted regression support
* Modular and extensible architecture

The package is implemented using NumPy and Numba, with a focus on clarity, performance, and maintainability.

## Installation

The package is currently distributed directly from GitHub. Install using:

```
pip install git+https://github.com/bjarkeh97/EARTH.git
```

Alternatively, clone the repository manually:

```
git clone https://github.com/bjarkeh97/EARTH
cd EARTH
pip install -e .
```

## Quick Start

A minimal example of fitting an EARTH model:

```python
import numpy as np
from npearth.earth import Earth

# Generate sample data
x = np.linspace(0, 10, 200)
y = np.sin(x) + 0.1 * np.random.randn(200)
X = x.reshape(-1, 1)

# Fit model
model = Earth()
model.fit(X, y)

# Predict
y_pred = model.predict(X)
```

## Project Structure

```
npearth/
  earth.py                     High-level EARTH model class
  _forward_pass.py             Forward pass implementation
  _backward_pass.py            Backward pruning and model simplification
  _basis_function.py           Basis function representation
  _knotsearcher_*              Knot search strategies and optimizations
  _cholesky_update.py          Rank-1 Cholesky update utilities
  data/                        Sample datasets for testing and examples
examples/                      Example scripts
tests/                         Unit tests
```

## Examples

Example scripts demonstrating typical use cases are available in the `examples/` directory:

* `example_linear.py`
* `example_sine.py`
* `example_complex_multidim.py`
* `example_weights.py`

Run an example with:

```
python examples/example_sine.py
```

## Contributing

Contributions are welcome. Areas where contributions are particularly helpful include improvements to numerical performance, additional diagnostics, enhanced loss functions, and expanded test coverage.

To contribute:

1. Fork the repository on GitHub
2. Create a branch for your feature or fix
3. Commit and push your changes
4. Submit a pull request

## Roadmap

The following items may be added in future releases:

* Packaging and release on PyPI
* Model diagnostics, including ANOVA decompositions as in Friedman (1991)
* Additional statistical summaries and tools

## License

This project is licensed under the MIT License.

## Acknowledgements

This implementation is inspired by foundational and prior work on MARS models, including:

* Jerome H. Friedman (1991): *Multivariate Adaptive Regression Splines*, The Annals of Statistics, 19(1), 1–67.
* The py-earth project: [https://github.com/scikit-learn-contrib/py-earth](https://github.com/scikit-learn-contrib/py-earth)

These works provided important reference material for algorithmic details, practical considerations, and design structure.
