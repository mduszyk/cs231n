import itertools
import numpy as np


def grid_layout(params):
    param_names = []
    param_values = []
    for key, values in params.items():
        param_names.append(key)
        param_values.append(values)
    for param_set in itertools.product(*param_values):
        yield dict(zip(param_names, param_set))


def random_layout(params, n=10):
    for _ in range(n):
        param_set = {}
        for key, fn in params.items():
            param_set[key] = fn()
        yield param_set


if __name__ == "__main__":
    grid_params = {
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "dropout": [0.25, 0.5, 0.75]
    }
    print("Grid layout for params: %s" % grid_params)
    for param_set in grid_layout(grid_params):
        print(param_set)

    random_params = {
        "learning_rate": lambda: 10 ** np.random.uniform(-6, 1),
        "dropout": lambda: np.random.uniform(0, 1)
    }
    print("Random layout for params: %s" % random_params)
    for param_set in random_layout(random_params, n=7):
        print(param_set)


