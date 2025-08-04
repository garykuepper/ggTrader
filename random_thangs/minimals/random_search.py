import numpy as np

# Example black-box function
def blackbox(x, y):
    return (x - 2)**2 + (y + 3)**2 + np.random.normal(0, 0.1)  # noisy paraboloid

def random_search(iterations=10):
    # Parameter space
    x_range = (-10, 10)
    y_range = (-10, 10)

    best_params = None
    best_value = float('inf')

    n_iterations = iterations

    for _ in range(n_iterations):
        # x = np.random.uniform(*x_range)
        # y = np.random.uniform(*y_range)
        x = np.random.randint(x_range[0], x_range[1] + 1)
        y = np.random.randint(y_range[0], y_range[1] + 1)

        value = blackbox(x, y)
        # print(F"x: {x}, y: {y}, value: {value}")
        if value < best_value:
            best_value = value
            best_params = (x, y)

    print(f"Best value: {best_value:.4f} at x={best_params[0]:.4f}, y={best_params[1]:.4f}")


if __name__ == "__main__":
    random_search(iterations=10)
    random_search(iterations=100)
    random_search(iterations=1000)
    random_search(iterations=10000)
