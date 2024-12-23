import numpy as np

def generate_weight_matrix(name):
    """
    Generates a weight matrix based on the provided name.

    Args:
        name (str): The name for selecting the weight matrix configuration.

    Returns:
        np.array: The weight matrix for aggregation.
    """
    if name == "du14":
        A = np.array([
            [0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0]
        ])
    elif name == "link3":
        A = np.array([
            [0, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0]
        ])
    elif name == "link4":
        A = np.array([
            [0, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0]
        ])
    elif name == "link5":
        A = np.array([
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0]
        ])
    elif name == "link6":
        A = np.array([
            [0, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0]
        ])
    elif name == "link7":
        A = np.array([
            [0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0]
        ])
    else:
        raise ValueError(f"Unknown matrix name: {name}")
    
    # Calculate degree of each node and generate weight matrix
    degree = np.sum(A, axis=1)
    
    # Initialize weight matrix W
    W = np.zeros_like(A, dtype=float)

    # Calculate weight matrix
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                W[i, j] = 1 / (max(degree[i], degree[j]) + 1)
            elif i != j:
                W[i, j] = 0

    # Adjust diagonal elements to ensure each row sums to 1
    for i in range(A.shape[0]):
        W[i, i] = 1 - np.sum(W[i, np.arange(A.shape[0]) != i])

    return W
