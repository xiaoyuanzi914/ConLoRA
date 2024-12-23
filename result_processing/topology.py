import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define adjacency matrices for the topologies
topologies = {
    "T1": np.array([
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0]
    ]),
    "T2": np.array([
        [0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0]
    ]),
    "T3": np.array([
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0]
    ]),
    "T4": np.array([
        [0, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0]
    ])
}

# Plot the topologies
# First, create the base ring structure from T1
num_nodes = topologies["T1"].shape[0]
base_graph = nx.Graph()
for i in range(num_nodes):
    base_graph.add_edge(i, (i + 1) % num_nodes)

# Add additional edges based on T1 adjacency matrix
for i in range(num_nodes):
    for j in range(num_nodes):
        if topologies["T1"][i, j] == 1 and not base_graph.has_edge(i, j):
            base_graph.add_edge(i, j)

# Define a fixed layout for consistent node positioning
fixed_pos = nx.spring_layout(base_graph, seed=42)

# Draw the base graph T1
plt.figure(figsize=(8, 6))
nx.draw(base_graph, pos=fixed_pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=30, font_weight='bold', edge_color='gray', width=6)
plt.savefig("T1.png", format='png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the other topologies based on T1
for name, adjacency_matrix in topologies.items():
    if name == "T1":
        continue

    G = base_graph.copy()

    # Add additional edges based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i, j] == 1 and not G.has_edge(i, j):
                G.add_edge(i, j)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=fixed_pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=30, font_weight='bold', edge_color='gray', width=6)
    plt.savefig(f"{name}.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()


