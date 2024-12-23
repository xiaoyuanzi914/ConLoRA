import argparse
from datasets import load_dataset
import numpy as np
import os

class DataSplitter:
    def __init__(self, dataset_path, num_clients, alpha=None):
        """
        Initializes the DataSplitter class.

        Args:
            dataset_path (str): Path to the dataset.
            num_clients (int): Number of clients.
            alpha (float): Dirichlet distribution parameter for non-IID split.
        """
        self.dataset = load_dataset(dataset_path, "main")
        self.train_dataset = self.dataset['train']
        self.num_clients = num_clients
        self.alpha = alpha
        self.clients_datasets = []

    def iid_split(self):
        """Splits the dataset into IID partitions across clients."""
        shuffled_dataset = self.train_dataset.shuffle(seed=42)
        client_size = len(shuffled_dataset) // self.num_clients
        self.clients_datasets = []

        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = (i + 1) * client_size if i != self.num_clients - 1 else len(shuffled_dataset)
            client_dataset = shuffled_dataset.select(range(start_idx, end_idx))
            self.clients_datasets.append(client_dataset)

    def dirichlet_split(self):
        """Splits the dataset into non-IID partitions using Dirichlet distribution."""
        total_size = len(self.train_dataset)
        proportions = np.random.dirichlet([self.alpha] * self.num_clients)
        client_sizes = (proportions * total_size).astype(int)
        client_sizes[-1] = total_size - client_sizes[:-1].sum()

        indices = np.random.permutation(total_size)
        current_index = 0
        self.clients_datasets = []

        for size in client_sizes:
            client_indices = indices[current_index: current_index + size]
            self.clients_datasets.append(self.train_dataset.select(client_indices.tolist()))
            current_index += size

    def compute_data_distribution(self):
        """Computes the distribution of data among clients."""
        return [len(client_dataset) for client_dataset in self.clients_datasets]

    def save_datasets(self, base_path):
        """Saves client datasets to disk."""
        os.makedirs(base_path, exist_ok=True)
        for i, client_dataset in enumerate(self.clients_datasets):
            client_dataset.save_to_disk(os.path.join(base_path, f"client_{i+1}"))


def main():
    parser = argparse.ArgumentParser(description="Dataset splitter for federated learning.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--output_path', type=str, required=True, help="Output path for saving client datasets.")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of clients.")
    parser.add_argument('--split_type', type=str, choices=['iid', 'dirichlet'], required=True, help="Type of split: 'iid' or 'dirichlet'.")
    parser.add_argument('--alpha', type=float, default=None, help="Alpha value for Dirichlet distribution.")

    args = parser.parse_args()

    # Initialize DataSplitter
    splitter = DataSplitter(args.dataset_path, args.num_clients, args.alpha)

    # Perform split
    if args.split_type == 'iid':
        splitter.iid_split()
    elif args.split_type == 'dirichlet':
        if args.alpha is None:
            parser.error("--alpha must be specified for Dirichlet split.")
        splitter.dirichlet_split()

    # Save datasets
    splitter.save_datasets(args.output_path)

    # Print data distribution
    distribution = splitter.compute_data_distribution()
    print("Client data distribution:", distribution)


if __name__ == "__main__":
    main()
