import argparse
import numpy as np
from datasets import load_dataset
from collections import Counter
import os

class DataSplitter:
    def __init__(self, dataset_path, num_clients, alpha=None, min_size=None):
        """
        Initializes the DataSplitter class.

        Args:
            dataset_path (str): Path to the dataset.
            num_clients (int): Number of clients.
            alpha (float): Dirichlet distribution parameter for non-IID split.
            min_size (int): Minimum size of data for a client (optional).
        """
        self.dataset = load_dataset(dataset_path)
        self.train_dataset = self.dataset['train']
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_size = min_size
        self.clients_datasets = []

    def iid_split(self):
        """
        Splits the dataset into IID partitions across clients.
        """
        shuffled_dataset = self.train_dataset.shuffle(seed=42)
        client_size = len(shuffled_dataset) // self.num_clients
        self.clients_datasets = []

        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = (i + 1) * client_size if i != self.num_clients - 1 else len(shuffled_dataset)
            client_dataset = shuffled_dataset.select(range(start_idx, end_idx))
            self.clients_datasets.append(client_dataset)

    def dirichlet_split(self):
        """
        Splits the dataset into non-IID partitions using Dirichlet distribution.
        """
        self.clients_datasets = [[] for _ in range(self.num_clients)]
        labels = np.array([example['label'] for example in self.train_dataset])
        num_classes = len(np.unique(labels))

        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        for c in range(num_classes):
            class_size = len(class_indices[c])
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            client_sizes = (proportions * class_size).astype(int)
            client_sizes[-1] = class_size - np.sum(client_sizes[:-1])

            current_index = 0
            for i, size in enumerate(client_sizes):
                self.clients_datasets[i].extend(class_indices[c][current_index:current_index + size].tolist())
                current_index += size

        # Ensure each client has enough data
        for attempt in range(1000):
            total_sizes = [len(client_dataset) for client_dataset in self.clients_datasets]
            if all(size >= self.min_size for size in total_sizes):
                break
            if any(size < self.min_size for size in total_sizes):
                self.clients_datasets = [[] for _ in range(self.num_clients)]
                for c in range(num_classes):
                    class_size = len(class_indices[c])
                    proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                    client_sizes = (proportions * class_size).astype(int)
                    client_sizes[-1] = class_size - np.sum(client_sizes[:-1])
                    current_index = 0
                    for i, size in enumerate(client_sizes):
                        self.clients_datasets[i].extend(class_indices[c][current_index:current_index + size].tolist())
                        current_index += size
            else:
                raise ValueError(f"Unable to adjust client sizes after {attempt + 1} attempts.")
        else:
            raise ValueError("Unable to generate a valid split after 1000 attempts.")

        # Convert indices to actual data and shuffle
        for i in range(self.num_clients):
            if len(self.clients_datasets[i]) == 0:
                print(f"Client {i} dataset is empty!")
            else:
                np.random.shuffle(self.clients_datasets[i])
                self.clients_datasets[i] = self.train_dataset.select(self.clients_datasets[i])

    def compute_label_distribution(self, dataset):
        """
        Computes the label distribution for the given dataset.
        """
        labels = [example['label'] for example in dataset]
        return Counter(labels)

    def label_distribution(self):
        """
        Computes the label distribution for each client dataset.
        """
        self.pct = []
        for i in range(self.num_clients):
            self.pct.append(self.compute_label_distribution(self.clients_datasets[i]))

    def save_datasets(self, base_path):
        """
        Saves the client datasets to disk.
        """
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
    parser.add_argument('--min_size', type=int, default=0, help="Minimum size for each client dataset.")

    args = parser.parse_args()

    splitter = DataSplitter(args.dataset_path, args.num_clients, args.alpha, args.min_size)

    # Perform split
    if args.split_type == 'iid':
        splitter.iid_split()
    elif args.split_type == 'dirichlet':
        if args.alpha is None:
            parser.error("--alpha must be specified for Dirichlet split.")
        splitter.dirichlet_split()

    # Save datasets
    splitter.save_datasets(args.output_path)

    # Print label distribution
    splitter.label_distribution()
    print("Label distribution per client:")
    for i, dist in enumerate(splitter.pct):
        print(f"Client {i+1}: {dist}")

if __name__ == "__main__":
    main()
