import argparse
from datasets import load_from_disk
from collections import Counter
import os

class DataLoader:
    def __init__(self, base_path, num_clients, dataset_type):
        """
        Initializes the DataLoader class.

        Args:
            base_path (str): Path where the client datasets are stored.
            num_clients (int): Number of clients.
            dataset_type (str): Type of dataset ('glue' or 'gsm8k').
        """
        self.base_path = base_path
        self.num_clients = num_clients
        self.dataset_type = dataset_type.lower()
        self.clients_datasets = self.load_datasets()

    def load_datasets(self):
        """
        Loads the client datasets from disk.

        Returns:
            list: A list of datasets for each client.
        """
        clients_datasets = []
        for i in range(self.num_clients):
            client_path = os.path.join(self.base_path, f"client_{i+1}")
            if os.path.exists(client_path):
                try:
                    client_dataset = load_from_disk(client_path)
                    clients_datasets.append(client_dataset)
                except Exception as e:
                    print(f"Error loading dataset for Client {i+1}: {e}")
                    clients_datasets.append(None)
            else:
                print(f"Warning: Client {i+1} dataset not found at {client_path}.")
                clients_datasets.append(None)
        return clients_datasets

    def compute_label_distribution(self, dataset):
        """
        Computes the label distribution for the given dataset.

        Args:
            dataset (Dataset): A Hugging Face dataset.

        Returns:
            Counter: A counter of label occurrences.
        """
        try:
            labels = [example['label'] for example in dataset]
            return Counter(labels)
        except KeyError:
            raise ValueError("The dataset does not contain a 'label' field.")

    def view_data_distribution(self):
        """
        Views data distribution based on dataset type ('glue' or 'gsm8k').
        """
        for i in range(self.num_clients):
            if self.clients_datasets[i]:
                if self.dataset_type == 'glue':
                    try:
                        print(f"Label distribution for Client {i+1}:")
                        dist = self.compute_label_distribution(self.clients_datasets[i])
                        print(dist)
                    except ValueError as e:
                        print(f"Error processing Client {i+1}: {e}")
                elif self.dataset_type == 'gsm8k':
                    print(f"Client {i+1} has {len(self.clients_datasets[i])} examples.")
            else:
                print(f"No dataset found for Client {i+1}.")

def main():
    parser = argparse.ArgumentParser(description="Load and view the distribution of split datasets.")
    parser.add_argument('--base_path', type=str, required=True, help="Path to the directory containing client datasets.")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of clients.")
    parser.add_argument('--dataset_type', type=str, choices=['glue', 'gsm8k'], required=True, help="Type of dataset ('glue' or 'gsm8k').")

    args = parser.parse_args()

    # Validate base path
    if not os.path.exists(args.base_path):
        raise FileNotFoundError(f"The specified base path does not exist: {args.base_path}")

    # Validate number of clients
    available_clients = len([d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))])
    if args.num_clients > available_clients:
        raise ValueError(f"The specified number of clients ({args.num_clients}) exceeds the available clients ({available_clients}).")

    # Load and view data distribution
    loader = DataLoader(args.base_path, args.num_clients, args.dataset_type)
    loader.view_data_distribution()

if __name__ == "__main__":
    main()
