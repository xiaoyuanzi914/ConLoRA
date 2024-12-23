import os
import torch
import logging
import numpy as np
from client import Client
from typing import List, Dict


class FederatedServer:
    def __init__(self, clients: List[Client]) -> None:
        """
        Initializes the server with a list of clients.

        Args:
            clients (list): List of Client objects.
        """
        self.clients = clients
        self.num_clients = len(clients)
        
        # Set up logging
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'server_init.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def clients_initialize_info(self) -> None:
        """
        Initializes each client by loading the model and dataset.
        """
        logging.info("Initializing clients...")
        for i, client in enumerate(self.clients):
            logging.info(f"Initializing client {i+1}...")
            client.print_trainable_parameters()  # Just to verify the model parameters
            accuracy = client.evaluate()  # Initial evaluation
            logging.info(f"Initial accuracy of client {i+1}: {accuracy}")

    def aggregate_lora_A(self) -> None:
        """
        Aggregate LoRA parameters from the first client and set them as trainable parameters
        for all other clients to ensure consistency.
        """
        logging.info("Aggregating LoRA parameters...")
        avg_lora_A_params = self.clients[0].get_lora_A()

        # Set the LoRA A parameters for all clients
        for client in self.clients:
            client.set_trainable_parameters(avg_lora_A_params)

    def aggregate_last_2_layers_params(self) -> None:
        """
        Aggregates the last 2 layers' parameters across all clients.
        """
        logging.info("Aggregating last 2 layers' parameters...")
        avg_params = self.clients[0].get_last_2_layers()

        # Set the last 2 layers' parameters for all clients
        for client in self.clients:
            client.set_trainable_parameters(avg_params)

    def aggregate_dfl(self, A: np.array) -> None:
        """
        Aggregates LoRA parameters using a weighted average, based on a matrix A.

        Args:
            A (np.array): A weight matrix used for aggregation.
        """
        logging.info("Aggregating LoRA parameters with DFL method...")
        self.new_params = []

        # Initialize each client's parameters as zero
        for i in range(self.num_clients):
            client_params = self.clients[i].get_lora_parameters()
            zero_params = {name: (torch.zeros_like(param[0]), param[1]) for name, param in client_params.items()}
            self.new_params.append(zero_params)

        # Perform aggregation based on weight matrix A
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                client_params = self.clients[j].get_lora_parameters()
                for name, (param, requires_grad) in client_params.items():
                    self.new_params[i][name] = (self.new_params[i][name][0] + param * A[i][j], requires_grad)

        # Update each client's parameters
        for i in range(self.num_clients):
            self.clients[i].set_trainable_parameters(self.new_params[i])

        logging.info("LoRA parameters aggregated with DFL.")

    def extract_and_multiply_lora_params(self, param_group: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
        """
        Extracts LoRA A and LoRA B parameters and computes their product.

        Args:
            param_group (dict): Dictionary of model parameters.

        Returns:
            dict: A dictionary containing the product of LoRA A and LoRA B.
        """
        result = {}
        for param_name, (param, _) in param_group.items():
            if 'lora_B.default.weight' in param_name:
                prefix = param_name.split('lora_B.default.weight')[0]
                lora_A_name = prefix + 'lora_A.default.weight'
                lora_B_name = prefix + 'lora_B.default.weight'

                if lora_A_name in param_group and lora_B_name in param_group:
                    lora_A = param_group[lora_A_name][0]
                    lora_B = param_group[lora_B_name][0]

                    product = torch.matmul(lora_B, lora_A)
                    result[prefix + 'product'] = product

        return result

    def calculate_lora_products_and_avg_diff(self, param_groups: List[Dict[str, tuple]]) -> float:
        """
        Calculates the average difference between the LoRA parameter products for all clients.

        Args:
            param_groups (list): A list of parameter groups from the clients.

        Returns:
            float: The average difference between LoRA parameter products.
        """
        if len(param_groups) < 2:
            raise ValueError("There should be at least two sets of parameters to calculate differences.")
        
        # Calculate average LoRA parameters across all clients
        avg_params = self.clients[0].get_lora_parameters()
        avg_params = {name: (torch.zeros_like(param[0]), param[1]) for name, param in avg_params.items()}
        for i in range(self.num_clients):
            client_params = self.clients[i].get_lora_parameters()
            for name, (param, _) in client_params.items():
                avg_params[name] = (avg_params[name][0] + param / self.num_clients, _)

        total_diff_sum = 0.0
        num_pairs = 0

        # Compute product differences for each pair of clients
        for i in range(self.num_clients):
            product_1 = self.extract_and_multiply_lora_params(param_groups[i])
            product_2 = self.extract_and_multiply_lora_params(avg_params)
            pair_diff_sum = 0.0

            # Calculate the difference between the products
            for key in product_1.keys():
                diff = product_1[key] - product_2[key]
                pair_diff_sum += torch.norm(diff).item()

            total_diff_sum += pair_diff_sum
            num_pairs += 1

        # Return the average difference
        average_diff = total_diff_sum / num_pairs if num_pairs > 0 else 0.0
        return average_diff


def main() -> None:
    """
    Main function to initialize clients and federated server, and perform client initialization.
    """
    # Configuration parameters
    model_checkpoint = '/home/ubuntu/smyin/models/distilbert-base-uncased'
    dataset_path_template = "/home/ubuntu/smyin/dataset/decentrilized_dataset/sst2_020/client_{}"
    val_dataset_path_template = "/home/ubuntu/smyin/dataset/glue/sst2"
    num_clients = 7
    lora_r = 4
    lora_alpha = 32
    target_modules = ["q_lin", "v_lin", "pre_classifier", "pre_classifier"]  # Example modules
    training_type = "LoRA"
    dataset_type = "sst2"
    batch_size=128

    # Initialize clients
    clients = []
    for i in range(num_clients):
        dataset_path = dataset_path_template.format(i + 1)
        val_dataset_path = val_dataset_path_template
        client = Client(
            model_checkpoint=model_checkpoint, 
            dataset_path=dataset_path, 
            val_dataset_path=val_dataset_path, 
            lora_r=lora_r, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            training_type=training_type, 
            dataset_type=dataset_type, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=batch_size
        )
        clients.append(client)

    # Initialize federated server
    server = FederatedServer(clients)

    # Initialize and evaluate the clients
    server.clients_initialize_info()


if __name__ == "__main__":
    main()
