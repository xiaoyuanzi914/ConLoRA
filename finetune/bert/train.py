import os
import sys
import logging
import torch
import argparse

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # code 根目录
sys.path.append(parent_dir)

# 导入模块
from utils.generate_weight_matrix import generate_weight_matrix
from server import FederatedServer
from client import Client

def train_federated_model(model_checkpoint, dataset_path_template, val_dataset_path_template, num_clients, lora_r, lora_alpha, target_modules, training_type, dataset_type, name, num_rounds=256, batch_size=16, log_path="federated_training.log"):
    """
    Runs the federated learning training process for the given number of rounds.

    Args:
        model_checkpoint (str): Path to the pre-trained model.
        dataset_path_template (str): Template for the dataset paths for each client.
        val_dataset_path_template (str): Path template for validation dataset.
        num_clients (int): Number of clients participating in federated learning.
        lora_r (int): LoRA rank for LoRA layers.
        lora_alpha (int): LoRA alpha for LoRA layers.
        target_modules (list): List of target modules for LoRA layers.
        training_type (str): Type of training ('LoRA', 'ConLoRA').
        dataset_type (str): Type of dataset ('sst2', 'mnli', or 'qnli').
        name (str): The name used to generate the weight matrix.
        num_rounds (int): Number of federated learning rounds (default is 256).
        log_path (str): Path to save the log file.
    """
    # Initialize clients
    clients = []
    for i in range(num_clients):
        dataset_path = dataset_path_template.format(i+1)
        val_dataset_path = val_dataset_path_template
        client = Client(model_checkpoint=model_checkpoint, 
                        dataset_path=dataset_path, 
                        val_dataset_path=val_dataset_path, 
                        lora_r=lora_r, 
                        lora_alpha=lora_alpha, 
                        target_modules=target_modules, 
                        training_type=training_type, 
                        dataset_type=dataset_type, 
                        batch_size=batch_size,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
        clients.append(client)

    # Initialize the federated server with clients
    server = FederatedServer(clients)

    # Initial aggregation of parameters
    server.aggregate_last_2_layers_params()
    server.aggregate_lora_A()

    # Initialize accuracy and parameter difference tracking
    acc = [[] for _ in range(num_clients)]
    diff = []

    # Get the weight matrix A for aggregation
    A = generate_weight_matrix(name)

    # Initial evaluation of all clients
    for i in range(server.num_clients):
        accuracy = server.clients[i].evaluate()
        acc[i].append(accuracy)
        logging.info(f'Initial accuracy of client {i+1}: {accuracy}')

    # Aggregate LoRA parameters with DFL
    server.aggregate_dfl(A)

    # Calculate and log the initial difference in LoRA products
    a = server.calculate_lora_products_and_avg_diff(server.new_params)
    diff.append(a)

    # Training loop for the given number of rounds
    for round in range(num_rounds):
        logging.info(f"Round {round+1}/{num_rounds}")
        
        # Train each client for one epoch
        for client in server.clients:
            loss = client.train_one_epoch()
            accuracy = client.evaluate()
            logging.info(f"Client {server.clients.index(client) + 1} - Loss: {loss}, Accuracy: {accuracy}")

        # Aggregate parameters with DFL
        server.aggregate_dfl(A)

        # Calculate and log the difference in LoRA products after aggregation
        a = server.calculate_lora_products_and_avg_diff(server.new_params)
        diff.append(a)
        logging.info(f"Round {round+1} - Parameter Difference: {a}")

        # Evaluate all clients after the round
        for i in range(server.num_clients):
            accuracy = server.clients[i].evaluate()
            acc[i].append(accuracy)
            logging.info(f'Accuracy of client {i+1} after round {round+1}: {accuracy}')

        # Log the parameter difference and accuracies after the round
        logging.info(f"Parameter Difference record: {diff}")
        logging.info(f'Accuracies after round {round+1}: {acc}')


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Federated Learning Training Script")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--dataset_path_template', type=str, required=True, help="Template for the dataset paths for each client.")
    parser.add_argument('--val_dataset_path_template', type=str, required=True, help="Path template for validation dataset.")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of clients participating in federated learning.")
    parser.add_argument('--lora_r', type=int, required=True, help="LoRA rank for LoRA layers.")
    parser.add_argument('--lora_alpha', type=int, required=True, help="LoRA alpha for LoRA layers.")
    parser.add_argument('--target_modules', type=str, required=True, help="Comma-separated list of target modules for LoRA layers.")
    parser.add_argument('--training_type', type=str, choices=['LoRA', 'ConLoRA'], required=True, help="Training type ('LoRA' or 'ConLoRA').")
    parser.add_argument('--dataset_type', type=str, choices=['sst2', 'mnli', 'qnli'], required=True, help="Dataset type.")
    parser.add_argument('--name', type=str, required=True, help="The name used to generate the weight matrix.")
    parser.add_argument('--num_rounds', type=int, default=256, help="Number of federated learning rounds.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for each client.")
    parser.add_argument('--log_path', type=str, default="federated_training.log", help="Path to save the log file.")

    args = parser.parse_args()

    # Parse target_modules into list
    target_modules = args.target_modules.split(',')

    # Configure logging to the specified log file path
    logging.basicConfig(filename=args.log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Run federated training process
    train_federated_model(
        model_checkpoint=args.model_checkpoint,
        dataset_path_template=args.dataset_path_template,
        val_dataset_path_template=args.val_dataset_path_template,
        num_clients=args.num_clients,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        training_type=args.training_type,
        dataset_type=args.dataset_type,
        name=args.name,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        log_path=args.log_path  # Pass log_path parameter
    )
