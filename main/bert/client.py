import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset, load_from_disk
from peft import get_peft_model, LoraConfig
from tqdm.auto import tqdm
from transformers import get_scheduler
from sklearn.metrics import accuracy_score


class Client:
    def __init__(self, model_checkpoint, dataset_path, val_dataset_path, lora_r, lora_alpha, target_modules, 
                 training_type, dataset_type, device='cpu'):
        """
        Initializes the Client for model training with specified configurations.

        Args:
            model_checkpoint (str): Pre-trained model checkpoint path.
            dataset_path (str): Path to the training dataset.
            val_dataset_path (str): Path to the validation dataset.
            lora_r (int): Rank of LoRA layers.
            lora_alpha (int): Alpha value for LoRA layers.
            target_modules (list): List of target modules for LoRA.
            training_type (str): Type of training ('LoRA' or 'ConLoRA').
            dataset_type (str): Type of dataset ('sst2', 'mnli', or 'qnli').
            device (str): Device for training ('cpu' or 'cuda').
        """
        self.device = device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.training_type = training_type
        self.dataset_type = dataset_type
        self.dataset = load_from_disk(dataset_path)
        self.raw_val_dataset = load_dataset(val_dataset_path)

        # Determine number of labels based on dataset type
        self.num_labels = self._get_num_labels()

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=self.num_labels).to(self.device)

        # Configure model for LoRA training
        self.configure_model(target_modules)

        # Prepare data loaders for training and validation datasets
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_data = self.prepare_data(self.dataset)
        self.val_data = self.prepare_data(self.raw_val_dataset['validation'])

        # Optimizer setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _get_num_labels(self):
        """
        Returns the number of labels based on the dataset type.
        """
        if self.dataset_type == "sst2":
            return 2  # SST-2 is a binary classification problem
        elif self.dataset_type == "mnli":
            return 3  # MNLI is a 3-class classification problem
        elif self.dataset_type == "qnli":
            return 2  # QNLI is a binary classification problem
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

    def tokenize_function(self, examples):
        """
        Tokenizes the input text based on dataset type (sst2, mnli, qnli).
        Args:
            examples (dict): A dictionary of input examples.
        """
        if self.dataset_type == "sst2":
            text = examples["sentence"]
            self.tokenizer.truncation_side = "left"
            tokenized_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)

        elif self.dataset_type == "mnli":
            texts = (examples["premise"], examples["hypothesis"])
            self.tokenizer.truncation_side = "left"
            tokenized_inputs = self.tokenizer(*texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)

        elif self.dataset_type == "qnli":
            texts = (examples["question"], examples["sentence"])
            self.tokenizer.truncation_side = "left"
            tokenized_inputs = self.tokenizer(*texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)

        return tokenized_inputs

    def prepare_data(self, dataset):
        """
        Prepares the dataset for training or evaluation by tokenizing and formatting it.

        Args:
            dataset (Dataset): The dataset to be processed.
        """
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return DataLoader(tokenized_dataset, batch_size=16, collate_fn=self.data_collator)

    def configure_model(self, target_modules):
        """
        Configures the model for LoRA (Low-Rank Adaptation) training.

        Args:
            target_modules (list): List of target modules for LoRA.
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.training_type == "LoRA":
            peft_config = LoraConfig(task_type="SEQ_CLS", r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.01, target_modules=target_modules)
            self.model = get_peft_model(self.model, peft_config)
            self._freeze_parameters()

        elif self.training_type == "ConLoRA":
            peft_config = LoraConfig(task_type="SEQ_CLS", r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.01, target_modules=target_modules)
            self.model = get_peft_model(self.model, peft_config)
            self._freeze_parameters(freeze_lora_A=True)

    def _freeze_parameters(self, freeze_lora_A=False):
        """
        Freezes the parameters of the model based on the training type.

        Args:
            freeze_lora_A (bool): Whether to freeze 'lora_A' parameters.
        """
        for name, param in self.model.named_parameters():
            if "pre_classifier.modules_to_save.default.base_layer" in name or "classifier.modules_to_save.default.base_layer" in name:
                param.requires_grad = False
            if freeze_lora_A and "lora_A" in name:
                param.requires_grad = False
        print(self.model.print_trainable_parameters())

    def train_one_epoch(self):
        """
        Trains the model for one epoch.

        Returns:
            loss (float): The loss value for the epoch.
        """
        self.model.train()
        progress_bar = tqdm(range(len(self.train_data)))
        lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_data))
        
        for batch in self.train_data:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            #lr_scheduler.step()
            self.optimizer.zero_grad()
            progress_bar.update(1)
        
        return loss.item()

    def evaluate(self):
        """
        Evaluates the model on the validation dataset.

        Returns:
            accuracy (float): The accuracy on the validation dataset.
        """
        self.model.eval()
        all_predictions = []
        all_labels = []

        for batch in self.val_data:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy

    def print_trainable_parameters(self):
        """
        Prints the number of trainable and total parameters in the model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    def num_get_trainable_parameters(self):
        """
        Prints detailed information about each trainable parameter.
        """
        trainable_params = self.get_trainable_parameters()
        total_trainable_params = sum(param.numel() for param, _ in trainable_params.values())
        print(f"Total number of trainable parameters: {total_trainable_params}")

        print("Details of trainable parameters:")
        for name, (param, requires_grad) in trainable_params.items():
            print(f"Layer: {name}, Parameter count: {param.numel()}, Requires Grad: {requires_grad}")

    def get_trainable_parameters(self):
        """
        Returns the trainable parameters of the model.
        """
        trainable_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = (param.clone().detach(), param.requires_grad)
        return trainable_params


def main():
    parser = argparse.ArgumentParser(description="Client model training script.")
    parser.add_argument('--target_modules', type=str, required=True, help="List of target modules for LoRA.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--lora_r', type=int, required=True, help="Rank of LoRA layers.")
    parser.add_argument('--lora_alpha', type=int, required=True, help="Alpha value for LoRA layers.")
    parser.add_argument('--training_type', type=str, choices=['LoRA', 'ConLoRA'], required=True, help="Training type.")
    parser.add_argument('--device', type=str, default='cpu', help="Device for training (default: cpu).")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train (default: 5).")
    parser.add_argument('--val_dataset_path', type=str, required=True, help="Path to the validation dataset.")
    parser.add_argument('--dataset_type', type=str, choices=['sst2', 'mnli', 'qnli'], required=True, help="Type of dataset.")

    args = parser.parse_args()

    # Convert target_modules string to list
    target_modules = args.target_modules.split()

    client = Client(
        model_checkpoint=args.model_checkpoint,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        training_type=args.training_type,
        target_modules=target_modules,
        device=args.device,
        dataset_type=args.dataset_type
    )

    # Training loop
    for epoch in range(args.num_epochs):
        loss = client.train_one_epoch()
        accuracy = client.evaluate()
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    client.print_trainable_parameters()


if __name__ == "__main__":
    main()
