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
    def __init__(self, model_checkpoint, dataset_path, val_dataset_path, lora_r, lora_alpha,target_modules, training_type, device='cpu'):
        self.device = device
        self.lora_r = lora_r
        self.training_type = training_type
        self.lora_alpha = lora_alpha
        self.dataset = load_from_disk(dataset_path)
        self.raw_val_dataset = load_dataset(val_dataset_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(self.device)
        self.configure_model(target_modules)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.train_data = self.prepare_data(self.dataset)
        self.val_data = self.prepare_data(self.raw_val_dataset['validation'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def configure_model(self, target_modules):
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.training_type == "LoRA":
            peft_config = LoraConfig(task_type="SEQ_CLS", r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.01, target_modules=target_modules)
            self.model = get_peft_model(self.model, peft_config)
            for name, param in self.model.named_parameters():
                if "pre_classifier.modules_to_save.default.base_layer" in name or "classifier.modules_to_save.default.base_layer" in name:
                    param.requires_grad = False
            print(self.model.print_trainable_parameters())
        elif self.training_type == "ConLoRA":
            peft_config = LoraConfig(task_type="SEQ_CLS", r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.01, target_modules=target_modules)
            self.model = get_peft_model(self.model, peft_config)
            for name, param in self.model.named_parameters():
                if "lora_A" in name or "base_layer" in name:
                    param.requires_grad = False
            print(self.model.print_trainable_parameters())

    def tokenize_function(self, examples):
        text = examples["sentence"]
        self.tokenizer.truncation_side = "left"
        tokenized_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        tokenized_inputs['labels'] = torch.tensor(examples['label'], dtype=torch.long)
        return tokenized_inputs

    def prepare_data(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return DataLoader(tokenized_dataset, batch_size=16, collate_fn=self.data_collator)

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm(range(len(self.train_data)))
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_data),
        )
        for batch in self.train_data:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss

    def evaluate(self):
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

    def get_trainable_parameters(self):
        trainable_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = (param.clone().detach(), param.requires_grad)
        return trainable_params

    def get_lora_parameters(self):
        lora_params = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_params[name] = (param.clone().detach(), param.requires_grad)
        return lora_params

    def get_all_parameters(self):
        all_params = {}
        for name, param in self.model.named_parameters():
            all_params[name] = (param.clone().detach(), param.requires_grad)
        return all_params

    def get_last_2_layers(self):
        last_2_layers_params = {}
        for name, param in self.model.named_parameters():
            if "pre_classifier" in name or "classifier" in name:
                last_2_layers_params[name] = (param.clone().detach(), param.requires_grad)
        return last_2_layers_params

    def get_lora_A(self):
        lora_A_params = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                lora_A_params[name] = (param.clone().detach(), param.requires_grad)
        return lora_A_params

    def set_trainable_parameters(self, trainable_params):
        for name, (param, requires_grad) in trainable_params.items():
            if name in dict(self.model.named_parameters()):
                current_param = dict(self.model.named_parameters())[name]
                current_param.data = param.clone().to(self.device)
                current_param.requires_grad = requires_grad

    def trainable_parameters_nums(self):
        print("Total params:", sum([p.numel() for p in self.model.parameters()]))
        print("Trainable params:", sum([p.numel() for p in self.model.parameters() if p.requires_grad == True]))

    def num_get_trainable_parameters(self):
        trainable_params = self.get_trainable_parameters()
        total_trainable_params = sum(param.numel() for param, _ in trainable_params.values())
        print(f"Total number of trainable parameters: {total_trainable_params}")

        print("Details of trainable parameters:")
        for name, (param, requires_grad) in trainable_params.items():
            print(f"Layer: {name}, Parameter count: {param.numel()}, Requires Grad: {requires_grad}")




def main():
    parser = argparse.ArgumentParser(description="Client model training script.")
    parser.add_argument('--target_modules', type=str, required=True, help="List of target modules for LoRA.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--lora_r', type=int, required=True, help="Rank of LoRA layers.")
    parser.add_argument('--lora_alpha', type=int, required=True, help="Alpha value for LoRA layers.")
    parser.add_argument('--training_type', type=str, choices=['LoRA', 'ConLoRA'], required=True, help="Training type")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run the model on (default: cpu).")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs to train (default: 5).")
    parser.add_argument('--val_dataset_path', type=str, required=True, help="Path to the validation dataset.")

    args = parser.parse_args()

    # 将传入的目标模块字符串转换为列表
    target_modules = args.target_modules.split()  # 使用空格分隔字符串并转换为列表

    client = Client(
        model_checkpoint=args.model_checkpoint,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        training_type=args.training_type,
        target_modules=target_modules,  # 传递解析后的列表
        device=args.device
    )

    for epoch in range(args.num_epochs):
        loss = client.train_one_epoch()
        accuracy = client.evaluate()
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    client.trainable_parameters_nums()

if __name__ == "__main__":
    main()
