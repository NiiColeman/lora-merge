import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from peft import LoraConfig, get_peft_model
from avalanche.benchmarks import SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
# from avalanche.training.determinism import set_deterministic_runß
from tqdm import tqdm
import time
import logging
import os
import torchvision
from torchvision import transforms


# Setup logging
timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir = 'experiment_outs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, f'slora_cifar100_{timestr}.log'),
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

train_transform= transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Configs
NUM_TASKS = 10
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_R = 8
LORA_ALPHA = 1
LORA_DROPOUT = 0.1

def create_benchmark():
    train_set, test_set =torchvision.datasets.CIFAR100(root='./data', train=False, download=False,transform=train_transform), torchvision.datasets.CIFAR100(root='./data', train=False, download=True,transform=test_transform)
    scenario=nc_benchmark(train_dataset=train_set, test_dataset=test_set, n_experiences=NUM_TASKS, shuffle=True, seed=23,task_labels=False) 

    return scenario


def create_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    return model

class SLoRA:
    def __init__(self, base_model, num_tasks):
        self.base_model = base_model.to(DEVICE)
        self.num_tasks = num_tasks
        self.task_alphas = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_tasks)])
        self.classifier = nn.Linear(768, 100).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.seen_classes = set()
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=1,  # Set to 1 since we handle scaling
            target_modules=["qkv"],
            lora_dropout=LORA_DROPOUT,
            bias="none"
        )
        self.lora_model = get_peft_model(base_model, lora_config)
        
    def forward(self, x):
        base_output = self.base_model(x)
        lora_output = self.lora_model(x) - base_output
        
        # Normalize LoRA update
        lora_norm = torch.norm(lora_output)
        if lora_norm > 0:
            lora_output = lora_output / lora_norm
            
        return base_output + self.task_alphas[0] * lora_output
        
    def train_task(self, train_loader, task_id, current_classes):
        self.seen_classes.update(current_classes)
        self.lora_model.train()
        
        if task_id == 0:
            trainable_params = [
                {'params': self.lora_model.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.task_alphas[task_id]}
            ]
        else:
            for param in self.lora_model.parameters():
                param.requires_grad = False
            trainable_params = [
                {'params': self.classifier.parameters()},
                {'params': self.task_alphas[task_id]}
            ]
            
        optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=0.9)
        
        for epoch in tqdm(range(EPOCHS), desc=f'Task {task_id} Training'):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                features = self.lora_model(inputs)
                outputs = self.classifier(features)
                
                # Mask only future unseen classes during training
                mask = torch.zeros_like(outputs, device=DEVICE)
                mask[:, list(self.seen_classes)] = 1
                outputs = outputs * mask
                
                loss = self.criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_acc = 100. * correct / total
            logging.info(f'Task {task_id}, Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, Acc = {train_acc:.2f}%')
            
        return avg_epoch_loss

    def evaluate(self, test_loader):
        self.lora_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                features = self.lora_model(inputs)
                outputs = self.classifier(features)
                
                # No masking during evaluation - test on all seen classes
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

def main():
    # set_deterministic_run(seed=42)
    logging.info("Starting S-LoRA CIFAR-100 experiment")
    
    base_model = create_model()
    scenario = create_benchmark()
    slora = SLoRA(base_model, NUM_TASKS)
    
    for task_id, experience in enumerate(scenario.train_stream):
        logging.info(f"\nStarting task {task_id}")
        current_classes = experience.classes_in_this_experience
        
        train_loader = DataLoader(experience.dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        avg_loss = slora.train_task(train_loader, task_id, current_classes)
        logging.info(f"Task {task_id} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluate on all seen tasks
        for eval_task_id in range(task_id + 1):
            eval_loader = DataLoader(scenario.test_stream[eval_task_id].dataset, batch_size=BATCH_SIZE)
            accuracy = slora.evaluate(eval_loader)
            logging.info(f"Accuracy on task {eval_task_id}: {accuracy:.2f}%")

if __name__ == "__main__":
    logging.info("""
    Summary of S-LoRA CIFAR-100 Benchmark

    1. Benchmark Setup:
    - Split CIFAR100 into 10 tasks using Avalanche
    - Each task has 10 classes with class-incremental setup

    2. SLoRA Class Structure:
    - Manages LoRA directions, alphas, and Incremental classifier + mask
    - ParameterList for alphas allows independent learning per task

    3. Training Strategy:
    - Task 0: Train LoRA direction (AB matrices) and alpha
    - Tasks 1-9: Freeze LoRA direction, only train new alpha and classifier
    - Uses masks to focus on current task classes

    4. Evaluation:
    - Tests each trained task on all previous tasks
    - Tracks accuracy using incrementally trained classifier with mask
    - Maintains running evaluation metrics

    5. Logging:
    - Timestamped files for experiment tracking
    - Records loss, accuracy, and key events
    - Uses tqdm for progress visualization

   
    """)

    main()




    