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
from peft.tuners.lora.dora import DoraLinearLayer


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
EPOCHS = 10
BATCH_SIZE = 128
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
        self.classifier = nn.Linear(768, 100).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.seen_classes = set()
        self.task_magnitudes = {}
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=1,
            target_modules=["qkv"],
            use_dora=True,
            lora_dropout=LORA_DROPOUT,
            bias="none"
        )
        self.lora_model = get_peft_model(base_model, lora_config)
        
    def save_magnitudes(self, task_id):
        magnitudes = {}
        for name, module in self.lora_model.named_modules():
            if isinstance(module, DoraLinearLayer):
                magnitudes[name] = module.weight.data.clone()
        self.task_magnitudes[task_id] = magnitudes
        
    def load_magnitudes(self, task_id):
        if task_id in self.task_magnitudes:
            for name, module in self.lora_model.named_modules():
                if isinstance(module, DoraLinearLayer):
                    module.weight.data = self.task_magnitudes[task_id][name].clone()
    
    def train_task(self, train_loader, task_id, current_classes):
        self.seen_classes.update(current_classes)
        self.lora_model.train()
        total_loss = 0
        
        if task_id == 0:
            trainable_params = [
                {'params': self.lora_model.parameters()},
                {'params': self.classifier.parameters()},
            ]
        else:
            # Freeze LoRA, only train magnitude
            for n, p in self.lora_model.named_parameters():
                if 'lora_' in n:  
                    p.requires_grad = False
                elif '.weight' in n:  
                    p.requires_grad = True
            trainable_params = [
                {'params': [p for n, p in self.lora_model.named_parameters() if '.weight' in n]},
                {'params': self.classifier.parameters()}
            ]
            
        optimizer = torch.optim.Adam(trainable_params, lr=0.001, weight_decay=0.9)
        
        # Get indices for classes not in current task
        all_classes = set(range(100))
        not_current_classes = list(all_classes - set(current_classes))
        mask_tensor = torch.tensor(not_current_classes, dtype=torch.int64).to(DEVICE)
        
        for epoch in tqdm(range(EPOCHS), desc=f'Task {task_id} Training'):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                features = self.lora_model(inputs)
                logits = self.classifier(features)
                
                # Mask logits with -inf for non-current classes
                masked_logits = logits.index_fill(dim=1, index=mask_tensor, value=float('-inf'))
                
                loss = self.criterion(masked_logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                _, predicted = masked_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_acc = 100. * correct / total
            logging.info(f'Task {task_id}, Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, Acc = {train_acc:.2f}%')
            total_loss += avg_epoch_loss
            
        # Save magnitudes after training
        self.save_magnitudes(task_id)
        return total_loss / EPOCHS

    def evaluate(self, test_loader):
        self.lora_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets,_ in test_loader:
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
    seed=42
    logging.info("Starting S-LoRA CIFAR-100 experiment")
    
    base_model = create_model()
    scenario = create_benchmark()
    slora = SLoRA(base_model, NUM_TASKS)
    
    for task_id, experience in enumerate(scenario.train_stream):
        logging.info(f"\nStarting task {task_id}")
        current_classes = experience.classes_in_this_experience
        
        train_loader = DataLoader(experience.dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train task
        avg_loss = slora.train_task(train_loader, task_id, current_classes)
        logging.info(f"Task {task_id} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluate on all seen tasks
        for eval_task_id in range(task_id + 1):
            # Load task-specific magnitudes for evaluation
            slora.load_magnitudes(eval_task_id)
            eval_loader = DataLoader(scenario.test_stream[eval_task_id].dataset, batch_size=BATCH_SIZE)
            accuracy = slora.evaluate(eval_loader)
            logging.info(f"Accuracy on task {eval_task_id}: {accuracy:.2f}%")



if __name__ == "__main__":
    logging.info("""
    Summary of S-DoRA CIFAR-100 Benchmark

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




    