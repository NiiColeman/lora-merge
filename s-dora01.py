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
import torch.nn.functional as F
import sys




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
EPOCHS = 15
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

def create_benchmark():
    train_set, test_set =torchvision.datasets.CIFAR100(root='./data', train=False, download=False,transform=train_transform), torchvision.datasets.CIFAR100(root='./data', train=False, download=True,transform=test_transform)
    scenario=nc_benchmark(train_dataset=train_set, test_dataset=test_set, n_experiences=NUM_TASKS, shuffle=True, seed=23,task_labels=False) 

    return scenario


def create_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    return model


class SimpleDoRA(nn.Module):
    def __init__(self, base_model, num_tasks):
        super().__init__()
        self.base_model = base_model.to(DEVICE)
        self.num_tasks = num_tasks
        self.task_alphas = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_tasks)])
        
        # Replace Identity head with our classifier
        self.base_model.head = nn.Linear(768, 100).to(DEVICE)
        
        self.criterion = nn.CrossEntropyLoss()
        self.seen_classes = set()
        
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=1,
            target_modules=["qkv"],
            # use_dora=True,
            lora_dropout=LORA_DROPOUT,
            bias="none"
        )
        self.lora_model = get_peft_model(base_model, lora_config)
    
    def forward(self, x, task_id):
        if task_id == 0:
            return self.lora_model(x)
        else:
            base_output = self.base_model(x)
            dora_output= self.lora_model(x)
            scaled_delta = self.task_alphas[task_id] * (dora_output - base_output)
            dora_delta = self.lora_model(x) - base_output
            return base_output + scaled_delta
    
    def train_task(self, train_loader, task_id, current_classes):
        self.seen_classes.update(current_classes)
        print(f"\nStarting training for task {task_id}")
        
        # Ensure head is trainable
        for param in self.base_model.head.parameters():
            param.requires_grad = True
        
        if task_id == 0:
            trainable_params = []
            for n, p in self.lora_model.named_parameters():
                if 'lora_' in n:
                    trainable_params.append(p)
            optimizer = torch.optim.Adam([
                {'params': trainable_params},
                {'params': self.base_model.head.parameters(), 'lr': 0.001}  # Explicit head params
            ], lr=0.001, weight_decay=1e-3)
        else:
            # Freeze LoRA parameters
            for n, p in self.lora_model.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = False
            
            # Explicitly get head parameters
            head_params = list(self.base_model.head.parameters())
            optimizer = torch.optim.Adam([
                {'params': self.task_alphas[task_id]},
                {'params': head_params, 'lr': 0.001}  # Explicit learning rate
            ], lr=0.001)
        
        # Print which parameters are trainable
        print("\nTrainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")
        
        final_epoch_loss = 0
        for epoch in range(EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            
            self.train()
            for batch_idx, (inputs, targets,_), in enumerate(train_loader):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.forward(inputs, task_id)
                
                # Debug first batch
                if batch_idx == 0:
                    with torch.no_grad():
                        base_out = self.base_model(inputs)
                        dora_out = self.lora_model(inputs)
                        print(f"\nEpoch {epoch} diagnostics:")
                        print(f"Base output norm: {base_out.norm():.4f}")
                        print(f"DoRA output norm: {dora_out.norm():.4f}")
                        print(f"Difference norm: {(dora_out - base_out).norm():.4f}")
                        if task_id > 0:
                            print(f"Current alpha value: {self.task_alphas[task_id].item():.4f}")
                
                # Apply mask for seen classes
                mask = torch.zeros_like(outputs, device=DEVICE)
                mask[:, list(self.seen_classes)] = 1
                outputs = outputs * mask
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Debug gradients
                if batch_idx == 0:
                    if task_id > 0:
                        alpha_grad = self.task_alphas[task_id].grad
                        if alpha_grad is not None:
                            logging.info(f"Alpha gradient: {alpha_grad.item():.4f}")
                            print(f"Alpha gradient: {alpha_grad.item():.4f}")
                    
                    head_grad = self.base_model.head.weight.grad
                    if head_grad is not None:
                        print(f"Head gradient norm: {head_grad.norm():.4f}")
                        logging.info(f"Head gradient norm: {head_grad.norm():.4f}")
                    else:
                        print("Head gradient is None - STILL AN ISSUE!")
                        logging.info("Head gradient is None - STILL AN ISSUE!")
                
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                total_loss += loss.item()
            
            epoch_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            logging.info(f'Task {task_id}, Epoch {epoch}: Loss = {epoch_loss:.4f}, Acc = {accuracy:.2f}%')
            final_epoch_loss = epoch_loss
        
        return final_epoch_loss

    def evaluate(self, test_loader, task_id):
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets,_ in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = self.forward(inputs, task_id)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
        



def main():
# Initialize Progressive DoRA
    scenario = create_benchmark()
    base_model = create_model()
    model = SimpleDoRA(base_model, NUM_TASKS)
    model = model.to(DEVICE)
  
  
    
    # Training loop
    for task_id, experience in enumerate(scenario.train_stream):
        logging.info(f"\nStarting task {task_id}")
        
        # Get current task data
        train_loader = DataLoader(experience.dataset, batch_size=BATCH_SIZE, shuffle=True)
        current_classes = experience.classes_in_this_experience
        
        # Train on current task
        avg_loss = model.train_task(train_loader, task_id, current_classes)
        logging.info(f"Task {task_id} completed. Average loss: {avg_loss:.4f}")
        
        # Evaluate on all seen tasks
        for eval_task_id in range(task_id + 1):
            eval_loader = DataLoader(
                scenario.test_stream[eval_task_id].dataset,
                batch_size=BATCH_SIZE
            )
            accuracy = model.evaluate(eval_loader, task_id)
            logging.info(f"Accuracy on task {eval_task_id}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()