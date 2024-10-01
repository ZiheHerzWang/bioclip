import argparse
import os
import sys
import json
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from cocoop_bioclip import load_bioclip_to_cpu, CustomCLIP

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, classnames, preprocess_func, mode="train"):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.classnames = classnames
        self.preprocess_func = preprocess_func
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['img_path']
        #print(img_path)
        label = self.data[idx]['label']
        image = Image.open(img_path)

        image = self.preprocess_func(image)
        return image, label

def load_classnames(classname_file):
    classnames = []
    with open(classname_file, 'r') as f:
        for line in f:
            line = line.strip() 
            if line:
                classname, label = line.split(',')
                classnames.append(classname)
    
    print(classnames)
    return classnames
def print_gradients_stats(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm() 
            grad_mean = param.grad.mean()  
            grad_max = param.grad.max()    
            grad_min = param.grad.min()   
            print(f"Gradient stats for {name}: norm = {grad_norm}, mean = {grad_mean}, max = {grad_max}, min = {grad_min}")
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        #print_gradients_stats(model)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss}")
    return avg_loss

def evaluate(model, data_loader, device):
    model.eval()  #evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad(): 
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            # Compute loss
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # Make predictions
            predicted_labels = torch.argmax(logits, dim=1)

            # Compare predictions with true labels
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    # Compute accuracy
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)

    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Custom CLIP Model")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help="Mode: train or eval")
    parser.add_argument('--json_file', type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument('--classname_file', type=str, required=True, help="Path to the classname.txt file")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 penalty) for AdamW optimizer")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help="Optimizer to use: adam, adamw, or sgd")
    parser.add_argument('--checkpoint', type=str, help="Path to the checkpoint file for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the BioCLIP model and preprocessing
    bioclip_model, preprocess_train, preprocess_val = load_bioclip_to_cpu()
    bioclip_model.visual.to(device)

    # Load classnames
    classnames = load_classnames(args.classname_file)

    # Create CustomCLIP model
    model = CustomCLIP( classnames, bioclip_model)

    model = model.to(device)
    model.text_encoder.to(device)

    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name:
            param.requires_grad_(False)

    # Choose preprocessing function based on mode
    preprocess_func = preprocess_train if args.mode == 'train' else preprocess_val

    # Load dataset
    dataset = CustomDataset(args.json_file, classnames, preprocess_func, mode=args.mode)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Set up optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Training or evaluation loop
    if args.mode == 'train':
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            train_loss = train_one_epoch(model, data_loader, optimizer, device)
            print(f"Epoch {epoch+1} Training Loss: {train_loss}")
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")
    elif args.mode == 'eval':
        if args.checkpoint and os.path.isfile(args.checkpoint):
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print("Checkpoint file not provided or does not exist. Exiting evaluation.")
            return
        avg_loss, accuracy = evaluate(model, data_loader, device)
        print(f"Evaluation Complete - Loss: {avg_loss}, Accuracy: {accuracy * 100:.2f}%")
if __name__ == "__main__":
    main()
