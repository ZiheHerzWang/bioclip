import argparse
import os
import sys
import json
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from bioclip import load_bioclip_to_cpu, CustomCLIP

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

        # Apply preprocessing based on mode
        image = self.preprocess_func(image)
        return image, label

def load_classnames(classname_file):
    """Load classnames from the given file (classname,label)."""
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

def evaluate(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass (no backpropagation)
            logits = model(images)  

            # Compute loss (optional for tracking during evaluation)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            # Make predictions (find the index of the max logit for each sample)
            predicted_labels = torch.argmax(logits, dim=1)

            # Compare predictions with true labels and accumulate correct predictions
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    # Compute accuracy
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)

    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation of Pretrained BioCLIP Model")
    parser.add_argument('--json_file', type=str, required=True, help="路径到数据集的 JSON 文件")
    parser.add_argument('--classname_file', type=str, required=True, help="路径到 classname.txt 文件")
    parser.add_argument('--batch_size', type=int, default=32, help="DataLoader 的批大小")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="优化器的学习率")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help="选择优化器: adam, adamw, 或 sgd")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bioclip_model, preprocess_train, preprocess_val = load_bioclip_to_cpu()
    bioclip_model.visual.to(device)
    bioclip_model.to(device)
    classnames = load_classnames(args.classname_file)

    model = CustomCLIP(classnames, bioclip_model)
    model = model.to(device)

    preprocess_func = preprocess_val

    dataset = CustomDataset(args.json_file, classnames, preprocess_func)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 进行评估
    avg_loss, accuracy = evaluate(model, data_loader, device)
    print(f"Evaluation Complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()