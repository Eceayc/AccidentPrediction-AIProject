import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
MAX_FRAMES = 30  # pad/truncate scenario to this many frames
MAX_OBJECTS = 10  # pad/truncate each frame to this many objects
FEATURE_DIM = 10  # [class, x, y, z, l, w, h, yaw, vx, vy]

# --- OBJECT CLASS ENCODING ---
OBJECT_CLASSES = ['car', 'van', 'truck', 'motorcycle', 'pedestrian']
CLASS2IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}

# --- VISUALIZATION FUNCTIONS ---
def create_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'training_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_loss_curve(train_losses, val_losses, results_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, results_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

def save_metrics(metrics, results_dir):
    metrics_file = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

# --- DATASET ---
def parse_object_line(line):
    parts = line.strip().split()
    if len(parts) < 11:
        return None
    obj_class = parts[0]
    if obj_class not in CLASS2IDX:
        return None
    class_idx = CLASS2IDX[obj_class]
    x, y, z = map(float, parts[1:4])
    l, w, h = map(float, parts[4:7])
    yaw = float(parts[7])
    vx, vy = map(float, parts[8:10])
    # ignore id, tracking, is_accident_vehicle
    return [class_idx, x, y, z, l, w, h, yaw, vx, vy]

def load_scenario(scenario_path):
    frame_files = sorted(glob.glob(os.path.join(scenario_path, '*.txt')))
    frames = []
    for f in frame_files:
        with open(f, 'r') as file:
            objs = []
            for line in file:
                feat = parse_object_line(line)
                if feat is not None:
                    objs.append(feat)
            # Pad/truncate objects
            if len(objs) < MAX_OBJECTS:
                objs += [[0]*FEATURE_DIM]*(MAX_OBJECTS - len(objs))
            else:
                objs = objs[:MAX_OBJECTS]
            frames.append(objs)
    # Pad/truncate frames
    if len(frames) < MAX_FRAMES:
        frames += [[[0]*FEATURE_DIM]*MAX_OBJECTS]*(MAX_FRAMES - len(frames))
    else:
        frames = frames[:MAX_FRAMES]
    return np.array(frames, dtype=np.float32)  # [frames, objects, features]

def get_label_from_path(path):
    if 'accident' in path:
        return 1
    return 0

class ScenarioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder, 'ego_vehicle', 'label')
            if not os.path.exists(folder_path):
                continue
            for scenario in os.listdir(folder_path):
                scenario_path = os.path.join(folder_path, scenario)
                if os.path.isdir(scenario_path):
                    label = get_label_from_path(folder)
                    self.samples.append((scenario_path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        scenario_path, label = self.samples[idx]
        data = load_scenario(scenario_path)  # [frames, objects, features]
        return torch.tensor(data), torch.tensor(label, dtype=torch.float32)

# --- MODEL ---
class ScenarioTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2, num_classes=1):
        super().__init__()
        self.input_fc = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, x):
        x = self.input_fc(x)
        x = x.mean(2)  
        x = self.transformer(x)
        x = x.mean(1)
        out = self.classifier(x).squeeze(-1)
        return out

# --- TRAINING & EVAL ---
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            pred = torch.sigmoid(logits).cpu().numpy() > 0.5
            preds.extend(pred.astype(int).tolist())
            trues.extend(y.numpy().astype(int).tolist())
    
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    val_loss = np.mean(losses)
    
    return acc, prec, rec, f1, val_loss, preds, trues

if __name__ == '__main__':
    # Create results directory
    results_dir = create_results_dir()
    
    # Load datasets
    train_dataset = ScenarioDataset('train')
    val_dataset = ScenarioDataset('val')
    test_dataset = ScenarioDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = ScenarioTransformer(FEATURE_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Track training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5  # Early stopping patience
    patience_counter = 0

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds, val_trues = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                pred = torch.sigmoid(logits).cpu().numpy() > 0.5
                val_preds.extend(pred.astype(int).tolist())
                val_trues.extend(y.numpy().astype(int).tolist())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_acc = accuracy_score(val_trues, val_preds)
        val_prec = precision_score(val_trues, val_preds, zero_division=0)
        val_rec = recall_score(val_trues, val_preds, zero_division=0)
        val_f1 = f1_score(val_trues, val_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.3f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print('Training complete.')
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(results_dir, 'best_model.pth'))
    
    # Final evaluation on test set
    model.eval()
    test_preds, test_trues = [], []
    test_loss = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()
            
            pred = torch.sigmoid(logits).cpu().numpy() > 0.5
            test_preds.extend(pred.astype(int).tolist())
            test_trues.extend(y.numpy().astype(int).tolist())
    
    test_loss /= len(test_loader)
    
    # Calculate final test metrics
    test_acc = accuracy_score(test_trues, test_preds)
    test_prec = precision_score(test_trues, test_preds, zero_division=0)
    test_rec = recall_score(test_trues, test_preds, zero_division=0)
    test_f1 = f1_score(test_trues, test_preds, zero_division=0)
    
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics - Acc: {test_acc:.3f} | Prec: {test_prec:.3f} | Rec: {test_rec:.3f} | F1: {test_f1:.3f}")

    # Generate and save plots
    plot_loss_curve(train_losses, val_losses, results_dir)
    plot_confusion_matrix(test_trues, test_preds, results_dir)
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1
        }
    }
    save_metrics(metrics, results_dir)
    
    print(f"\nTraining results saved to {results_dir}")
