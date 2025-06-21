import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Data, Batch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
BATCH_SIZE = 8
EPOCHS = 20
LR = 2e-4  # learning rate
MAX_FRAMES = 30
MAX_OBJECTS = 10
FEATURE_DIM = 10
GNN_HIDDEN_DIM = 256
TRANSFORMER_DIM = 512
PREDICTION_THRESHOLD = 0.5

# Create output directory
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join(OUTPUT_DIR, f'run_{TIMESTAMP}')
os.makedirs(RUN_DIR, exist_ok=True)

# --- OBJECT CLASS ENCODING ---
OBJECT_CLASSES = ['car', 'van', 'truck', 'motorcycle', 'pedestrian']
CLASS2IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}

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
    return [class_idx, x, y, z, l, w, h, yaw, vx, vy]

def create_graph_from_frame(objects):
    # Create node features
    x = torch.tensor(objects, dtype=torch.float32)
    
    # Create fully connected graph
    num_nodes = len(objects)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)

def custom_collate(batch):
    frames_list, labels = zip(*batch)
    # Stack all frames from all samples in the batch
    all_frames = []
    for frames in frames_list:
        all_frames.extend(frames)
    # Create a batch of all frames
    batch_frames = Batch.from_data_list(all_frames)
    return batch_frames, torch.stack(labels)

class FrameGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        # GAT layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 4)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, dropout=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 4)
        
        # Skip connection projection
        self.skip_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 4),
            nn.LayerNorm(hidden_channels * 4)
        )
        
        # Attention pooling
        self.att_pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels * 4, hidden_channels * 2),
                nn.LayerNorm(hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels * 2, 1)
            )
        )
        
    def forward(self, x, edge_index, batch):
        identity = x
        x = self.input_norm(x)
        
        # First GAT layer
        x1 = self.gat1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        
        # Second GAT layer with skip connection
        x2 = self.gat2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        
        # Add skip connection
        x = x2 + self.skip_proj(identity)
        
        # Global attention pooling
        x = self.att_pool(x, batch)
        return x

class ScenarioTransformer(nn.Module):
    def __init__(self, gnn_hidden_dim, d_model, nhead=8, num_layers=3):
        super().__init__()
        self.frame_gnn = FrameGNN(FEATURE_DIM, gnn_hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_FRAMES, d_model))
        nn.init.normal_(self.pos_encoder, std=0.02)
        
        # Input projection
        self.input_fc = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, batch_data):
        x = self.frame_gnn(batch_data.x, batch_data.edge_index, batch_data.batch)
        batch_size = batch_data.num_graphs // MAX_FRAMES
        x = x.view(batch_size, MAX_FRAMES, -1)
        
        x = self.input_fc(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.classifier(x).squeeze(-1)
        return out

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
                    label = 1 if 'accident' in folder else 0
                    self.samples.append((scenario_path, label))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        scenario_path, label = self.samples[idx]
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
                frames.append(create_graph_from_frame(objs))
        
        # Pad/truncate frames
        if len(frames) < MAX_FRAMES:
            empty_frame = create_graph_from_frame([[0]*FEATURE_DIM]*MAX_OBJECTS)
            frames += [empty_frame]*(MAX_FRAMES - len(frames))
        else:
            frames = frames[:MAX_FRAMES]
            
        return frames, torch.tensor(label, dtype=torch.float32)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Accident'],
                yticklabels=['Normal', 'Accident'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses, metrics, save_path):
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(metrics['accuracy'], label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(2, 2, 3)
    plt.plot(metrics['precision'], label='Precision')
    plt.plot(metrics['recall'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(metrics['f1'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path):
    with open(save_path, 'w') as f:
        f.write('Epoch,Accuracy,Precision,Recall,F1\n')
        for i in range(len(metrics['accuracy'])):
            f.write(f"{i+1},{metrics['accuracy'][i]:.4f},{metrics['precision'][i]:.4f},{metrics['recall'][i]:.4f},{metrics['f1'][i]:.4f}\n")

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for batch_frames, y in loader:
        batch_frames = batch_frames.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch_frames)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_frames, y in loader:
            batch_frames = batch_frames.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(batch_frames)
            # Use higher threshold for accident prediction
            pred = torch.sigmoid(logits).cpu().numpy() > PREDICTION_THRESHOLD
            preds.extend(pred.astype(int).tolist())
            trues.extend(y.cpu().numpy().astype(int).tolist())
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    return acc, prec, rec, f1, preds, trues

# Add learning rate scheduler
def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

if __name__ == '__main__':
    train_dataset = ScenarioDataset('train')
    test_dataset = ScenarioDataset('test')
    
    # Calculate balanced class weights with smoothing
    labels = [label for _, label in train_dataset.samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([(neg_count + 1) / (pos_count + 1)]).to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    model = ScenarioTransformer(GNN_HIDDEN_DIM, TRANSFORMER_DIM).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize metrics tracking
    train_losses = []
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    best_acc = 0
    best_f1 = 0
    patience = 15
    no_improve = 0
    target_accuracy = 0.85  # Target accuracy threshold

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        acc, prec, rec, f1, preds, trues = eval_model(model, test_loader)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")
        
        # Save best model based on accuracy
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': acc,
                'f1': f1,
            }, os.path.join(RUN_DIR, 'best_model.pth'))
            print(f"New best model saved with accuracy: {acc:.3f}")
        else:
            no_improve += 1
            
        if acc >= target_accuracy:
            print(f"\nTarget accuracy of {target_accuracy:.2%} reached! Stopping training.")
            break
            
        # Early stopping if no improvement
        if no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best accuracy achieved: {best_acc:.3f}")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': acc,
        'f1': f1,
    }, os.path.join(RUN_DIR, 'final_model.pth'))

    # Plot and save visualizations
    plot_training_curves(train_losses, metrics, os.path.join(RUN_DIR, 'training_curves.png'))
    plot_confusion_matrix(trues, preds, os.path.join(RUN_DIR, 'confusion_matrix.png'))
    save_metrics(metrics, os.path.join(RUN_DIR, 'metrics.csv'))

    print('\nTraining complete. Results saved to:', RUN_DIR)
    print(f'Final accuracy: {acc:.3f}')
    print(f'Best accuracy: {best_acc:.3f}')
    print(f'Best F1 score: {best_f1:.3f}') 

