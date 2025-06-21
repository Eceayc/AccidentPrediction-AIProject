import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
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
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
MAX_FRAMES = 30  # pad/truncate scenario to this many frames
MAX_OBJECTS = 10  # pad/truncate each frame to this many objects
FEATURE_DIM = 12  # [class, x, y, z, l, w, h, yaw, vx, vy, obj_id, tracking_id]

# --- OBJECT CLASS ENCODING ---
OBJECT_CLASSES = ['car', 'van', 'truck', 'motorcycle', 'pedestrian']
CLASS2IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}

# --- DATASET ---
def parse_object_line(line):
    parts = line.strip().split()
    if len(parts) < 12:
        return None
    obj_class = parts[0]
    if obj_class not in CLASS2IDX:
        return None
    class_idx = CLASS2IDX[obj_class]
    x, y, z = map(float, parts[1:4])
    l, w, h = map(float, parts[4:7])
    yaw = float(parts[7])
    vx, vy = map(float, parts[8:10])
    obj_id = int(parts[10]) if len(parts) > 10 else -1
    tracking_id = int(parts[11]) if len(parts) > 11 else 0
    return [class_idx, x, y, z, l, w, h, yaw, vx, vy, obj_id, tracking_id]

def load_scenario(scenario_path):
    frame_files = sorted(glob.glob(os.path.join(scenario_path, '*.txt')))
    frames = []
    timestamps = []
    
    for f in frame_files:
        with open(f, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:  # Need at least timestamp and one object
                continue
                
            # Parse timestamp (first line)
            timestamp_parts = lines[0].strip().split()
            if len(timestamp_parts) >= 2:
                timestamp = float(timestamp_parts[0])  # Use first timestamp value
                timestamps.append(timestamp)
            else:
                timestamps.append(0.0)
            
            # Parse objects
            objs = []
            for line in lines[1:]:
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
        timestamps += [0.0] * (MAX_FRAMES - len(timestamps))
    else:
        frames = frames[:MAX_FRAMES]
        timestamps = timestamps[:MAX_FRAMES]
    
    return np.array(frames, dtype=np.float32), np.array(timestamps, dtype=np.float32)

def get_ground_truth_from_last_frame(scenario_path):
    """Extract ground truth information from the last frame of a scenario."""
    frame_files = sorted(glob.glob(os.path.join(scenario_path, '*.txt')))
    if not frame_files:
        return None
    
    last_frame_file = frame_files[-1]
    involved_vehicles = []
    accident_location = None
    time_to_accident = 0
    
    with open(last_frame_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        
        # Get timestamp from last frame
        timestamp_parts = lines[0].strip().split()
        if len(timestamp_parts) >= 2:
            time_to_accident = float(timestamp_parts[0])
        
        # Parse objects and find involved vehicles
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 13:  # Need at least 13 parts including is_accident_vehicle
                continue
            
            # check if vehicle is involved in accident
            is_involved = parts[12].lower() == 'true' if len(parts) > 12 else False
            involved_vehicles.append(1 if is_involved else 0)
            
            # use the first involved vehicle's position as accident location
            if is_involved and accident_location is None:
                x, y = map(float, parts[1:3])
                accident_location = [x, y]
    
    # Pad/truncate involved_vehicles list to MAX_OBJECTS
    if len(involved_vehicles) < MAX_OBJECTS:
        involved_vehicles = involved_vehicles + [0] * (MAX_OBJECTS - len(involved_vehicles))
    else:
        involved_vehicles = involved_vehicles[:MAX_OBJECTS]
    
    return {
        'involved_vehicles': involved_vehicles,
        'accident_location': accident_location,
        'time_to_accident': time_to_accident
    }

def get_label_from_path(path):
    if 'accident' in path:
        return 1
    return 0

class ScenarioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.ground_truths = []
        
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder, 'ego_vehicle', 'label')
            if not os.path.exists(folder_path):
                continue
            for scenario in os.listdir(folder_path):
                scenario_path = os.path.join(folder_path, scenario)
                if os.path.isdir(scenario_path):
                    label = get_label_from_path(folder)
                    ground_truth = get_ground_truth_from_last_frame(scenario_path)
                    self.samples.append((scenario_path, label))
                    self.ground_truths.append(ground_truth)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        scenario_path, label = self.samples[idx]
        data, timestamps = load_scenario(scenario_path)  # [frames, objects, features]
        ground_truth = self.ground_truths[idx]
        
        return torch.tensor(data), torch.tensor(label, dtype=torch.float32), ground_truth

# --- MODEL ---
class DetailedScenarioTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Binary accident flag
        self.accident_head = nn.Linear(d_model, 1)
        # Time-to-accident prediction (regression)
        self.time_head = nn.Linear(d_model, 1)
        # Location prediction (2D coordinates)
        self.location_head = nn.Linear(d_model, 2)
        # Involved vehicle IDs (multi-label classification)
        self.involved_head = nn.Linear(d_model, MAX_OBJECTS)

    def forward(self, x):
        B, F, O, C = x.shape
        x = x.reshape(B, F * O, C)  # Reshape to (B, N, C) where N = F * O
        x = self.input_fc(x)   # [B, N, d_model]
        # process all object-timesteps
        x = self.transformer(x)  # [B, N, d_model]
        # pool the outputs of all object-timesteps.
        x = x.mean(dim=1)     # [B, d_model]

        accident_logits = self.accident_head(x).squeeze(-1)  # [B]
        time_pred = self.time_head(x).squeeze(-1)  # [B]
        location_pred = self.location_head(x)  # [B, 2]
        involved_logits = self.involved_head(x)  # [B, MAX_OBJECTS]
        return accident_logits, time_pred, location_pred, involved_logits

# --- TRAINING & EVAL ---
def train_epoch(model, loader, optimizer, criterion_acc, criterion_time, criterion_loc, criterion_involved):
    model.train()
    losses = []
    for x, y, ground_truth in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        accident_logits, time_pred, location_pred, involved_logits = model(x)
        
        # Binary accident loss
        loss_acc = criterion_acc(accident_logits, y)
        
        # Time-to-accident loss (only for accident scenarios)
        time_targets = torch.zeros_like(time_pred)
        for i, gt in enumerate(ground_truth):
            if gt and gt['time_to_accident']:
                time_targets[i] = gt['time_to_accident']
        loss_time = criterion_time(time_pred, time_targets)
        
        # Location loss (only for accident scenarios)
        location_targets = torch.zeros_like(location_pred)
        for i, gt in enumerate(ground_truth):
            if gt and gt['accident_location']:
                location_targets[i] = torch.tensor(gt['accident_location'], dtype=torch.float32)
        loss_loc = criterion_loc(location_pred, location_targets)
        
        # Involved vehicles loss (only for accident scenarios)
        involved_targets = torch.zeros_like(involved_logits)
        for i, gt in enumerate(ground_truth):
            if gt and gt['involved_vehicles']:
                involved_targets[i] = torch.tensor(gt['involved_vehicles'], dtype=torch.float32)
        loss_involved = criterion_involved(involved_logits, involved_targets)
        
        # Combined loss (accident loss weighted more heavily)
        loss = loss_acc + 0.1 * loss_time + 0.1 * loss_loc + 0.1 * loss_involved
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE)
            accident_logits, time_pred, location_pred, involved_logits = model(x)
            pred = torch.sigmoid(accident_logits).cpu().numpy() > 0.5
            preds.extend(pred.astype(int).tolist())
            trues.extend(y.numpy().astype(int).tolist())
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    return acc, prec, rec, f1

def custom_collate_fn(batch):
    # batch is a list of (data, label, ground_truth)
    data = torch.stack([item[0] for item in batch])
    label = torch.stack([item[1] for item in batch])
    ground_truth = [item[2] for item in batch]  # keep as list of dicts
    return data, label, ground_truth

if __name__ == '__main__':
    # --- Fit scaler on all train data ---
    temp_samples = []
    for folder in os.listdir('train'):
        folder_path = os.path.join('train', folder, 'ego_vehicle', 'label')
        if not os.path.exists(folder_path):
            continue
        for scenario in os.listdir(folder_path):
            scenario_path = os.path.join(folder_path, scenario)
            if os.path.isdir(scenario_path):
                frame_files = sorted(glob.glob(os.path.join(scenario_path, '*.txt')))
                for f in frame_files:
                    with open(f, 'r') as file:
                        lines = file.readlines()
                        for line in lines[1:]:  # Skip timestamp line
                            parts = line.strip().split()
                            if len(parts) < 12:
                                continue
                            obj_class = parts[0]
                            if obj_class not in CLASS2IDX:
                                continue
                            x, y, z = map(float, parts[1:4])
                            l, w, h = map(float, parts[4:7])
                            yaw = float(parts[7])
                            vx, vy = map(float, parts[8:10])
                            temp_samples.append([x, y, z, l, w, h, yaw, vx, vy])
    scaler = StandardScaler()
    if temp_samples:
        scaler.fit(temp_samples)
    else:
        scaler = None

    train_dataset = ScenarioDataset('train')
    test_dataset = ScenarioDataset('test')
    val_dataset = ScenarioDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    # Calculate pos_weight for accident class
    num_accident = sum(label == 1 for _, label in train_dataset.samples)
    num_normal = sum(label == 0 for _, label in train_dataset.samples)
    pos_weight = torch.tensor([num_normal / (num_accident + 1e-6) * 2]).to(DEVICE)
    print(f"Number of accident samples: {num_accident}")
    print(f"Number of normal samples: {num_normal}")
    print(f"pos_weight: {pos_weight}")

    model = DetailedScenarioTransformer(FEATURE_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_acc = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_time = nn.MSELoss()
    criterion_loc = nn.MSELoss()
    criterion_involved = nn.BCEWithLogitsLoss()

    # Track training history
    train_losses = []
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion_acc, criterion_time, criterion_loc, criterion_involved)
        train_losses.append(train_loss)
        acc, prec, rec, f1 = eval_model(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

        # Print predicted accident probabilities for the first batch of the test set
        model.eval()
        with torch.no_grad():
            for x, y, gt in test_loader:
                accident_logits, _, _, _ = model(x.to(DEVICE))
                accident_probs = torch.sigmoid(accident_logits).cpu().numpy()
                print(f"Sample accident probabilities (first test batch): {accident_probs}")
                print(f"True labels (first test batch): {y.numpy()}")
                break

        # Save best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'model_detailed.pth')
            print(f"Saved new best model with F1 score: {f1:.3f}")

    print('Training complete.')

    # Save training history
    history = {
        'train_losses': train_losses,
        'best_f1': best_f1
    }
    torch.save(history, 'training_history.pt')

    torch.save(model.state_dict(), 'model_detailed.pth')
    print('Model saved at end of training (regardless of F1).') 