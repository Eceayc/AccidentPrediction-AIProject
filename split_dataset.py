import os
import shutil
import random

# Configuration
DATASET_FOLDERS = [
    'type1_subtype1_normal',
    'type1_subtype1_accident',
    'type1_subtype2_normal',
    'type1_subtype2_accident',
]
LABEL_PATH = os.path.join('ego_vehicle', 'label')
TRAIN_DIR = 'train'
TEST_DIR = 'test'
SPLIT_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Create train/test directories
for split in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(split):
        os.mkdir(split)
    for folder in DATASET_FOLDERS:
        split_folder = os.path.join(split, folder)
        if not os.path.exists(split_folder):
            os.makedirs(split_folder)

def split_and_copy(folder):
    label_dir = os.path.join(folder, LABEL_PATH)
    if not os.path.exists(label_dir):
        print(f"Label dir not found: {label_dir}")
        return
    scenarios = [d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))]
    random.shuffle(scenarios)
    split_idx = int(len(scenarios) * SPLIT_RATIO)
    train_scenarios = scenarios[:split_idx]
    test_scenarios = scenarios[split_idx:]
    # Copy scenarios
    for split, scenario_list in zip([TRAIN_DIR, TEST_DIR], [train_scenarios, test_scenarios]):
        for scenario in scenario_list:
            src = os.path.join(folder, LABEL_PATH, scenario)
            dst = os.path.join(split, folder, LABEL_PATH, scenario)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst)
            print(f"Copied {src} -> {dst}")

if __name__ == '__main__':
    for folder in DATASET_FOLDERS:
        split_and_copy(folder)
    print('Dataset split complete.') 