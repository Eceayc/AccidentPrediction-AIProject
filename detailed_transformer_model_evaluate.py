import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from model_detailed import DetailedScenarioTransformer, ScenarioDataset, DEVICE, FEATURE_DIM, MAX_OBJECTS, OBJECT_CLASSES
from torch.utils.data import DataLoader
import json
from datetime import datetime
import glob

def create_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def plot_confusion_matrix(y_true, y_pred, results_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

def plot_accident_scenario(scenario_data, predictions, results_dir, scenario_idx):
    # Extract predictions
    accident_prob = torch.sigmoid(predictions['accident_logits']).item()
    time_pred = predictions['time_pred'].item()
    location_pred = predictions['location_pred'].cpu().numpy()
    involved_probs = torch.sigmoid(predictions['involved_logits']).cpu().numpy()
    
    # Get the last frame of the scenario
    last_frame = scenario_data[-1]  # [objects, features]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Vehicle positions and predicted accident location plotting
    for obj_idx in range(MAX_OBJECTS):
        if np.any(last_frame[obj_idx]):  # if object exists
            x, y = last_frame[obj_idx][1], last_frame[obj_idx][2]  # x, y coordinates
            involved_prob = involved_probs[obj_idx]
            color = 'red' if involved_prob > 0.5 else 'blue'
            alpha = involved_prob if involved_prob > 0.5 else 0.3
            ax1.scatter(x, y, c=color, alpha=alpha, s=100)
            ax1.text(x, y, f'V{obj_idx}', fontsize=8)
    
    # predicted accident location plot
    ax1.scatter(location_pred[0], location_pred[1], c='red', marker='*', s=200, label='Predicted Accident Location')
    ax1.set_title('Vehicle Positions and Predicted Accident Location')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend()
    ax1.grid(True)
    
    # vehicle involvement probabilities plot
    involved_vehicles = [f'V{i}' for i in range(MAX_OBJECTS) if np.any(last_frame[i])]
    involved_probs_filtered = [involved_probs[i] for i in range(MAX_OBJECTS) if np.any(last_frame[i])]
    
    ax2.bar(involved_vehicles, involved_probs_filtered)
    ax2.set_title('Vehicle Involvement Probabilities')
    ax2.set_xlabel('Vehicle ID')
    ax2.set_ylabel('Involvement Probability')
    ax2.set_ylim(0, 1)
    
    # scenario information
    info_text = f'Scenario {scenario_idx}\n'
    info_text += f'Accident Probability: {accident_prob:.2f}\n'
    info_text += f'Time to Accident: {time_pred:.2f}s\n'
    info_text += f'Predicted Location: ({location_pred[0]:.2f}, {location_pred[1]:.2f})'
    
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'scenario_{scenario_idx}_prediction.png'))
    plt.close()

def get_ground_truth_from_last_frame(scenario_path):
    """Extract ground truth from the last frame of a scenario."""
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

def calculate_prediction_errors(predictions_list, results_dir):
    """Calculate and save error metrics for predictions."""
    errors = {
        'location_errors': [],
        'involved_vehicle_errors': [],
        'time_errors': [],
        'scenario_details': []
    }
    
    for pred in predictions_list:
        if pred['accident_probability'] > 0.5:  # Only evaluate high-probability predictions
            scenario_path = pred['scenario_path']
            ground_truth = get_ground_truth_from_last_frame(scenario_path)
            
            if ground_truth and ground_truth['accident_location']:
                # Location error (Euclidean distance)
                pred_loc = np.array(pred['predicted_location'])
                true_loc = np.array(ground_truth['accident_location'])
                location_error = np.linalg.norm(pred_loc - true_loc)
                errors['location_errors'].append(location_error)
                
                # Vehicle involvement error (F1 score)
                pred_involved = (np.array(pred['involved_vehicles']) > 0.5).astype(int)
                true_involved = np.array(ground_truth['involved_vehicles'])
                involved_error = 1 - f1_score(true_involved, pred_involved, zero_division=0)
                errors['involved_vehicle_errors'].append(involved_error)
                
                # Time error (absolute difference)
                time_error = abs(pred['time_to_accident'] - ground_truth['time_to_accident'])
                errors['time_errors'].append(time_error)
                
                # Store detailed errors for this scenario
                errors['scenario_details'].append({
                    'scenario_id': pred['scenario_id'],
                    'location_error': float(location_error),
                    'involved_vehicle_error': float(involved_error),
                    'time_error': float(time_error),
                    'predicted_location': pred['predicted_location'],
                    'true_location': ground_truth['accident_location'],
                    'predicted_time': pred['time_to_accident'],
                    'true_time': ground_truth['time_to_accident'],
                    'predicted_involved': pred_involved.tolist(),
                    'true_involved': true_involved.tolist()
                })
    
    # Calculate and save error statistics
    error_stats = {
        'mean_location_error': float(np.mean(errors['location_errors'])) if errors['location_errors'] else 0.0,
        'std_location_error': float(np.std(errors['location_errors'])) if errors['location_errors'] else 0.0,
        'mean_involved_error': float(np.mean(errors['involved_vehicle_errors'])) if errors['involved_vehicle_errors'] else 0.0,
        'std_involved_error': float(np.std(errors['involved_vehicle_errors'])) if errors['involved_vehicle_errors'] else 0.0,
        'mean_time_error': float(np.mean(errors['time_errors'])) if errors['time_errors'] else 0.0,
        'std_time_error': float(np.std(errors['time_errors'])) if errors['time_errors'] else 0.0
    }
    
    # Save error statistics
    with open(os.path.join(results_dir, 'error_metrics.json'), 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    # Save detailed errors
    with open(os.path.join(results_dir, 'detailed_errors.json'), 'w') as f:
        json.dump(errors['scenario_details'], f, indent=2)
    
    # Plot error distributions
    if errors['location_errors'] or errors['involved_vehicle_errors'] or errors['time_errors']:
        plt.figure(figsize=(15, 5))
        
        if errors['location_errors']:
            plt.subplot(1, 3, 1)
            plt.hist(errors['location_errors'], bins=20)
            plt.title('Location Prediction Errors')
            plt.xlabel('Euclidean Distance')
            plt.ylabel('Count')
        
        if errors['involved_vehicle_errors']:
            plt.subplot(1, 3, 2)
            plt.hist(errors['involved_vehicle_errors'], bins=20)
            plt.title('Vehicle Involvement Prediction Errors')
            plt.xlabel('1 - F1 Score')
            plt.ylabel('Count')
        
        if errors['time_errors']:
            plt.subplot(1, 3, 3)
            plt.hist(errors['time_errors'], bins=20)
            plt.title('Time Prediction Errors')
            plt.xlabel('Absolute Time Difference (s)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'error_distributions.png'))
        plt.close()
    
    return error_stats

def save_predictions(predictions_list, results_dir):
    # Save detailed predictions
    predictions_file = os.path.join(results_dir, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(predictions_list, f, indent=2)
    
    # Save summary predictions with error metrics
    summary_file = os.path.join(results_dir, 'predictions_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Detailed Accident Predictions Summary\n")
        f.write("===================================\n\n")
        
        for pred in predictions_list:
            if pred['accident_probability'] > 0.5:  # Only show high-probability predictions
                f.write(f"Scenario: {pred['scenario_id']}\n")
                f.write(f"Accident Probability: {pred['accident_probability']:.2f}\n")
                f.write(f"Time to Accident: {pred['time_to_accident']:.2f} seconds\n")
                f.write(f"Predicted Location: ({pred['predicted_location'][0]:.2f}, {pred['predicted_location'][1]:.2f})\n")
                
                # List involved vehicles
                involved_vehicles = [f"Vehicle {i} ({prob:.2f})" 
                                  for i, prob in enumerate(pred['involved_vehicles']) 
                                  if prob > 0.5]
                f.write("Involved Vehicles:\n")
                for vehicle in involved_vehicles:
                    f.write(f"  - {vehicle}\n")
                f.write("\n" + "-"*50 + "\n\n")

def evaluate_model(model_path, test_loader, results_dir):
    # Load model
    model = DetailedScenarioTransformer(FEATURE_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize lists for metrics
    all_accident_probs = []
    all_true_labels = []
    all_predictions = []
    
    # Store detailed predictions
    predictions_list = []
    
    with torch.no_grad():
        for batch_idx, (x, y, ground_truth) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get model predictions
            accident_logits, time_pred, location_pred, involved_logits = model(x)
            accident_probs = torch.sigmoid(accident_logits)
            
            # Store predictions for each scenario in the batch
            for i in range(len(x)):
                # Get the actual scenario path from the dataset
                sample_idx = batch_idx * test_loader.batch_size + i
                if sample_idx < len(test_loader.dataset.samples):
                    scenario_path = test_loader.dataset.samples[sample_idx][0]
                else:
                    scenario_path = f'batch_{batch_idx}_scenario_{i}'
                
                scenario_pred = {
                    'scenario_id': f'batch_{batch_idx}_scenario_{i}',
                    'scenario_path': scenario_path,
                    'accident_probability': accident_probs[i].item(),
                    'time_to_accident': time_pred[i].item(),
                    'predicted_location': location_pred[i].cpu().numpy().tolist(),
                    'involved_vehicles': torch.sigmoid(involved_logits[i]).cpu().numpy().tolist(),
                    'true_label': y[i].item()
                }
                predictions_list.append(scenario_pred)
                
                # Plot accident scenario for all
                plot_accident_scenario(
                    x[i].cpu().numpy(),
                    {
                        'accident_logits': accident_logits[i],
                        'time_pred': time_pred[i],
                        'location_pred': location_pred[i],
                        'involved_logits': involved_logits[i]
                    },
                    results_dir,
                    f'batch_{batch_idx}_scenario_{i}'
                )
            
            # Store metrics
            all_accident_probs.extend(accident_probs.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            all_predictions.extend((accident_probs > 0.5).cpu().numpy())
    
    # Generate plots and save predictions
    plot_confusion_matrix(all_true_labels, all_predictions, results_dir)
    save_predictions(predictions_list, results_dir)
    
    # Calculate and save error metrics
    error_stats = calculate_prediction_errors(predictions_list, results_dir)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    # Print error statistics
    print("\nError Statistics:")
    print(f"Mean Location Error: {error_stats['mean_location_error']:.2f} ± {error_stats['std_location_error']:.2f}")
    print(f"Mean Vehicle Involvement Error: {error_stats['mean_involved_error']:.2f} ± {error_stats['std_involved_error']:.2f}")
    print(f"Mean Time Error: {error_stats['mean_time_error']:.2f} ± {error_stats['std_time_error']:.2f}")
    
    # Print summary
    print(f"\nEvaluation Results saved to {results_dir}")
    print(f"Total scenarios evaluated: {len(predictions_list)}")
    print(f"Number of high-probability accident predictions: {sum(p['accident_probability'] > 0.5 for p in predictions_list)}")

if __name__ == '__main__':
    # Create results directory
    results_dir = create_results_dir()
    
    # Load test data
    test_dataset = ScenarioDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluate model
    model_path = 'model_detailed.pth'
    evaluate_model(model_path, test_loader, results_dir) 