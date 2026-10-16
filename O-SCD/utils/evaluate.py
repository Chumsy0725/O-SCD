import os
import cv2
import argparse
from tqdm import tqdm
import torch
import torchmetrics
from typing import Tuple, Optional

# Initialize TorchMetrics for mIoU and F1
miou_metric = torchmetrics.JaccardIndex(task='binary')
f1_metric = torchmetrics.F1Score(task='binary')


def calculate_metrics_torch(ground_truth: torch.Tensor, predicted: torch.Tensor) -> Tuple[float, float]:
    """Calculate mIoU and F1 score using TorchMetrics."""
    ground_truth_tensor = torch.tensor(ground_truth).unsqueeze(0)
    predicted_tensor = torch.tensor(predicted).unsqueeze(0)

    # mIoU
    miou = miou_metric(predicted_tensor, ground_truth_tensor).item()

    # F1 Score
    f1 = f1_metric(predicted_tensor, ground_truth_tensor).item()

    return miou, f1

def find_matching_file(gt_file_name: str, predicted_dir: str) -> Optional[str]:
    """Find a matching predicted file by name, ignoring the extension."""
    gt_base_name, _ = os.path.splitext(gt_file_name)
    for pred_file in os.listdir(predicted_dir):
        pred_base_name, _ = os.path.splitext(pred_file)
        if gt_base_name == pred_base_name:
            return os.path.join(predicted_dir, pred_file)
    return None

def evaluate_segmentation(ground_truth_dir: str, predicted_binary_dir: str) -> Tuple[float, float]:
    """Evaluate segmentation with binary masks."""
    miou_scores = []
    f1_scores = []

    gt_files = os.listdir(ground_truth_dir)

    for gt_file_name in tqdm(gt_files, desc="Evaluating segmentation masks", unit="mask"):
        ground_truth_path = os.path.join(ground_truth_dir, gt_file_name)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        
        _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        ground_truth = ground_truth // 255

        predicted_binary_path = find_matching_file(gt_file_name, predicted_binary_dir)

        if predicted_binary_path:
            predicted_binary = cv2.imread(predicted_binary_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth.shape != predicted_binary.shape:
                predicted_binary = cv2.resize(predicted_binary, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, predicted_binary = cv2.threshold(predicted_binary, 127, 255, cv2.THRESH_BINARY)
            predicted_binary = predicted_binary // 255

            miou, f1 = calculate_metrics_torch(ground_truth, predicted_binary)
            miou_scores.append(miou)
            f1_scores.append(f1)

    mean_miou = 0.0
    mean_f1 = 0.0
    
    if miou_scores:
        mean_miou = torch.tensor(miou_scores).mean().item()
        mean_f1 = torch.tensor(f1_scores).mean().item()

        print(f"Mean IoU: {mean_miou}")
        print(f"Mean F1: {mean_f1}")

    return mean_miou, mean_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results using mIoU and F1")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth masks directory")
    parser.add_argument("--pred_binary", type=str, required=True, help="Path to the binary predicted masks directory")

    args = parser.parse_args()

    evaluate_segmentation(args.gt, args.pred_binary)

if __name__ == "__main__":
    main()
