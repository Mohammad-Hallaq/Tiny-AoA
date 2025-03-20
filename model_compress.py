import argparse
import torch
import torch.nn as nn
from compression_utils.compression_functions import (
    perplexity_analysis_with_contributions,
    calculate_pruning_ratios_intense,
    selective_pruning
)
from data_utils.R22_dataset import R22_H5_Dataset
import logging
import numpy as np
import os

def main(args):
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load the original model
    original_model = torch.load(args.model_path, map_location=args.device)
    
    # Load test and training datasets
    train_set = R22_H5_Dataset(args.train_data)
    test_set = R22_H5_Dataset(args.test_data)

    train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    )

    test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8
    )

    # Define loss criterion
    criterion = nn.L1Loss()
    
    # Step 1: Compute relative contribution using perplexity analysis
    relative_contribution = perplexity_analysis_with_contributions(original_model, test_loader, criterion, args.device, num_iterations=args.num_iterations)
    
    # Step 2: Compute pruning ratios
    pruning_ratios = calculate_pruning_ratios_intense(relative_contribution, max_pruning_ratio=args.max_pruning_ratio, k=args.k)
    logging.info("\n=== Pruning Ratios ===")
    logging.info(f"Pruning Ratios: {', '.join(f'{r:.2f}' for r in pruning_ratios)}")
    
    # Step 3: Perform selective pruning
    logging.info("\n=== Performing Selective Pruning ===")
    pruned_model = selective_pruning(original_model, args.pruning_method, pruning_ratios, train_loader, args.device)
    
    # Save pruned model
    os.makedirs('pruning_results', exist_ok=True)
    torch.save(pruned_model, 'pruning_results/pruned_model')
    logging.info(f"Pruned model saved at: pruning_results/pruned_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model pruning using selective pruning method.")
    parser.add_argument("--model_path", type=str, default='best_trained_model.pth', help="Path to the original model.")
    parser.add_argument("--train_data", type=str, default='data_h5py_files/sepData_train_P100_N10.h5', help="Path to the training dataset.")
    parser.add_argument("--test_data", type=str, default='data_h5py_files/sepData_test_P100_N10.h5', help="Path to the testing dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader.")
    parser.add_argument("--device", type=str, help="Device to run the code on: CPU or CUDA")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations for perplexity analysis.")
    parser.add_argument("--max_pruning_ratio", type=float, default=0.9, help="Maximum pruning ratio.")
    parser.add_argument("--k", type=int, default=5, help="Scaling factor for pruning ratio computation.")
    parser.add_argument("--pruning_method", type=str, default="channel_pruning_Taylor_importance", help="Pruning method to use.")
    
    args = parser.parse_args()
    main(args)