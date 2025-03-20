import argparse
import os
import torch
import torch_pruning as tp
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from performance_utils.evaluation_functions import measure_inference_time
from data_utils.RFclassifier import RFClassifier 
from data_utils.R22_dataset import R22_H5_Dataset
import pytorch_lightning as L

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Pruned Models')
    parser.add_argument('--models_folder', type=str, default='Best_latest_models', help='Folder containing pruned models')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the models on')
    parser.add_argument('--pruning_ratios', nargs='+', type=int, default=[98, 93, 85, 72, 58, 0], help='List of pruning ratios')
    parser.add_argument('--measure_inference', action='store_true', help='Measure inference time on CPU and GPU')
    return parser.parse_args()

def main():
    args = parse_args()
    
    models_folder = args.models_folder
    general_device = torch.device(args.device)
    your_hardware = args.measure_inference
    pruning_ratios = args.pruning_ratios
    
    # Initialize lists
    loss_list = []
    model_sizes_list = []
    num_params = []
    macs_list = []
    inference_time_gpu_list = []
    inference_time_cpu_list = []
    
    # Create a folder to save plots
    plots_folder = "plots"
    os.makedirs(plots_folder, exist_ok=True)

    # Load test loader (modify accordingly)
    test_data = 'data_h5py_files/sepData_test_P100_N10.h5'
    test_set = R22_H5_Dataset(test_data)
    test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=128,
    shuffle=False,
    num_workers=8
    )

    # Sort models
    model_files = sorted(Path(models_folder).glob('*.pth'), key=lambda x: x.name)
    
    for model_file in model_files:
        print(f"Processing {model_file.name}")
        
        pruned_model = torch.load(model_file, map_location=general_device)
        rf_classifier_pruned = RFClassifier(pruned_model)
        trainer = L.Trainer(max_epochs=1, accelerator=args.device, benchmark=True, precision='32-true')
        
        test_results = trainer.test(rf_classifier_pruned, test_loader)
        test_loss = test_results[0]['test_loss']  
        loss_list.append(test_loss)
        
        # Count parameters and MACs
        example_inputs = torch.randn(1, 8, 4096).to(general_device)
        pruned_model = pruned_model.to(general_device)
        macs, nparams = tp.utils.count_ops_and_params(pruned_model, example_inputs)
        num_params.append(nparams)
        macs_list.append(macs)
        
        # Model size
        model_size = os.path.getsize(model_file) / (1024 ** 2)
        model_sizes_list.append(model_size)
        
        if your_hardware:
            inference_time_cpu_list.append(measure_inference_time(pruned_model, test_loader, 'cpu'))
            inference_time_gpu_list.append(measure_inference_time(pruned_model, test_loader, 'cuda'))

        
        print(f"Model: {model_file.name}, Loss: {test_loss}, MACs: {macs/1e9:.2f} G, Params: {nparams/1e3:.2f} K, Size: {model_size:.2f} MB")
    
    if not your_hardware:
        inference_time_gpu_list = [6.696503027343749, 6.919866894531249, 6.989533398437499, 7.8285044677734374, 9.500997460937501, 16.17931850585937]
        inference_time_cpu_list = [0.0393, 0.0403, 0.0408, 0.0487, 0.0498, 0.1849]
        
    # Sort results by pruning ratio
    combined_data = sorted(zip(pruning_ratios, loss_list, num_params, model_sizes_list, macs_list, inference_time_gpu_list, inference_time_cpu_list), key=lambda x: x[0])
    pruning_ratios, losses, num_parameters, model_sizes, macs_count, inference_time_gpu, inference_time_cpu = map(list, zip(*combined_data))
    
    # Compute reduction factors
    original_model_size = model_sizes[0]
    original_macs = macs_count[0]
    original_inference_time_gpu = inference_time_gpu[0]
    original_inference_time_cpu = inference_time_cpu[0]
    
    model_size_factor = [original_model_size / size for size in model_sizes]
    macs_factor = [original_macs / macs for macs in macs_list]
    inference_time_factor_gpu = [original_inference_time_gpu / time for time in inference_time_gpu]
    inference_time_factor_cpu = [original_inference_time_cpu / time for time in inference_time_cpu]
    
    
    # Define colors
    dark_blue = '#2A3F5F'
    steel_blue = '#87CEEB'
    medium_blue = '#4F6D8C'
    dark_navy_blue = '#87CEEB'
    

    # Circle sizes for scatter plot
    size_scale = 100
    sizes = [p / size_scale for p in num_parameters]

    # Plot settings
    plt.rcParams.update({'font.size': 15, 'font.family': 'serif'})

    # Scatter plot: Loss vs. Pruning Ratio
    plt.figure(figsize=(12, 7))
    plt.scatter(pruning_ratios, losses, s=sizes, c=steel_blue, alpha=1, edgecolors="black", label="Pruned Models")

    # Add line connecting the circles
    plt.plot(pruning_ratios, losses, linestyle='-', color='black', linewidth=1, alpha=0.5)  # Fine line connecting points

    # Label and title settings
    plt.xlabel("Pruning Ratio (%)", fontsize=22)
    plt.ylabel("Loss (MAE)", fontsize=22)
    plt.title("Loss (MAE) vs. Pruning Ratio with Circle Sizes Reflecting Model Size", fontsize=20, weight='bold')

    # Set font size for x and y axis tick labels
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xlim(pruning_ratios[0]-8, pruning_ratios[-1] * 1.08)
    plt.ylim(losses[0] * 0.8, losses[-1] * 1.22)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Add annotations for number of parameters for each point with relative placement
    for i, txt in enumerate(num_parameters):
        formatted_txt = f"{txt / 1e3:.2f}K"  # Format as thousands (K)
        
        # Adjust the offset dynamically based on specific conditions
        if i == 0:
            offset_x = 5  
            offset_y = 40
            formatted_txt = f"{txt / 1e3:.2f}K \n(Baseline)"
        elif i in {len(num_parameters) - 1, len(num_parameters) - 2, len(num_parameters) - 3}:
            offset_x = 5 
            offset_y = 17
        else:
            offset_x = 5 
            offset_y = 27

        plt.annotate(formatted_txt, (pruning_ratios[i], losses[i]), 
                    textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=16.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "loss_vs_pruning_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate reductions and factors
    model_size_factor = [original_model_size / size for size in model_sizes]
    macs_factor = [original_macs / macs for macs in macs_count]
    inference_time_factor_gpu = [original_inference_time_gpu / time for time in inference_time_gpu]
    inference_time_factor_cpu = [original_inference_time_cpu / time for time in inference_time_cpu]

    pruning_ratios = pruning_ratios[1:]

    
    # X positions for bar plots
    x = np.arange(len(pruning_ratios))
    bar_width = 0.35
    label_fontsize = 25
    title_fontsize = 25
    tick_fontsize = 25
    value_fontsize = 25

    # Plot MACs Reduction as speed-up factor
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.set_ylim(0, max(macs_factor) * 1.1)
    bars2 = ax2.bar(x, macs_factor[1:], width=bar_width, color=dark_blue, edgecolor='black')
    ax2.set_xlabel("Pruning Ratio (%)", fontsize=label_fontsize)
    ax2.set_ylabel("MACs Reduction", fontsize=label_fontsize)
    ax2.set_title("MACs Reduction Factor vs. Pruning Ratio", fontsize=title_fontsize, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(ratio) for ratio in pruning_ratios], fontsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}X", ha='center', va='bottom', fontsize=value_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "macs_vs_pruning_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Inference Time Speed-Up on CPU and GPU
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_ylim(0, max(inference_time_factor_cpu) * 1.1)

    x_gpu = [pos - bar_width / 2 for pos in x]  
    x_cpu = [pos + bar_width / 2 for pos in x] 

    # Plot bars
    bars_gpu = ax.bar(x_gpu, inference_time_factor_gpu[1:], width=bar_width, color=dark_navy_blue, edgecolor='black', label="GPU Acceleration")
    bars_cpu = ax.bar(x_cpu, inference_time_factor_cpu[1:], width=bar_width, color=dark_blue, edgecolor='black', label="CPU Acceleration")

    # Labels and title
    ax.set_xlabel("Pruning Ratio (%)", fontsize=label_fontsize * 0.8)
    ax.set_ylabel("Inference Time Reduction", fontsize=label_fontsize * 0.8)
    ax.set_title("Inference Time Speed-Up on CPU and GPU vs. Pruning Ratio", fontsize=title_fontsize * 0.8, weight='bold')

    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(ratio) for ratio in pruning_ratios], fontsize=tick_fontsize * 0.8)

    # Add labels to the bars
    for bars, label in zip([bars_gpu, bars_cpu], ["GPU", "CPU"]):
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}X", ha='center', va='bottom', fontsize=value_fontsize * 0.8)

    # Add legend
    ax.legend(fontsize=label_fontsize * 0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "cpu&gpu_vs_pruning_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()

    original_num_parameters = num_parameters[0]
    parameter_reduction_factor = [original_num_parameters / param for param in num_parameters]

    # Plot Model Size and Parameter Reduction as multiplicative factors
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_ylim(0, max(max(model_size_factor), max(parameter_reduction_factor)) * 1.1)

    # Model size reduction bars
    bars1 = ax1.bar(x - bar_width / 2, model_size_factor[1:], width=bar_width, color=steel_blue, edgecolor='black', label="Model Size Reduction")
    # Parameter count reduction bars
    bars2 = ax1.bar(x + bar_width / 2, parameter_reduction_factor[1:], width=bar_width, color=dark_blue, edgecolor='black', label="Parameter Count Reduction")

    # Labels and Title
    label_fontsize = 22
    title_fontsize = 22
    tick_fontsize = 22
    value_fontsize = 20
    ax1.set_xlabel("Pruning Ratio (%)", fontsize=label_fontsize)
    ax1.set_ylabel("Reduction Factor", fontsize=label_fontsize)
    ax1.set_title("Reduction Factor for Model Size and Parameter Count vs. Pruning Ratio", fontsize=title_fontsize, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(ratio) for ratio in pruning_ratios], fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)

    # Adding text labels on the bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}X", ha='center', va='bottom', fontsize=value_fontsize)

    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.1f}X", ha='center', va='bottom', fontsize=value_fontsize)

    # Legend
    ax1.legend(fontsize=tick_fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "size&paramsing_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()

    
if __name__ == '__main__':
    main()
