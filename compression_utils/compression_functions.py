import copy
import torch
import torch.nn.functional as F
import torch_pruning as tp
import numpy as np

def prune_model(trained_model, prune_method, pruning_ratios, train_loader):
    # Make a copy of the trained model
    model = copy.deepcopy(trained_model).to('cpu')  

    # Prepare pruning information for each block
    pruning_info = {
        i: {"block": model.blocks[i], "pruning_ratio": ratio}
        for i, ratio in enumerate(pruning_ratios)
    }

    if prune_method == 'channel_pruning_Taylor_importance':
        # Initialize TaylorImportance for pruning
        imp = tp.importance.TaylorImportance() 

        # Prepare a batch from the train loader for pruning and backward pass
        x, y = next(iter(train_loader))
        x, y = x.to('cpu'), y.to('cpu')

        # Perform forward and backward passes to calculate importance scores
        if isinstance(imp, tp.importance.TaylorImportance):
            y_hat = model(x)
            loss = F.l1_loss(y_hat, y)
            loss.backward()

        # Define layers to be ignored during pruning
        ignored_layers = [
            model.conv_stem, 
            model.bn1,
            model.global_pool,
            model.conv_head,
            model.norm_head,
            model.act2,
            model.flatten,
            model.classifier
        ]

        # Example input for MACs and Params calculation
        example_inputs = torch.randn(1, 8, 4096).to('cpu')
        original_macs, original_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        # Iterate through each block and apply pruning
        for i, info in pruning_info.items():
            block_to_prune = info["block"]
            pruning_ratio = info["pruning_ratio"]

            # Ignore all blocks except the current block to prune
            ignored_layers_block = [pruning_info[j]["block"] for j in range(len(pruning_info)) if j != i]
            combined_ignored_layers = ignored_layers + ignored_layers_block

            # print(f"Pruning block {i} with initial ratio: {pruning_ratio}")

            # Pruning loop: Continue pruning until no parameters are further reduced
            while True:
                # Apply pruning for the current block
                pruner_group = tp.pruner.MagnitudePruner(
                    model,
                    example_inputs=x,
                    importance=imp,
                    pruning_ratio=pruning_ratio,
                    ignored_layers=combined_ignored_layers
                )
                pruner_group.step()

                # Recalculate MACs and parameters after pruning
                macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
                # print(f"MACs: {macs / 1e9:.2f} G, #Params: {nparams / 1e3:.2f} K")
                # print(f"Parameter reduction: {original_nparams - nparams}")

                # Check if no parameters were reduced, then break the loop
                if original_nparams - nparams == 0:
                    break

                # Update the pruning ratio for iterative pruning
                original_nparams = nparams
                pruning_ratio = 0.5  # Adjust pruning ratio for the next iteration

        # Free up memory
        del x, y
        torch.cuda.empty_cache()

    return model, macs, nparams



# Function to perform perplexity analysis, return average loss, and calculate relative contribution percentages
def perplexity_analysis_with_contributions(original_model, data_loader, criterion, device, num_iterations=5):
    total_block_losses = [0.0 for _ in range(len(original_model.blocks))]
    params_reduction = []  # Store the parameter reduction for each block
    macs_reduction = []

    # Step 1: Compute the baseline loss (without block replacement)
    print("Computing baseline loss without block replacement...")
    baseline_loss = compute_baseline_loss(original_model, data_loader, criterion, device)
    print(f"Baseline Loss: {baseline_loss}")

    example_inputs = torch.randn(1, 8, 4096).to(device)  # Generate example input for calculating MACs and parameters
    original_macs, original_nparams = tp.utils.count_ops_and_params(original_model, example_inputs)
    
    # Iterate through each block for replacement
    for block_idx in range(len(original_model.blocks)):

        # output_channels = get_first_layer_output_channels(original_model)

        print(f"Replacing block {block_idx}")
        pruning_ratios = pruning_ratios = (np.eye(len(original_model.blocks)) * 0.8)[block_idx]

        # print("Pruning ratios for this iteration are: ", pruning_ratios)
        
        pruned_model, macs, nparams = prune_model(original_model,'channel_pruning_Taylor_importance', pruning_ratios, data_loader)

        print(f"Macs reduction is: {((original_macs - macs) / original_macs * 100):.2f}%\n", f"Parameters reduction is: {((original_nparams - nparams)/original_nparams*100):.2f}%")

        # Record the parameter reduction
        params_reduction.append(original_nparams - nparams)
        macs_reduction.append(original_macs - macs)
        
        pruned_model.to(device)

        # Run validation and compute loss
        pruned_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = pruned_model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        # Calculate average loss for the current block
        average_loss = total_loss / len(data_loader)
        print(f'Loss for block {block_idx}: {average_loss}')
        
        # Accumulate the loss for each block
        total_block_losses[block_idx] += average_loss

    # Step 2: Calculate the final averaged loss and contribution for each block across iterations
    total_increase_in_loss = 0.0
    block_increases = []
    total_params_reduction = 0.0
    total_macs_reduction = 0.0
    
    for block_idx in range(len(original_model.blocks)):
        final_average_loss = total_block_losses[block_idx] / num_iterations
        increase_in_loss = final_average_loss - baseline_loss
        block_increases.append(increase_in_loss)
        total_increase_in_loss += increase_in_loss  # Accumulate total increase in loss
        total_params_reduction += params_reduction[block_idx]  # Accumulate total parameter reduction
        total_macs_reduction += macs_reduction[block_idx]  # Accumulate total macs reduction
    
    # Step 3: Calculate the relative contribution of each block to the total increase in loss and params saved
    relative_contributions = []
    weighted_importance_scores = []
    print("\nRelative contribution of each block to total loss increase and parameter reduction:")

    for block_idx in range(len(original_model.blocks)):
        # Calculate relative contribution to the loss increase
        relative_contribution_loss = (block_increases[block_idx] / total_increase_in_loss) * 100
        
        # Adjust parameter contribution to reflect reduced importance for larger reductions
        relative_contribution_params = (1 - (params_reduction[block_idx] / total_params_reduction)) * 100
        relative_contribution_macs = (1 - (macs_reduction[block_idx] / total_macs_reduction)) * 100

        # Combine these two using a weighted importance score (example: 70% weight to loss, 30% to parameter savings)
        weight_loss = 0.5
        weight_params = 0.3
        weight_macs = 0.2
        weighted_importance = (weight_loss * relative_contribution_loss) + (weight_params * relative_contribution_params) + (weight_macs * relative_contribution_macs)

        print(f'Block {block_idx} contributes {relative_contribution_loss:.2f}% to the total increase in loss and reduces {100 - relative_contribution_params:.2f}% of parameters.')
        print(f'Weighted importance score for Block {block_idx}: {weighted_importance:.2f}')
        
        relative_contributions.append(relative_contribution_loss)
        weighted_importance_scores.append(weighted_importance)

    # Return both the relative loss contributions and weighted importance scores
    return weighted_importance_scores

# Helper function to compute the baseline loss (without replacing any block)
def compute_baseline_loss(original_model, data_loader, criterion, device):
    original_model.to(device)
    original_model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = original_model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    baseline_loss = total_loss / len(data_loader)
    return baseline_loss


def calculate_pruning_ratios_intense(contributions, max_pruning_ratio=0.9, k=5):
    """
    Calculate pruning ratios based on intense nonlinear scaling (exponential decay) of the relative contributions.

    Parameters:
    - contributions (list): List of relative contributions (in percentages) of each block to total loss increase.
    - max_pruning_ratio (float): Maximum pruning ratio to be assigned to the least important layer. Default is 0.9.
    - k (int): Factor controlling the intensity of the scaling (larger k makes the ratio more intense).

    Returns:
    - pruning_ratios (list): List of pruning ratios for each block.
    """
    # Normalize the contributions to get values between 0 and 1
    total_contribution = sum(contributions)
    normalized_contributions = [contribution / total_contribution for contribution in contributions]

    # Apply exponential decay to magnify the effect for less important blocks
    pruning_factors = [np.exp(-k * nc) for nc in normalized_contributions]

    # Normalize the pruning factors so they stay within the max pruning ratio
    max_factor = max(pruning_factors)
    normalized_factors = [pf / max_factor for pf in pruning_factors]

    # Scale by the maximum pruning ratio
    pruning_ratios = [max_pruning_ratio * nf for nf in normalized_factors]

    pruning_ratios = [round(num, 2) for num in pruning_ratios]

    return pruning_ratios


def selective_pruning(trained_model, prune_method, pruning_ratios, train_loader, device):
   
    # Make a copy of the trained model
    model = copy.deepcopy(trained_model)
    device = device  # Assuming general_device is defined elsewhere
    model.to(device)

    pruning_info = {
        i: {"block": model.blocks[i], "pruning_ratio": ratio}
        for i, ratio in enumerate(pruning_ratios)
    }

    if prune_method == 'channel_pruning_Taylor_importance':
        imp = tp.importance.TaylorImportance() 
        
        # Prepare a batch from the train loader for pruning and backward pass
        # Reset the DataLoader iterator to ensure the same batch is selected each time
        train_loader_iter = iter(train_loader)
        batch = next(train_loader_iter)
        x, y = batch
        x, y = x.to(device), y.to(device)

        # Perform forward and backward passes to calculate importance scores if using TaylorImportance
        if isinstance(imp, tp.importance.TaylorImportance):
            y_hat = model(x)
            loss = F.l1_loss(y_hat, y)
            loss.backward()

        # Define layers to always ignore (conv_stem, bn1, and classifier)
        ignored_layers = [
            model.conv_stem, 
            model.bn1,
            model.global_pool,
            model.conv_head,
            model.norm_head,
            model.act2,
            model.flatten,
            model.classifier
        ]

        # Prune each block while ignoring other layers
        for i, info in pruning_info.items():
            pruning_ratio = info["pruning_ratio"]
            
            # Add all blocks to the ignored layers except the block being pruned
            ignored_layers_block = [pruning_info[j]["block"] for j in range(len(pruning_info)) if j != i]

            # Combine fixed ignored layers (conv_stem, bn1, classifier) with the ignored blocks
            combined_ignored_layers = ignored_layers + ignored_layers_block

            # Apply pruning using the combined ignored layers
            pruner_group = tp.pruner.MagnitudePruner( 
                model,
                example_inputs=x,
                importance=imp,
                pruning_ratio=pruning_ratio,
                ignored_layers=combined_ignored_layers,
                iterative_steps=1,
            )

            # Step through pruning
            pruner_group.step()

    # Counting MACs and Params after pruning
    example_inputs = torch.randn(1, 8, 4096).to(device)  # Generate example input for calculating MACs and parameters
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs of the pruned model: {macs / 1e9} G, #Params of the pruned model: {nparams / 1e3} K")

    # Free up GPU memory
    del x, y, batch
    torch.cuda.empty_cache()

    return model