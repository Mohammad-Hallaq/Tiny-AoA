import torch
import time

def measure_inference_time(model: torch.nn.Module, validation_loader: torch.utils.data.DataLoader, inference_device: str, num_tests: int = 20, num_iterations: int = 1000) -> None:
    """
    Measure the inference time of a PyTorch model on CPU for multiple tests and iterations.
    """

    torch.manual_seed(42)

    
    # Move the model to CPU and set it to evaluation mode
    device = inference_device
    model.to(device)
    model.eval()

    # Disable gradient calculations for inference
    overall_total_time = 0.0

    # Fetch a single batch outside the loop
    first_batch = next(iter(validation_loader))
    input_tensor, _ = first_batch
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for test in range(num_tests):
            # print(f"Running Test {test + 1}/{num_tests}")
            test_total_time = 0.0

            for iteration in range(num_iterations):
                # Start timing
                start_time = time.time()
                
                # Perform inference
                _ = model(input_tensor)

                # End timing and record
                batch_inference_time = time.time() - start_time
                test_total_time += batch_inference_time

            # Calculate the average inference time per iteration for this test
            avg_test_time = test_total_time / num_iterations
            overall_total_time += test_total_time

            # print(f"Total inference time for Test {test + 1}: {test_total_time:.4f} seconds")
            # print(f"Average inference time per iteration for Test {test + 1}: {avg_test_time:.4f} seconds\n")

    # Calculate and print overall statistics
    overall_avg_time = overall_total_time / (num_tests * num_iterations)
    # print(f"Overall total inference time for {num_tests} tests: {overall_total_time:.4f} seconds")
    # print(f"Overall average inference time per iteration: {overall_avg_time:.4f} seconds")

    return overall_avg_time
