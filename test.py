import sys
import os
import time
import resource
import numpy as np
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt

# Add Marabou library path
sys.path.insert(0, os.path.abspath("./Marabou"))
from maraboupy import Marabou


def main():
    # 1. Load the model
    args = argparse.ArgumentParser(description="Verify robustness of a FashionMNIST model using Marabou")
    args.add_argument("--model_path", type=str, default="onnx/custom_fashion_mnist_sim.onnx", help="Path to the ONNX model to verify")
    args.add_argument("--output_path", type=str, default="output/marabou_output.txt", help="Path to save Marabou output (counter-examples if found)")
    args.add_argument("--epsilon", type=float, default=0.01, help="L-infinity radius for robustness verification")
    args.add_argument("--adv_image_path", type=str, default="output/adv_image.png", help="Path to save adversarial image")
    args = args.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    epsilon = args.epsilon
    adv_image_path = args.adv_image_path

    print(f"Loading model: {model_path}")
    network = Marabou.read_onnx(model_path)

    # 2. Load FashionMNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    image, true_label = test_dataset[0]
    image_flat = image.numpy().flatten()
    
    print(f"Loaded image index 0. True label: {true_label}")

    
    # 3. Create Query 
    target_label = (true_label + 1) % 10  # target index 
    print(f"Setting Query -> Epsilon: {epsilon}, Target Label: {target_label}")

    # Load the network variables
    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    # Input constraints: L-infinity bounds (x - eps <= x' <= x + eps)
    for i, x in enumerate(input_vars):
        lower_bound = max(0.0, float(image_flat[i]) - epsilon)
        upper_bound = min(1.0, float(image_flat[i]) + epsilon)
        network.setLowerBound(x, lower_bound)
        network.setUpperBound(x, upper_bound)

    # Output constraints (Counter-example setup): 
    # i.e., Output(true_label) - Output(target_label) <= 0
    network.addInequality([output_vars[true_label], output_vars[target_label]], [1, -1], 0)

    # 4. Start Solving
    print(f"Solving targeted robustness against class {target_label}...")
    options = Marabou.createOptions(verbosity=0) # restrict printing
    start_time = time.perf_counter()
    exit_code, vals, stats = network.solve(output_path, options=options)
    elapsed_sec = time.perf_counter() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux
    peak_rss_kb = usage.ru_maxrss
    
    # 5. result
    with open(output_path, "a") as f:
        f.write(f"Verification time (sec): {elapsed_sec:.4f}\n")
        f.write(f"Peak RSS (KB): {peak_rss_kb}\n")
        if exit_code == "sat":
            # Write the satisfying assignment for inputs/outputs into the file.
            for i, var in enumerate(input_vars):
                if var in vals:
                    f.write(f"input {i} = {vals[var]}\n")
            for i, var in enumerate(output_vars):
                if var in vals:
                    f.write(f"output {i} = {vals[var]}\n")

            # Save adversarial input image if matplotlib is available.
            if plt is not None:
                adv_pixels = [vals[var] for var in input_vars]
                adv_image = np.array(adv_pixels, dtype=np.float32).reshape(28, 28)
                plt.imsave(adv_image_path, adv_image, cmap="gray")
                f.write(f"Saved adversarial image to: {adv_image_path}\n")
            else:
                f.write("Matplotlib not available. Skipped saving adversarial image.\n")

            msg = (
                f"\n[Result: SAT] -> The model is NOT robust against class {target_label}.\n"
                f"An adversarial example exists within epsilon={epsilon} that tricks the model.\n"
            )
            f.write(msg)
            print(msg)
        elif exit_code == "unsat":

            print(
                f"\n[Result: UNSAT] -> The model IS robust against class {target_label}.\n"
                f"Verified: No inputs within ||x' - x||_inf <= {epsilon} are classified as {target_label}.\n"
            )
        else:
            msg = f"\n[Result: {exit_code}] -> Something went wrong or timed out.\n"
            f.write(msg)
            print(msg)


if __name__ == "__main__":
    main()
