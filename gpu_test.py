#!/usr/bin/env python3
"""
Quick GPU detection and test script
"""

import torch
import time


def test_gpu():
    print("=== GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

        # Test GPU computation
        print("\nTesting GPU computation...")
        device = torch.device("cuda")

        # Create some test tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        start_time = time.time()
        c = torch.matmul(a, b)
        gpu_time = time.time() - start_time

        print(f"GPU matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
        print(f"Result tensor is on device: {c.device}")

        # Test CPU comparison
        print("\nComparing with CPU...")
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time

        print(f"CPU matrix multiplication (1000x1000): {cpu_time:.4f} seconds")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")

    else:
        print("No GPU available. Training will use CPU.")
        print("This will be significantly slower for large models.")

    print("================")


if __name__ == "__main__":
    test_gpu()
