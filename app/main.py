from waggle.plugin import Plugin
import torch
import time


def main():
    with Plugin() as plugin:
        print("This is the start of the test app!")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")

    # Set the duration for the loop (1 minute)
    duration = 10  # seconds
    end_time = time.time() + duration

    while time.time() < end_time:
        # Generate a random tensor
        tensor = torch.randn(100, 100)
        
        # Send the tensor to the CUDA GPU
        tensor = tensor.to('cuda')
        
        # Optionally, perform some operations on the tensor
        tensor = tensor * 2

    print("Finished generating tensors and sending them to CUDA GPU.")


if __name__ == "__main__":
    main()
