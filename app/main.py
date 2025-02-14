import logging
from waggle.plugin import Plugin
import torch
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with Plugin() as plugin:
        logger.info("This is the start of the test app!")
    
    if torch.cuda.is_available():
        logger.info("CUDA is available.")

    # Set the duration for the loop (1 minute)
    try:
        duration = 60  # seconds
        end_time = time.time() + duration

        while time.time() < end_time:
            start_time = time.time()
            
            # Check available GPU memory
            free_memory, total_memory = torch.cuda.mem_get_info()
            logger.info(f"Available GPU memory: {free_memory / (1024 ** 2)} MB")

            # Generate a random tensor
            tensor = torch.randn(3, 3)
            logger.info("Generated a random tensor.")
            
            # Send the tensor to the CUDA GPU
            tensor = tensor.to(device='cuda')
            logger.info("Sent the tensor to the CUDA GPU.")
            
            # Optionally, perform some operations on the tensor
            tensor = tensor * 2
            logger.info("Performed operations on the tensor.")
            
            # Remove the tensor from CUDA memory
            del tensor
            torch.cuda.empty_cache()
            logger.info("Removed the tensor from CUDA memory.")
            
            # Calculate the time taken for the operations
            elapsed_time = time.time() - start_time
            
            # Sleep for the remaining time to ensure one tensor per second
            sleep_time = max(1.0 - elapsed_time, 0)
            time.sleep(sleep_time)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    logger.info("Finished generating tensors and sending them to CUDA GPU.")

if __name__ == "__main__":
    main()