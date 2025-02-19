import cv2
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
def is_cuda_gpu_in_use():
    try:
        # Check if CUDA is available
        if not cv2.cuda.getCudaEnabledDeviceCount():
            logging.info("No CUDA-enabled GPU found.")
            return False
        
        # Set CUDA device
        device_id = 0
        cv2.cuda.setDevice(device_id)
        logging.info(f"CUDA-enabled GPU found. Device ID: {device_id}")
        # Create a CUDA matrix
        gpu_mat = cv2.cuda_GpuMat()
        logging.debug("Created a CUDA GpuMat object.")
        # Upload a dummy matrix to the GPU
        dummy_mat = cv2.UMat(1, 1, cv2.CV_8UC1)
        gpu_mat.upload(dummy_mat)
        logging.debug("Uploaded a dummy matrix to the GPU.")
        # Download the matrix back to the CPU
        result_mat = gpu_mat.download()
        logging.debug("Downloaded the matrix back to the CPU.")
        logging.info("CUDA GPU is in use.")
        return True
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False
# Example usage
if __name__ == "__main__":
    if is_cuda_gpu_in_use():
        logging.info("CUDA GPU is successfully being used.")
    else:
        logging.info("CUDA GPU is not being used.")