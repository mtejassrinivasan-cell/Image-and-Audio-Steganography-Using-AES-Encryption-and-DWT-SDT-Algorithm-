import cv2
import numpy as np
from stego_aes_dwt_svd import run_pipeline

# Create a random cover and secret
cover = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
cv2.imwrite("test_cov.png", cover)

secret = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
cv2.imwrite("test_sec.png", secret)

try:
    run_pipeline("test_cov.png", "test_sec.png", "test_stego.png", "test_rec.png", "test", alpha=0.08)
except Exception as e:
    print("Error:", e)
