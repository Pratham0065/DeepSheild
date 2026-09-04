import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import sklearn
import PIL
import streamlit

print("===== DeepShield Environment =====")

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("\nAll libraries working successfully! ✅")