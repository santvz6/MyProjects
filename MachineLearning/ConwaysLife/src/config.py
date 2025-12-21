import torch
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "src" / "logs"

# Ahora apuntamos a la raíz, luego a datasets
DATASETS_DIR = ROOT_DIR / "datasets" 
MNIST_DIR = DATASETS_DIR / "mnist"
MNIST_DATA_DIR = DATA_DIR / "mnist"
NUM_LOGS = 5
SEED = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")