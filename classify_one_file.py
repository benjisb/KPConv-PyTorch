import os
import shutil
import time

from torch.utils.data import DataLoader

from datasets.LAS import LASCollate, LASDataset, LASSampler
from models.architectures import KPFCNN
from utils.config import Config
from utils.tester import ModelTester

# Set which gpu is going to be used
GPU_ID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Path to checkpoint TAR archive
chkp = r"results\Log_2025-07-11_18-22-47\checkpoints\current_chkp.tar"

# Path to parameters TXT file
params = r"results\Log_2025-07-11_18-22-47\parameters.txt"

# Make sure the filename is "parameters.txt"
filename = os.path.basename(params)
if not filename.lower() == "parameters.txt":
    os.rename(filename, params.replace(filename, "parameters.txt"))

# Initialize configuration
config = Config()
config.load(os.path.dirname(params))

# Set configuration parameters
config.validation_size = 200
config.input_threads = 0

# Path to input LAS file
las = r"C:\Users\BEBLADES\data\dales\train\5080_54435_reclass.las"

# Make sure the file is in validation set
split_path = las.split(os.sep)
dst_path = os.sep.join(split_path[:-2])
if not split_path[-2] == "validate":
    val_dir = os.path.join(dst_path, "validate")
    os.makedirs(val_dir, exist_ok=True)
    shutil.copy(las, val_dir)

print("\nData Preparation")
print("****************")

# Initiate dataset
test_dataset = LASDataset(config, set="test", path=dst_path)
test_sampler = LASSampler(test_dataset)
collate_fn = LASCollate
test_loader = DataLoader(
    test_dataset, 
    batch_size=1, 
    sampler=test_sampler,
    collate_fn=collate_fn,
    num_workers=config.input_threads,
    pin_memory=True
)

# Calibrate samplers
test_sampler.calibration(test_loader, verbose=True)

print("\nModel Preparation")
print("*****************")

# Define network model
t1 = time.time()
net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

# Define a visualizer class
tester = ModelTester(net, chkp_path=chkp)
print(f"Done in {(time.time() - t1):.1f}s\n")

print("\nStart test")
print("**********\n")

# Set number of votes for test
num_votes = 100

# Run test
tester.cloud_segmentation_test(net, test_loader, config, num_votes)