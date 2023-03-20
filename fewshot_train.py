import torch
import numpy as np
from torch.utils.data import Subset
from dataset import CWRUDataset
from models import SiameseNet
from configs import window_size

from train import siamese_trainer
from configs import batch_size, learning_rate, n_iter, exp_list, rpm_list

dataset = CWRUDataset(exp_list, rpm_list, window_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# num_datasets = [60, 90, 120, 200, 300, 600, 900, 1500, 6000, 19800]
num_datasets = [60, 90, 120, 200, 300, 600, 900, 1600, 2000]

for num_dataset in num_datasets:

    model = SiameseNet()
    model.to(device)

    train_size = int(0.6 * num_dataset)
    val_size = int(0.2 * num_dataset)
    test_size = num_dataset - train_size - val_size

    train_indices = np.random.choice(len(dataset), train_size, replace=False)
    val_indices = np.random.choice(len(dataset), val_size , replace=False)
    test_indices = np.random.choice(len(dataset), test_size , replace=False)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    trainer = siamese_trainer(model, train_dataset, val_dataset, test_dataset, batch_size, learning_rate, n_iter, device, num_dataset)

    trainer.train()
    

