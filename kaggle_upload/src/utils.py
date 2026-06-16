import os
import numpy as np
import pandas as pd
import random
import torch

import math

from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)                         # Python built-in random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch GPU
    torch.cuda.manual_seed_all(seed)          # If using multi-GPU

    torch.backends.cudnn.deterministic = True # Forces deterministic algorithms
    torch.backends.cudnn.benchmark = False    # Disable to avoid non-deterministic algorithms

def get_lr_lambda(batch_size=8, mode='cos', epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lr_lambda(epoch):  # returns a factor to multiply with initial LR
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == 'exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step':
            lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs = epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        else:
            lr = lr_max
        return lr / lr_max  # normalize to multiply with optimizer's initial LR

    if plot:
        lrs = [lr_lambda(e) * lr_max for e in range(epochs)]
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(epochs), lrs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('LR Scheduler')
        plt.show()

    return lr_lambda
