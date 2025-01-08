import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import importlib
from termcolor import colored, cprint
import time

from models.models import SimpleLinear, MLP, CNN
from experiments.loader import load_dataset, load_dataset_ndbc
from experiments.aux import loss_analysis, station_wave_information, station_wave_information_ndbc
from experiments.utils import RMSE, create_sequences, normalize_data, unnormalize_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def train(config, model, data, verbose=False):
    pass

def driver(config_name, aux=False):
    pass