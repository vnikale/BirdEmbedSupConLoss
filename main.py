import pandas as pd
import numpy as np
import librosa
import glob

import random

import torch

import csv
import io
import os

import IPython
from IPython.display import Audio
from sklearn.model_selection import train_test_split

from dataset import BirdcallMixsWavsDataset, audio_norm, BinaryBirdcallDataset
from trainer_separation import Trainer
from config import Config, ConfigRaw
from tdcnpp import TDCNpp
from loss import si_snr, snr_loss

from torch.utils.data import DataLoader

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main(SEED):
    cfg = ConfigRaw()

    # audio_files_2021 = glob.glob(os.path.join(cfg.data_folder,r'train_short_audio/',r'./*/*.ogg'))
    # audio_files_2021 = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in audio_files_2021]
    # audio_files_2021 = [os.path.join('train_short_audio', f) for f in audio_files_2021]


    # audio_files_2023 = glob.glob(os.path.join(cfg.data_folder,r'train_audio/',r'./*/*.ogg'))
    # audio_files_2023 = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in audio_files_2023]
    # audio_files_2023 = [os.path.join('train_audio', f) for f in audio_files_2023]

    # audio_files = audio_files_2021 + audio_files_2023
    # print(pd.DataFrame(audio_files))
    # metadata = files

    # train_files, valid_files = train_test_split(audio_files, test_size=0.2, random_state=42)

    # # Initialize datasets
    # train_dataset = BirdcallMixsWavsDataset(train_files, cfg)
    # valid_dataset = BirdcallMixsWavsDataset(valid_files, cfg)

    # Training DataLoader
    # train_loader = torch.utils.data.DataLoader(train_dataset, 
    #                                             batch_size=cfg.batch_size,
    #                                             drop_last=True,
    #                                             shuffle=True,
    #                                             pin_memory=False,
    #                                             prefetch_factor=1,
    #                                             num_workers=4)

    # # Validation DataLoader
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, 
    #                                             batch_size=cfg.batch_size,
    #                                             drop_last=True,
    #                                             shuffle=False,
    #                                             pin_memory=False,
    #                                             prefetch_factor=1,
    #                                             num_workers=4)
    
    files = sorted([os.path.join(cfg.data_folder, f) for f in os.listdir(cfg.data_folder) if f.endswith('.pt')])
    
    train_files, valid_files = train_test_split(files, test_size=0.1, random_state=SEED)

    train_dataset = BinaryBirdcallDataset(files=train_files, batch_size=cfg.batch_size)
    valid_files = BinaryBirdcallDataset(files=valid_files, batch_size=cfg.batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=1,
                                                    drop_last=False,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    prefetch_factor=2,
                                                    num_workers=7)
    
    valid_loader = torch.utils.data.DataLoader(valid_files,
                                                    batch_size=1,
                                                    drop_last=False,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    prefetch_factor=2,
                                                    num_workers=7)

    # Initialize model
    device = cfg.device
    model = TDCNpp(cfg.N, cfg.L, cfg.B, cfg.H, cfg.P, cfg.X, cfg.R, cfg.C, use_consistency=True).to(device)
    # model.cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters amount: {pytorch_total_params}')
    print(f'Receptive field of the model: {model.mask_net.receptive_field} or {model.mask_net.receptive_field / cfg.sample_rate * 1e3} ms')

    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_kwargs)
    criterion = snr_loss
    metric = si_snr

    checkpoint_fn = r'checkpoints_TDCNpp_binary_data_4/best_model_44.pt'
    # checkpoint_fn = None
    trainer = Trainer(model, 
                    criterion, metric,
                    optimizer, 
                    cfg.logger_kwargs, device, 
                    cfg,
                    checkpoint_fn=checkpoint_fn)
    
    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    SEED = 42

    set_seed(SEED)
    main(SEED)