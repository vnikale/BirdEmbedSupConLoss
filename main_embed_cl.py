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

from dataset import BirdcallMixsWavsDataset, audio_norm, BinaryBirdcallDataset,BinaryBirdcallDatasetAugment,BinaryBirdcallDatasetAugmentMultiView, BirdcallWavsDatasetMultiviewSimple
# from trainer_contrast_only import Trainer as TrainerContrast
from trainer_embed_cl import Trainer as TrainerContrast

from config import Config, ConfigRaw
from classifier import Classifier

from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from torchmetrics import AveragePrecision,Accuracy
from dataset import BinaryBirdcallDataset, BinaryBirdcallDatasetContrast
from collections import defaultdict

import torch.nn as nn
from tdcnpp import TDCNpp, MaskNetworkPlus

import json
import torch.jit as jit


from coder import ClassCoder
from embed_models import EmbeddingTDCNpp, EmbeddingClassifier,EmbeddingTDCNpp5, EmbeddingTDCNpp4, EmbeddingClassifierLayer, DownstreamClassifier

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift ,AddBackgroundNoise,AddGaussianSNR
from audiomentations import LowShelfFilter, HighShelfFilter,PolarityInversion, LowPassFilter, HighPassFilter,Normalize
import random
import numpy as np
import re


def class_count(files, classes):
    class_patterns = {cls: re.compile(cls) for cls in classes}
    
    class_counts = {}
    for cls, pattern in class_patterns.items():
        class_counts[cls] = sum(1 for _ in filter(pattern.search, files))

    return class_counts

def balance_classes(files, classes):
    # Count the occurrence of each class
    class_counts = class_count(files,classes)
    
    # Find the maximum class occurrence
    max_count = max(class_counts.values())
    # if max_count >= 2 * np.mean(list(class_counts.values())):
    #     max_count = int(max_count / 2)
    
    # Rebalance the list with repetition, residuals, and sampling down
    balanced_files = []
    for cls, count in class_counts.items():
        cls_files = [file for file in files if cls in file]

        # If the class count is greater than max_count, randomly sample down to max_count
        if count > max_count:
            balanced_files.extend(random.sample(cls_files, max_count))
            continue

        # Calculate the number of additional files needed to reach max_count
        additional_files_needed = max_count - count

        if additional_files_needed < count:
            # If we need fewer files than the current count, just randomly sample the additional ones
            residual_files = random.sample(cls_files, additional_files_needed)
            balanced_files.extend(cls_files + residual_files)
        else:
            # Replicate the files to get closer to max_count
            repetition_files = cls_files * (additional_files_needed // count)
            
            # Calculate the residual after repetition
            residual_count = max_count - (count + len(repetition_files))
            
            # Randomly sample the residuals to make the count exactly max_count
            residual_files = random.sample(cls_files, max(0, residual_count))
            
            # Combine the original, replicated, and residual files
            balanced_files.extend(cls_files + repetition_files + residual_files)
    return balanced_files

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
    cfg = Config()

    cfg.data_folder = 'birdclef'

    audio_files_2022 = glob.glob(os.path.join(cfg.data_folder,r'train_audio_2022/',r'./*/*.ogg'))
    audio_files_2022 = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in audio_files_2022]
    audio_files_2022 = [os.path.join('train_audio_2022', f) for f in audio_files_2022]


    audio_files_2023 = glob.glob(os.path.join(cfg.data_folder,r'train_audio/',r'./*/*.ogg'))
    audio_files_2023 = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in audio_files_2023]
    audio_files_2023 = [os.path.join('train_audio', f) for f in audio_files_2023]

    bg = glob.glob(os.path.join(cfg.data_folder,r'background/',r'./*/*.ogg'))
    bg = [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in bg]
    bg = [os.path.join('background', f) for f in bg]
    np.random.shuffle(bg)
    bg1 = bg[:350]

    # audio_files =  audio_files_2022 + audio_files_2023 + bg1
    audio_files =  audio_files_2022 + bg1


    classes = [os.path.basename(os.path.dirname(fn)) for fn in audio_files]
    set_classes = set(classes)
    set_classes = sorted(set_classes)
    class_encoder = {c: i for i, c in enumerate(set_classes)}
    class_encoder = defaultdict(lambda: len(set_classes), class_encoder)
    class_decoder = {i: c for c, i in class_encoder.items()}
    class_decoder[len(set_classes)] = 'unknown'
    num_classes = len(set_classes) + 1 # + 1 for unknown
    taxonomy_fn = 'eBird_Taxonomy_v2021.csv'
    taxonomy = pd.read_csv(taxonomy_fn)
    # taxonomy = None
    
    coder = ClassCoder(class_encoder, set_classes, taxonomy=taxonomy)

    num_of_o = len(coder.taxonomy['ORDER1'].dropna().unique()) + 1
    num_of_f = len(coder.taxonomy['FAMILY'].dropna().unique()) + 1

    cfg = ConfigRaw()

    cfg.enc_ker = 256
    cfg.enc_dim = 512


    enc_dim = cfg.enc_dim
    enc_ker = cfg.enc_ker
    conv_in_channels = cfg.conv_in_channels
    conv_out_channels = cfg.conv_out_channels
    conv_ker = cfg.conv_ker
    n_blocks = cfg.n_blocks
    n_repeats = cfg.n_repeats
    # num_sources = cfg.num_sources

    metric = AveragePrecision(task="multilabel", num_labels=num_classes, average='macro').to(cfg.device)
    # metric = Accuracy(task="multiclass", num_classes=len(set_classes)+1, ignore_index=len(set_classes)).to(cfg.device)

    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    # criterion = lambda x, y: sigmoid_focal_loss(x, y, gamma=2.0, alpha=0.1, reduction='sum')
    
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    cfg.data_folder = 'birdclef'
    noises = bg
    noises = [os.path.join(cfg.data_folder, f) for f in noises]


    augment = Compose([
        # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.5),
        AddGaussianSNR(min_snr_db = 5, max_snr_db = 30, p=0.5),
        LowShelfFilter(p = 0.3, min_gain_db=-3, max_gain_db=3),
        AddBackgroundNoise(sounds_path=noises, p=0.4, min_snr_db=8, max_snr_db=30.0,),
        HighShelfFilter( p=0.2, min_gain_db=-3, max_gain_db=3),
        LowPassFilter(p=0.4, min_rolloff=6, max_rolloff=12),
        HighPassFilter(p=0.3, min_rolloff=6, max_rolloff=12),
        TimeStretch(min_rate=0.99, max_rate=1.01, p=0.3),
        PitchShift(min_semitones=-0.15, max_semitones=0.2, p=0.3),
        Shift(p=0.5),
        Normalize(p=0.4)
    ])


    # data_folder_pos = 'binary_data_loudest_all_pos'
    # data_folder_neg = 'binary_data_loudest_all_neg'
    cfg.batch_size = 756
    cfg.AMP = True
    cfg.frame_size_s = 5

    # files_pos = sorted([os.path.join(data_folder_pos, f) for f in os.listdir(data_folder_pos) if f.endswith('.pt')])
    # files_neg = sorted([os.path.join(data_folder_neg, f) for f in os.listdir(data_folder_neg) if f.endswith('.pt')])
    
    # files_pos = files_pos[:len(files_pos)//4] + files_pos[len(files_pos)//2:len(files_pos)//2 + len(files_pos)//4]
    # files_neg = files_neg[len(files_neg)//4:len(files_neg)//2] + files_neg[len(files_neg)//2 + len(files_neg)//4:]
    # files = files_pos + files_neg

    # data_folder_neg = 'binary_data_loudest_all_bg_neg_2022'
    # files = sorted([os.path.join(data_folder_neg, f) for f in os.listdir(data_folder_neg) if f.endswith('.pt')])
    # files = random.sample(files, 20)
    # # random sample 500 examples for testing of code
    # files_pos = random.sample(files_pos, 500)
    # files_neg = random.sample(files_neg, 500)

    # files_pos_train, files_pos_valid = train_test_split(files_pos, test_size=0.1, random_state=42)
    # files_neg_train, files_neg_valid = train_test_split(files_neg, test_size=0.1, random_state=42)

    
    set_classes_2022 = set([os.path.basename(os.path.dirname(fn)) for fn in audio_files_2022])
    set_classes_2023 = set([os.path.basename(os.path.dirname(fn)) for fn in audio_files_2023])

    audio_files2022_balanced = balance_classes(audio_files_2022, set_classes_2022)
    classes_2022_balanced = [os.path.basename(os.path.dirname(fn)) for fn in audio_files2022_balanced]

    # audio_files2023_balanced = balance_classes(audio_files_2023, set_classes_2023)
    # classes_2023_balanced = [os.path.basename(os.path.dirname(fn)) for fn in audio_files2023_balanced]

    bg1_classes = ['noise']*len(bg1)

    # files = audio_files2022_balanced + audio_files2023_balanced + bg1
    # classes = classes_2022_balanced + classes_2023_balanced + bg1_classes
    
    files = audio_files2022_balanced + bg1
    classes = classes_2022_balanced + bg1_classes

    
    train_files, valid_files = train_test_split(files, test_size=0.2, random_state=42, stratify=classes)

    # train_files = train_files[:10]
    # valid_files = valid_files[:5]

    delete_classes = ['maupar', 'afpkin1', 'lotcor1', 'yebsto1', 'brtcha1', 'whctur2', 'whhsaw1','crefra2']
    valid_files = [f for f in valid_files if os.path.basename(os.path.dirname(f)) not in delete_classes]

    classes_train = [os.path.basename(os.path.dirname(fn)) for fn in train_files]
    classes_valid = [os.path.basename(os.path.dirname(fn)) for fn in valid_files]

    cfg.n_views = 2

    transform_aug = lambda x: torch.Tensor(augment(samples=x.numpy(), sample_rate=cfg.sample_rate))


    train_dataset = BirdcallWavsDatasetMultiviewSimple(train_files, cfg, classes = classes_train,  crop_type = 'loudest',
                                                       transform_aug=transform_aug, do_augment = True, n_views=cfg.n_views, keep_original=True)
    
    valid_dataset = BirdcallWavsDatasetMultiviewSimple(valid_files,cfg, classes = classes_valid,  crop_type = 'loudest', 
                                                       transform_aug=None, do_augment = False, n_views=cfg.n_views)
    
    # train_dataset = BinaryBirdcallDatasetContrast(files_pos_train, files_neg_train, cfg.batch_size, only_sources=True)
    # valid_dataset = BinaryBirdcallDatasetContrast(files_pos_valid, files_neg_valid, cfg.batch_size, only_sources=True)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.batch_size,
                                                    drop_last=False,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    prefetch_factor=2,
                                                    num_workers=32)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=cfg.batch_size*cfg.n_views,
                                                    drop_last=False,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    prefetch_factor=2,
                                                    num_workers=32)
    device = 'cuda'
    cfg.device = device

    emb_dim = 1024
    cfg.emb_dim = emb_dim

    # Initialize model
    device = cfg.device
    input_size = cfg.frame_size_s * cfg.sample_rate
    # model = EmbeddingTDCNpp(emb_dim, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, input_size)
    # model = EmbeddingTDCNpp2(emb_dim, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, input_size)
    # model_emb = EmbeddingTDCNpp4(emb_dim, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, input_size)
    model_emb = EmbeddingTDCNpp5(emb_dim, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, input_size, norm_emb=False)
    model_emb = model_emb.to(cfg.device)

    num_of_hidden = 2
    size_of_layers = 1024

    cfg.num_of_hidden = num_of_hidden
    cfg.size_of_layers = size_of_layers

    # model = EmbeddingClassifier(emb_dim, num_classes, num_of_hidden, size_of_layers, activation=nn.ReLU(), num_of_f = num_of_f, num_of_o = num_of_o)
    # model = EmbeddingClassifierLayer(emb_dim, num_classes, activation=nn.Hardswish(), num_of_f = None, num_of_o = None)
    model = DownstreamClassifier(emb_dim, num_classes)

    model.apply(init_weights)

    model = model.to(cfg.device)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Model embed parameters amount: {pytorch_total_params}')

    cfg.logger_kwargs['comment'] = '_embednet_cl'
    cfg.accumulation_steps = 1
    cfg.optim_kwargs = {
                        'lr': 0.5 * cfg.accumulation_steps, 
                        'weight_decay': 0
                        }
    cfg.max_norm = 10

    cfg.optimizer = torch.optim.AdamW
    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_kwargs)

    cfg.max_epochs = 100

    # cfg.margin = 2.0
    # cfg.factor = 1.5 # margin taxon increase factor
    cfg.l_spe = 1
    cfg.l_f = 0
    cfg.l_o = 0

    # cfg.l_cl = 0.8
    # cfg.l_cl_f = 0.1
    # cfg.l_cl_o = 0.1

    cfg.mixup_p = 0.4
    cfg.mixup_alpha = 1.0
    
    cfg.emb_dim = emb_dim
    
    # checkpoint_fn = r'checkpoints_embednet_only_1024_8e-05_new_loss_new_model_5s/best_model_1025.pt'
    # checkpoint_fn = r'checkpoints_embednet_only_1024_0.0003_new_loss_new_model_5s_retrain2/best_model_68.pt'
    checkpoint_fn = r'checkpoints_embednet_only_1024_0.0001_new_loss_new_model_5s_alignedSimCLR_less_f_o/best_model_214.pt'
    checkpoint = torch.load(checkpoint_fn, map_location=cfg.device)
    with torch.no_grad():
        model_emb.embed_net.scaling_param.set_(checkpoint['embed_net.scaling_param'])
    model_emb.load_state_dict(checkpoint)
    model_emb.eval()

    print('Embedding model loaded')

    models = {'classifier': model, 'embedding': model_emb}

    # checkpoint_fn = r'checkpoints_embed_cl_1024_0.0001_CE_5s_1025/best_model_17.pt'
    # checkpoint_fn = r'checkpoints_embed_cl_1024_0.0001_2_layers_BCE_5s_bm5/best_model_17.pt'
    checkpoint_fn = None
    # checkpoint_fn = r'checkpoints_embed_cl_1024_0.0001_2_layers_BCE_5s_last_nohierch_2022/best_model_53.pt'

    cfg.logger_kwargs['comment'] = f'embed_cl_{cfg.emb_dim}_{cfg.optim_kwargs["lr"]}_{num_of_hidden}_layers_BCE_5s_aligned'

    os.makedirs(cfg.chk_dir+'_'+cfg.logger_kwargs['comment'], exist_ok=True)
    config_dict = {attr: getattr(cfg, attr) for attr in dir(cfg) 
               if not callable(getattr(cfg, attr)) and not attr.startswith("__")}
    with open(os.path.join(cfg.chk_dir+'_'+cfg.logger_kwargs['comment'], 'cfg.json'), 'w') as fp:
        json.dump(config_dict, fp, indent=2)

    trainer = TrainerContrast(models, 
                    criterion, metric,
                    optimizer, 
                    cfg.logger_kwargs, cfg.device, 
                    cfg, checkpoint_fn=checkpoint_fn,
                    coder=coder, only_sources=True)

    
    trainer.train(train_loader, valid_loader)

    # os.system('sudo shutdown now')

if __name__ == '__main__':
    SEED = 42

    set_seed(SEED)
    # try:
    main(SEED)
    # except Exception as e:
        # print(e)
        # os.system('sudo shutdown now')