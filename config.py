import torch
import torch.optim as optim

class Config():
    # wav dataset
    sample_rate = 32000
    do_attenuation = True
    do_lp_filter = True
    do_hp_filter = True
    frame_size_s = 3
    max_n_birdcalls = 2
    data_folder = r'birdclef-2021+2023/'

    # spectrum dataset is not used
    win_length_ms = 1.5
    hop_length_ms = win_length_ms / 2
    window_fn = torch.hann_window
    power = 2
    normalized = True
    n_fft = int(win_length_ms*sample_rate/1000)
    n_fft_to_pow2 = True

    device = 'cuda'

    # training parameters
    batch_size = 27
    max_epochs = 20
    AMP = True
    accumulation_steps = 10
    max_norm = 5
    optimizer = optim.Adam
    optim_kwargs = {
                    'lr': 0.0002 * accumulation_steps, 
                    'weight_decay': 1e-5
                    }

    # model parameters according to original paper convtasnet
    N, L, B, H, P, X, R, C = 256, 40, 128, 256, 3, 7, 2, 4 # L 64 is 2 ms with sample rate 32kHz
    enc_dim = N
    enc_ker = L
    conv_in_channels = B
    conv_out_channels = H
    conv_ker = P
    n_blocks = X
    n_repeats = R
    num_sources = C

    # logger
    log_freq = 100
    chk_dir = 'checkpoints'
    log_dir = 'logs'
    logger_kwargs = {
                    'comment': 'TDCNpp_fix_si_snr',
                    }
    
class ConfigRaw():
    # wav dataset
    sample_rate = 32000
    do_attenuation = True
    do_lp_filter = True
    do_hp_filter = True
    frame_size_s = 3
    max_n_birdcalls = 2
    data_folder = r'binary_data/'

    # spectrum dataset is not used
    win_length_ms = 1.5
    n_fft_to_pow2 = True
    hop_length_ms = win_length_ms / 2
    window_fn = torch.hann_window
    power = 2
    normalized = True
    n_fft = int(win_length_ms*sample_rate/1000)
    n_fft_to_pow2 = True
    
    # training parameters
    batch_size = 25
    max_epochs = 20
    AMP = True
    accumulation_steps = 10
    max_norm = 5
    optimizer = optim.Adam
    optim_kwargs = {
                    'lr': 0.0002 * accumulation_steps, 
                    'weight_decay': 1e-5
                    }
    device = 'cuda'
    # model parameters according to original paper convtasnet
    N, L, B, H, P, X, R, C = 256, 40, 128, 256, 3, 7, 2, 4 # L 64 is 2 ms with sample rate 32kHz
    enc_dim = N
    enc_ker = L
    conv_in_channels = B
    conv_out_channels = H
    conv_ker = P
    n_blocks = X
    n_repeats = R
    num_sources = C

    # logger
    log_freq = 100
    chk_dir = 'checkpoints'
    log_dir = 'logs'
    logger_kwargs = {
                    'comment': 'TDCNpp_binary_data_5',
                    }