import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

from torch.utils.data import Dataset
from collections import defaultdict

import random
import os
import torch.nn as nn
import numpy as np

def audio_norm(audio, type='linear'):
    if type == 'linear':
        audio = (audio-torch.min(audio))/(torch.max(audio)-torch.min(audio)+1e-5)*2-1
    return audio

def audio_resample(audio, orig_sr, target_sr):
    return F.resample(audio, orig_sr, target_sr) 

def audio_pad(audio, target_size):
    pad_s = target_size - audio.numel()
    return nn.functional.pad(audio, (pad_s, 0))

# memory inefficient    
def get_loudest_segment_start_index(audio, frame_size):
    windowed_amplitude = audio.unfold(dimension=0, size=frame_size, step=1).square().mean(dim=1)
    
    loudest_start_index = windowed_amplitude.argmax()
    
    return loudest_start_index

def get_loudest_segment_start_index_conv( audio, frame_size):
    kernel = torch.ones(1, 1, frame_size)
    
    audio = audio.abs().unsqueeze(0).unsqueeze(0)
    conv_result = torch.nn.functional.conv1d(audio, kernel)
    loudest_start_index = conv_result[0, 0].argmax().item()
    
    return loudest_start_index

def get_loudest_segment_start_index_stft(audio, frame_size, sr=32000, low_freq_threshold=650):
    n_fft=2**int(np.log2(frame_size//20))
    hop_length = n_fft // 2
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, onesided=True).abs()
    freqs = torch.linspace(0, sr / 2, steps=n_fft // 2 + 1)

    # Select bins where the frequency is greater than the low_freq_threshold
    selected_bins = freqs > low_freq_threshold
    energy = torch.sum(stft[selected_bins,:]**2, axis=0)
    max_energy_frame = torch.argmax(energy)

    # Determine the region in the original audio to search for the peak
    start_idx = max(0, max_energy_frame * hop_length - frame_size // 2)
    end_idx = min(len(audio), start_idx + frame_size)

    center_idx = (start_idx + end_idx) // 2
    adjusted_start = max(0, center_idx - frame_size // 2)
    adjusted_end = adjusted_start + frame_size

    if adjusted_end > len(audio):
        # Adjust the window if it goes beyond the audio length
        adjusted_end = len(audio)
        adjusted_start = adjusted_end - frame_size

    if adjusted_start < 0:
        # If audio is shorter than frame size
        adjusted_start = 0

    return adjusted_start


def audio_crop(audio, crop_size, sr, type='random', frac = 3):
    if type == 'random':
        start = np.random.randint(0, audio.numel() - crop_size + 1)

    if type == 'loudest':
        if audio.numel() >= crop_size*frac:
            start0 = np.random.randint(0, audio.numel() - crop_size*frac + 1)
            audio_crop = audio[start0:start0 + crop_size*frac]
            start = get_loudest_segment_start_index_stft(audio_crop, crop_size, sr=sr)
        else:
            start = np.random.randint(0, audio.numel() - crop_size + 1)
    
    return audio[start:start + crop_size] 

class BirdcallMixsWavsDataset(Dataset):
    '''
    Dataset for birdcall mixs
    cfg: config object
    audio_files: list of audio files
    classes: list of classes for each audio file
    crop_type: 'random' or 'loudest',loudest takes eternity to perform
    '''
    def __init__(self, audio_files: list, cfg, classes:list = None, crop_type='random', only_negative=True, frac=3):

        self.data_folder = cfg.data_folder
        self.audio_files = audio_files
        self.classes = None
        self.frac = frac
        self.only_negative = only_negative

        if classes is not None:
            assert len(classes) == len(audio_files), 'classes and audio_files must have the same length'
            self.classes = np.array(classes)
            # Group audio files by class
            class_to_files = defaultdict(list)
            for idx, class_ in enumerate(self.classes):
                class_to_files[class_].append(idx)
            self.class_to_files = class_to_files
        
        self.max_n_birdcalls  = cfg.max_n_birdcalls
        self.sample_rate = cfg.sample_rate
        self.do_attenuation = cfg.do_attenuation
        self.atten_prob = cfg.atten_prob if 'atten_prob' in cfg.__dir__() else 0.6

        self.do_lp_filter = cfg.do_lp_filter
        self.lp_prob = cfg.lp_prob if 'lp_prob' in cfg.__dir__() else 0.6

        self.do_hp_filter = cfg.do_hp_filter
        self.frame_size_s = cfg.frame_size_s

        self.AMP = cfg.AMP

        self.crop_type = crop_type if 'crop_type' not in cfg.__dir__() else cfg.crop_type
        assert self.crop_type in ['random', 'loudest'], 'crop_type must be either random or loudest'

    def __len__(self):
        return len(self.audio_files)
    

    def _create_mixes(self, separated_sources):
        # Create zero tensors for the mixes
        mix1 = torch.zeros(self.frame_size_s * self.sample_rate)
        mix2 = torch.zeros(self.frame_size_s * self.sample_rate)
        
        perm = torch.randperm(separated_sources.shape[0])
        
        for l in range(self.max_n_birdcalls // 2):
            mix1 += separated_sources[perm[l], :]
        
        for l in range(self.max_n_birdcalls // 2, self.max_n_birdcalls):
            mix2 += separated_sources[perm[l], :]
            
        return mix1, mix2
    
    def _transform(self, audio):
        # random volume attenuation
        if self.do_attenuation and random.random() <= self.atten_prob:
            attenuation = random.random()
        else:
            attenuation = 1
        audio = audio*attenuation   
        
        # random low pass filter
        if self.do_lp_filter and random.random() <= self.lp_prob: 
            max_cutoff = self.sample_rate/2-100
            freq_fraq = random.random()*0.2 
            freq_fraq = 1 - (attenuation + freq_fraq)/2
            freq = freq_fraq * max_cutoff
            if abs(freq) > 1e-2:
                audio = F.lowpass_biquad(audio, self.sample_rate, cutoff_freq=freq, Q = 0.707)
                
        return audio

    def _preprocess(self, audio, sr, frame_size):
        audio = audio.squeeze()

        if len(audio.shape) == 2:
            audio = torch.mean(audio, dim=0) # stereo to mono
            
        # resample if required
        if sr != self.sample_rate: 
            audio = audio_resample(audio, sr, self.sample_rate)
        if audio.numel() < frame_size:
            audio = audio_pad(audio, frame_size)
        else:
            audio = audio_crop(audio, frame_size, sr = self.sample_rate, type = self.crop_type, frac=self.frac)
            
        audio = audio_norm(audio)

        audio = self._transform(audio)

        return audio

    def __getitem__(self, idx):
        # audio_files_shrink = self.audio_files[idx:]
        audio_files_shrink = self.audio_files
        # if len(audio_files_shrink) < self.max_n_birdcalls:
        #         audio_files_shrink = self.audio_files[idx-self.max_n_birdcalls:]

        # Randomly choose N birdcalls
        # n_birdcalls = random.randint(2, self.max_n_birdcalls)
        n_birdcalls = self.max_n_birdcalls
        # n_birdcalls = min(self.max_n_birdcalls, len(audio_files_shrink))

        # sample classes
        if self.classes is not None:
            chosen_indices = []
            chosen_classes = []
            if self.only_negative:
                # Sample different classes
                chosen_classes = random.sample(self.class_to_files.keys(), n_birdcalls)
                for class_ in chosen_classes:
                    chosen_indices.append(random.choice(self.class_to_files[class_]))
            else:
                # Sample from the same class
                chosen_class = random.choice(list(self.class_to_files.keys()))
                if len(self.class_to_files[chosen_class]) < n_birdcalls:
                    # print(f"Not enough samples available for class {chosen_class}")
                    # print(f'Copying samples from class {chosen_class} to reach {n_birdcalls} samples')
                    while len(chosen_indices) < n_birdcalls:
                        chosen_indices.append(random.choice(self.class_to_files[chosen_class]))
                else:
                    chosen_indices = random.sample(self.class_to_files[chosen_class], n_birdcalls)
                chosen_classes = [chosen_class]*n_birdcalls
        else:
            chosen_classes = None

        separated_audio = []
        frame_size_s = self.frame_size_s
        frame_size = frame_size_s*self.sample_rate
        
        for index in chosen_indices:
            try:
                audio_fn = os.path.join(self.data_folder, audio_files_shrink[index])
                audio, sr = torchaudio.load(audio_fn, normalize=True)
            except RuntimeError:
                print(f'Error reading file {audio_fn}')
                # chosen_indices.append(random.randint(0, len(audio_files_shrink))) # wrong because of classes
                # continue
                exit()
            audio = self._preprocess(audio, sr, frame_size)
    
            separated_audio.append(audio)

        separated_audio = torch.stack(separated_audio)

        # remove DC
        if self.do_hp_filter: 
            audio = F.highpass_biquad(separated_audio, self.sample_rate, cutoff_freq=500, Q = 0.707)

        mix1, mix2 = self._create_mixes(separated_audio)
        
        if self.AMP:
            return separated_audio.half(), mix1.half(), mix2.half(), chosen_classes
        
        return separated_audio, mix1, mix2, chosen_classes

from collections import OrderedDict
import threading

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()  # Lock for thread safety

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            else:
                # Move the accessed item to the end
                self.cache.move_to_end(key)
                return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # If item exists, move it to the end
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.capacity:
                # Remove the first item (Least Recently Used)
                self.cache.popitem(last=False)
            self.cache[key] = value


class BirdcallWavsDatasetSimpleCached(BirdcallMixsWavsDataset):
    def __init__(self, audio_files: list, cfg, classes:list = None, crop_type='loudest', transform=None, frac=3, cache_size=100):
        super().__init__(audio_files, cfg, classes, crop_type, False, frac)
        if cache_size > 0:
            self.cache = LRUCache(cache_size)
        else: 
            self.cache = None
        self.cfg = cfg
        if transform is not None:
            self._transform = transform
    
    def set_transform(self, transform):
        self._transform = transform

    def _preprocess(self, audio, frame_size):
        if audio.dtype == torch.float16:
            audio = audio.float()

        if audio.numel() < frame_size:
            audio = audio_pad(audio, frame_size)
        else:
            audio = audio_crop(audio, frame_size, sr = self.sample_rate, type = self.crop_type, frac=self.frac)
        audio = audio_norm(audio)
        audio = self._transform(audio)

        return audio

    def __prepreprocess(self, audio, sr):
        audio = audio.squeeze()
        # stereo to mono
        if len(audio.shape) == 2:
            audio = torch.mean(audio, dim=0, keepdim=False) 
            
        # resample if required
        if sr != self.sample_rate: 
            audio = audio_resample(audio, sr, self.sample_rate)
        return audio

    def __getitem__(self, idx):
        audio_fn = os.path.join(self.data_folder, self.audio_files[idx])
        if self.cache is not None:
            audio = self.cache.get(audio_fn)
            if audio is None:
                audio, sr = torchaudio.load(audio_fn, normalize=True)
                audio = self.__prepreprocess(audio, sr)
                if self.cfg.AMP: 
                    audio = audio.half()
                self.cache.put(audio_fn, audio)
        else:
            audio, sr = torchaudio.load(audio_fn, normalize=True)
            audio = self.__prepreprocess(audio, sr)
        audio = self._preprocess(audio, self.frame_size_s*self.sample_rate)
        return audio, self.classes[idx]


class BirdcallWavsDatasetSimple(BirdcallMixsWavsDataset):
    def __init__(self, audio_files: list, cfg, classes:list = None, crop_type='loudest', transform=None, frac=3):
        super().__init__(audio_files, cfg, classes, crop_type, False, frac)

        if transform is not None:
            self._transform = transform
    
    def set_transform(self, transform):
        self._transform = transform

    def __getitem__(self, idx):
        audio_fn = os.path.join(self.data_folder, self.audio_files[idx])
        audio, sr = torchaudio.load(audio_fn, normalize=True)
        audio = self._preprocess(audio, sr, self.frame_size_s*self.sample_rate)
        return audio, self.classes[idx]



import pandas as pd

class TestDataset(Dataset):
    def __init__(self, data_folder = None, annotations=None, test_size = 5000, transform=None, species=None, cfg=None, center_frame=False):
        '''
        data_folder: path to the data folder
        annotations: pandas dataframe with the annotations
        test_size: number of samples to use, int or 'all'
        transform: transform to apply to the data
        species: pandas dataframe with the species codes
        cfg: config
        center_frame: if True, the frame is centered around the annotation, otherwise it is cropped as [start_t : end_t]
        '''
        assert data_folder is not None, 'data_folder must be set'
        assert annotations is not None, 'annotations must be set'

        self.annotations = annotations
        self.test_size = test_size
        self.transform = transform
        self.data_folder = data_folder
        self.sampled = False
        self.submission_f = None
        self.species = species
        self.center_frame = center_frame

        self.audio_loaded_fn = None
        self.cfg = cfg

    def _to_submission_format(self):
        assert self.species is not None, 'species must be set'

        species_codes = self.species['Species eBird Code'].values.tolist()

        def process_row(row):
            """Process a row from the DataFrame to match desired format."""
            name = row['Filename']
            name = os.path.splitext(name)[0]  # Remove file extension
            start_t = row['Start Time (s)']
            end_t = row['End Time (s)']
            name = f"{name}_{start_t}_{end_t}"
            output = [name]  

            for species in species_codes:
                if row['Species eBird Code'] == species:
                    output.append(1)  
                else:
                    output.append(0)
            return output

        output_data = [process_row(row) for _, row in self.data.iterrows()]
        output_df = pd.DataFrame(output_data, columns=['row_id'] + species_codes)
        # output_df.to_csv('test_data_solution.csv', index=False)
        self.submission_f = output_df
        return output_df
    
    def __len__(self):
        return self.test_size if self.test_size != 'all' else len(self.annotations)
    
    def _sample(self):
        if self.sampled:
            return
        
        if self.test_size != 'all':
            self.data = self.annotations.sample(self.test_size)
            self.sampled = True
        else:
            self.data = self.annotations
            self.sampled = True

        self._to_submission_format()
    
    def _preprocess(self, audio, sr, frame_size):
        audio = audio.squeeze()

        # stereo to mono
        if len(audio.shape) == 2:
            audio = torch.mean(audio, dim=0, keepdim=False) 
            
        # resample if required
        if sr != self.cfg.sample_rate: 
            audio = audio_resample(audio, sr, self.cfg.sample_rate)

        if audio.numel() < frame_size:
            audio = audio_pad(audio, frame_size)
            
        audio = audio_norm(audio)
        return audio
    
    def __getitem__(self, idx):
        if not self.sampled:
            raise Exception('Dataset not sampled, call _sample() first')
        row = self.data.iloc[idx]
        fn = row['Filename']
        name = os.path.splitext(fn)[0]  # Remove file extension
        start_t = row['Start Time (s)']
        end_t = row['End Time (s)']
        name = f"{name}_{start_t}_{end_t}"

        # load audio if not loaded
        if self.audio_loaded_fn != fn:
            audio, sr = torchaudio.load(os.path.join(self.data_folder, fn))
            self.audio_loaded_fn = fn
            audio = self._preprocess(audio, sr, self.cfg.frame_size_s*self.cfg.sample_rate)
            self.audio_loaded = audio
        else:
            sr = self.cfg.sample_rate
        
        # get audio slice
        start = int(start_t * sr)
        end = int(end_t * sr)

        # Center the audio slice or adjust if shorter than frame_size
        if end - start < self.cfg.frame_size_s * sr or self.center_frame:
            start = int(start_t * sr - self.cfg.frame_size_s * sr / 2)
            end = int(start_t * sr + self.cfg.frame_size_s * sr / 2)

        # Check boundaries and pad if necessary
        if start < 0 or end > len(self.audio_loaded):
            # Compute how much padding is required on each side
            pad_left = max(0, -start)
            pad_right = max(0, end - len(self.audio_loaded))
            
            # Pad the audio
            self.audio_loaded = torch.nn.functional.pad(self.audio_loaded, (pad_left, pad_right))
            
            # Adjust the start and end after padding
            start += pad_left
            end += pad_right

        audio_slice = self.audio_loaded[start:end]
        if self.transform is not None:
            audio_slice_transformed = self.transform(audio_slice)
        else:
            audio_slice_transformed = None
        
        return audio_slice,audio_slice_transformed, row['Species eBird Code'], name



class BirdcallMixsSpectrumDataset(BirdcallMixsWavsDataset):
    def __init__(self, metadata, cfg):
        super().__init__(metadata, cfg)
        
        n_fft = cfg.n_fft

        if cfg.n_fft_to_pow2:
            n_fft = np.ceil(np.log(n_fft)/np.log(2))
            n_fft = pow(2, int(n_fft)) 

        self.transform = T.Spectrogram(n_fft = n_fft,
                                        win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                        hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                        window_fn = cfg.window_fn,
                                        power = cfg.power,
                                        normalized = cfg.normalized 
                                        )
        if cfg.power is not None:
            self.itransform = T.GriffinLim(n_fft = n_fft,
                                            win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                            hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                            window_fn = cfg.window_fn,
                                            power = cfg.power,
                                            length = cfg.sample_rate*cfg.frame_size_s
                                            )
        else:
            self.itransform = T.InverseSpectrogram(n_fft = n_fft,
                                    win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                    hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                    window_fn = cfg.window_fn,
                                    )
        self.cfg = cfg

    def __getitem__(self, idx):
        separated_audio, mix1, mix2, chosen_classes = super().__getitem__(idx)    

        separated_spec = self.transform(separated_audio)
        mix1_spec = self.transform(mix1)
        mix2_spec = self.transform(mix2)

        return separated_spec, mix1_spec, mix2_spec, chosen_classes


    def _itransform(self, spec):
        return self.itransform(spec)
    

class BinaryBirdcallDataset(Dataset):
    def __init__(self, files, batch_size, only_sources=False, transform=None):
        self.batch_size = batch_size
        self.files = files
        # Calculate total length without loading all the data
        sample_file = torch.load(self.files[0])
        self.samples_per_file = len(sample_file['separated_audio'])
        self.total_samples = self.samples_per_file * len(self.files)
        self.only_sources = only_sources
        self.transform = transform

    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, idx):
        # samples = []
        separated_sources_samples = []
        if not self.only_sources:
            mix1_samples = []
            mix2_samples = []
        class_samples = []

        while len(separated_sources_samples) < self.batch_size:
            # Calculate which file to load
            file_idx = (idx * self.batch_size + len(separated_sources_samples)) // self.samples_per_file
            within_file_idx = (idx * self.batch_size + len(separated_sources_samples)) % self.samples_per_file
            
            data_list = torch.load(self.files[file_idx])
            
            while within_file_idx < len(data_list['separated_audio']) and len(separated_sources_samples) < self.batch_size:
                separated = data_list['separated_audio'][within_file_idx]
                if self.transform is not None:
                    for i in range(separated.shape[0]):
                        separated[i] = self.transform(separated[i])
                separated_sources_samples.append(separated)
                
                if not self.only_sources:
                    mix1_samples.append(data_list['mix1'][within_file_idx])
                    mix2_samples.append(data_list['mix2'][within_file_idx])
                class_samples.append((data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]))
        
                within_file_idx += 1

        separated_sources_tensor = torch.stack(separated_sources_samples)
        if not self.only_sources:
            mix1_tensor = torch.stack(mix1_samples)
            mix2_tensor = torch.stack(mix2_samples)
        if not self.only_sources:
            return separated_sources_tensor, mix1_tensor.squeeze(), mix2_tensor.squeeze(), class_samples
        return separated_sources_tensor, class_samples
    

class BinaryBirdcallDatasetContrast(Dataset):
    def __init__(self, files_pos, files_neg, batch_size, only_sources=False):
        self.batch_size = batch_size
        self.only_sources = only_sources

        self.files_pos = files_pos
        self.files_neg = files_neg
        # Calculate total length without loading all the data
        sample_file_pos = torch.load(self.files_pos[0])
        self.samples_per_file_pos = len(sample_file_pos['separated_audio'])
        self.total_samples_pos = self.samples_per_file_pos * len(self.files_pos)

        sample_file_neg = torch.load(self.files_neg[0])
        self.samples_per_file_neg = len(sample_file_neg['separated_audio'])
        self.total_samples_neg = self.samples_per_file_neg * len(self.files_neg)

        assert self.total_samples_pos == self.total_samples_neg, 'files_pos and files_neg must have the same length'

    def __len__(self):
        return self.total_samples_pos // self.batch_size
    
    def set_pos(self, pos):
        self.pos = pos

    def __getitem__(self, idx):
        self.set_pos(True if idx % 2 == 0 else False)

        if self.pos:
            files = self.files_pos
            samples_per_file = self.samples_per_file_pos
        else:
            files = self.files_neg
            samples_per_file = self.samples_per_file_neg

        separated_sources_samples = []
        if not self.only_sources:
            mix1_samples = []
            mix2_samples = []
        class_samples = []

        while len(separated_sources_samples) < self.batch_size:
            # Calculate which file to load
            file_idx = (idx * self.batch_size + len(separated_sources_samples)) // samples_per_file
            within_file_idx = (idx * self.batch_size + len(separated_sources_samples)) % samples_per_file
            
            data_list = torch.load(files[file_idx])
            
            while within_file_idx < len(data_list['separated_audio']) and len(separated_sources_samples) < self.batch_size:
                separated_sources_samples.append(data_list['separated_audio'][within_file_idx])
                if not self.only_sources:
                    mix1_samples.append(data_list['mix1'][within_file_idx])
                    mix2_samples.append(data_list['mix2'][within_file_idx])
                class_samples.append((data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]))
        
                within_file_idx += 1

        separated_sources_tensor = torch.stack(separated_sources_samples)
        if not self.only_sources:
            mix1_tensor = torch.stack(mix1_samples)
            mix2_tensor = torch.stack(mix2_samples)
        if not self.only_sources:
            return separated_sources_tensor, mix1_tensor.squeeze(), mix2_tensor.squeeze(), class_samples
        return separated_sources_tensor, class_samples, self.pos
    

class BinaryBirdcallDatasetToSpectrum(Dataset):
    def __init__(self, files, batch_size, cfg, only_sources=True):

        n_fft = cfg.n_fft
        n_mels = cfg.n_mels if 'n_mels' in cfg.__dir__() else 128

        if cfg.n_fft_to_pow2:
            n_fft = np.ceil(np.log(n_fft)/np.log(2))
            n_fft = pow(2, int(n_fft)) 

        if cfg.s_type == 'mel':
            self.to_spec = T.MelSpectrogram(n_fft = n_fft,
                                        n_mels=n_mels,
                                        win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                        hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                        window_fn = cfg.window_fn,
                                        normalized = cfg.normalized 
                                        )
            self.from_spec = None
        else:
            self.to_spec = T.Spectrogram(n_fft = n_fft,
                                            win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                            hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                            window_fn = cfg.window_fn,
                                            power = cfg.power,
                                            normalized = cfg.normalized 
                                            )
            self.from_spec = T.GriffinLim(n_fft = n_fft,
                                            win_length = int(cfg.win_length_ms*cfg.sample_rate/1000),
                                            hop_length = int(cfg.hop_length_ms*cfg.sample_rate/1000),
                                            window_fn = cfg.window_fn,
                                            power = cfg.power,
                                            length = cfg.sample_rate*cfg.frame_size_s
                                            )

        self.cfg = cfg
        self.batch_size = batch_size
        self.files = files
        # Calculate total length without loading all the data
        sample_file = torch.load(self.files[0])
        self.samples_per_file = len(sample_file['separated_audio'])
        self.total_samples = self.samples_per_file * len(self.files)
        self.only_sources = only_sources

    def __len__(self):
        return self.total_samples // self.batch_size

    def transform(self, audio):
        if audio.dtype == torch.float16:
            x = T.AmplitudeToDB()(self.to_spec(audio.float()))
        else:
            x = T.AmplitudeToDB()(self.to_spec(audio))
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x

    def __getitem__(self, idx):
        # samples = []
        separated_sources_samples = []
        mix1_samples = []
        mix2_samples = []
        class_samples = []

        while len(separated_sources_samples) < self.batch_size:
            # Calculate which file to load
            file_idx = (idx * self.batch_size + len(separated_sources_samples)) // self.samples_per_file
            within_file_idx = (idx * self.batch_size + len(separated_sources_samples)) % self.samples_per_file
            
            data_list = torch.load(self.files[file_idx])
            
            while within_file_idx < len(data_list['mix1']) and len(separated_sources_samples) < self.batch_size:
                separated_sources_samples.append(data_list['separated_audio'][within_file_idx])
                if not self.only_sources:
                    mix1_samples.append(data_list['mix1'][within_file_idx])
                    mix2_samples.append(data_list['mix2'][within_file_idx])
                class_samples.append((data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]))
        
                within_file_idx += 1

        separated_sources_tensor = torch.stack(separated_sources_samples)
        if not self.only_sources:
            mix1_tensor = torch.stack(mix1_samples)
            mix2_tensor = torch.stack(mix2_samples)

        separated_sources_tensor = self.transform(separated_sources_tensor)
        if not self.only_sources:
            mix1_tensor = self.transform(mix1_tensor)
            mix2_tensor = self.transform(mix2_tensor)

        if not self.only_sources:
            return separated_sources_tensor, mix1_tensor.squeeze(), mix2_tensor.squeeze(), class_samples
        return separated_sources_tensor, class_samples
    

class BinaryBirdcallDatasetContrastRandom(Dataset):
    def __init__(self, files_pos, files_neg, batch_size, r=0.5, only_sources=False):
        self.batch_size = batch_size
        self.r = r
        self.only_sources = only_sources

        self.files_pos = files_pos
        self.files_neg = files_neg

        # Calculate total length without loading all the data
        sample_file_pos = torch.load(self.files_pos[0])
        self.samples_per_file_pos = len(sample_file_pos['separated_audio'])

        sample_file_neg = torch.load(self.files_neg[0])
        self.samples_per_file_neg = len(sample_file_neg['separated_audio'])

        assert len(self.files_pos) == len(self.files_neg), 'files_pos and files_neg must have the same length'

    def __len__(self):
        return len(self.files_pos)

    def __getitem__(self, idx):
        num_pos = int(self.batch_size * self.r)
        num_neg = self.batch_size - num_pos

        # Function to load data either positive or negative
        def load_data(files, num_samples):
            samples = []
            mix1 = []
            mix2 = []
            classes = []

            # Randomly select a file
            file_idx = random.choice(range(len(files)))
            data_list = torch.load(files[file_idx])

            # Randomly select a sample from the chosen file until we meet our quota
            while len(samples) < num_samples:
                within_file_idx = random.choice(range(len(data_list['separated_audio'])))
                samples.append(data_list['separated_audio'][within_file_idx])
                if not self.only_sources:
                    mix1.append(data_list['mix1'][within_file_idx])
                    mix2.append(data_list['mix2'][within_file_idx])
                classes.append((data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]))

            return samples, mix1, mix2, classes

        pos_samples, pos_mix1, pos_mix2, pos_classes = load_data(self.files_pos, num_pos)
        neg_samples, neg_mix1, neg_mix2, neg_classes = load_data(self.files_neg, num_neg)

        separated_sources_samples = pos_samples + neg_samples
        separated_sources_tensor = torch.stack(separated_sources_samples)

        if not self.only_sources:
            mix1_samples = pos_mix1 + neg_mix1
            mix2_samples = pos_mix2 + neg_mix2

            mix1_tensor = torch.stack(mix1_samples)
            mix2_tensor = torch.stack(mix2_samples)
            
            return separated_sources_tensor, mix1_tensor.squeeze(), mix2_tensor.squeeze(), pos_classes + neg_classes

        return separated_sources_tensor, pos_classes + neg_classes

class BinaryBirdcallDatasetRandom(Dataset):
    def __init__(self, files, batch_size, only_sources=False):
        self.batch_size = batch_size
        self.files = files
        # Calculate total length without loading all the data
        sample_file = torch.load(self.files[0])
        self.samples_per_file = len(sample_file['separated_audio'])
        self.only_sources = only_sources

    def __len__(self):
        return len(self.files) * self.samples_per_file // self.batch_size

    def __getitem__(self, idx):
        separated_sources_samples = []
        if not self.only_sources:
            mix1_samples = []
            mix2_samples = []
        class_samples = []

        while len(separated_sources_samples) < self.batch_size:
            # Randomly select a file
            file_idx = random.choice(range(len(self.files)))
            data_list = torch.load(self.files[file_idx])

            # Randomly select a sample from the chosen file until we meet our quota
            while len(separated_sources_samples) < self.batch_size:
                within_file_idx = random.choice(range(len(data_list['separated_audio'])))
                separated_sources_samples.append(data_list['separated_audio'][within_file_idx])
                
                if not self.only_sources:
                    mix1_samples.append(data_list['mix1'][within_file_idx])
                    mix2_samples.append(data_list['mix2'][within_file_idx])
                class_samples.append((data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]))

        separated_sources_tensor = torch.stack(separated_sources_samples)
        if not self.only_sources:
            mix1_tensor = torch.stack(mix1_samples)
            mix2_tensor = torch.stack(mix2_samples)
            return separated_sources_tensor, mix1_tensor.squeeze(), mix2_tensor.squeeze(), class_samples

        return separated_sources_tensor, class_samples


class BinaryBirdcallDatasetAugment(Dataset):
    def __init__(self, files, batch_size, aug_ratio=0, transform=None, sample_rate = 32000):
        '''
        files: is a list of files
        aug_ratio: is the ratio of augmented samples to non-augmented samples
        transform: is the transform to apply to the augmented samples
        '''
        if aug_ratio > 0:
            assert transform is not None, 'transform must be provided if aug_ratio > 0'
        assert aug_ratio < 1, 'aug_ratio must be less than 1 to provide 1 sample for augmentation'

        self.batch_size = batch_size
        self.files = files
        self.sample_rate = sample_rate

        # Calculate total length without loading all the data
        sample_file = torch.load(self.files[0])
        self.samples_per_file = len(sample_file['separated_audio'])
        self.total_samples = self.samples_per_file * len(self.files)
        self.aug_ratio = aug_ratio
        self.not_augmented_size = int(self.batch_size * (1 - self.aug_ratio))
        self.augmented_size = self.batch_size - self.not_augmented_size

        self.transform = transform


    def __len__(self):
        return self.total_samples // self.not_augmented_size

    def __getitem__(self, idx):
        separated_sources_samples = []
        class_samples = []

        while len(separated_sources_samples)*2 < self.not_augmented_size:
            # Calculate which file to load
            file_idx = (idx * self.not_augmented_size + len(separated_sources_samples)) // self.samples_per_file
            within_file_idx = (idx * self.not_augmented_size + len(separated_sources_samples)) % self.samples_per_file
            
            data_list = torch.load(self.files[file_idx])
            
            while within_file_idx < len(data_list['separated_audio']) and len(separated_sources_samples)*2 < self.not_augmented_size:
                separated = data_list['separated_audio'][within_file_idx]
                separated_sources_samples.append(separated)
                classes = [data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]]
                class_samples.append(classes)
                within_file_idx += 1

        
        separated_sources_tensor = torch.stack(separated_sources_samples)
        separated_sources_tensor = separated_sources_tensor.view(-1, separated_sources_tensor.shape[-1])
        class_samples_np = np.concatenate(class_samples)

        if separated_sources_tensor.shape[0] != self.not_augmented_size:
            separated_sources_tensor = separated_sources_tensor[:-1]
            class_samples_np = class_samples_np[:-1]
            
        if self.augmented_size > 0:
            augmented_samples = 0
            # Loop until we have the desired number of augmented samples
            while augmented_samples < self.augmented_size:
                num_to_augment = min(self.augmented_size - augmented_samples, self.not_augmented_size)
                augmented_data = separated_sources_tensor[:num_to_augment].clone().float()
                augmented = self.transform(augmented_data.numpy(), sample_rate=self.sample_rate)
                augmented = torch.from_numpy(augmented)
                separated_sources_tensor = torch.cat((separated_sources_tensor, augmented), dim=0)
                class_samples_np = np.concatenate((class_samples_np, class_samples_np[:num_to_augment]), axis=0)
                augmented_samples += num_to_augment            

        return separated_sources_tensor, class_samples_np
    
class BinaryBirdcallDatasetAugmentMultiView(Dataset):
    def __init__(self, files, batch_size, transform=None, do_augment = True, sample_rate = 32000, mixup_p=0.4):
        '''
        files: is a list of files
        transform: is the transform to apply to the augmented samples
        '''
        if do_augment:
            assert transform is not None, 'transform must be provided '

        self.batch_size = batch_size
        self.files = files
        self.sample_rate = sample_rate
        self.mixup_p = mixup_p
        self.do_augment = do_augment
        # Calculate total length without loading all the data
        sample_file = torch.load(self.files[0])
        self.samples_per_file = len(sample_file['separated_audio'])
        self.total_samples = self.samples_per_file * len(self.files)

        self.transform = transform


    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, idx):
        separated_sources_samples = []
        class_samples = []

        while len(separated_sources_samples)*2 < self.batch_size:
            # Calculate which file to load
            file_idx = (idx * self.batch_size + len(separated_sources_samples)) // self.samples_per_file
            within_file_idx = (idx * self.batch_size + len(separated_sources_samples)) % self.samples_per_file
            
            data_list = torch.load(self.files[file_idx])
            
            while within_file_idx < len(data_list['separated_audio']) and len(separated_sources_samples)*2 < self.batch_size:
                separated = data_list['separated_audio'][within_file_idx]
                classes = [data_list['chosen_classes'][0][within_file_idx], data_list['chosen_classes'][1][within_file_idx]]
                class_samples.append(classes)

                # random mixup
                if random.random() > 1-self.mixup_p and self.do_augment:
                    separated[0] = separated[0] + (random.random()*0.5+0.15) * separated[1]

                separated_sources_samples.append(separated)
                within_file_idx += 1
        
        separated_sources_tensor = torch.stack(separated_sources_samples)
        separated_sources_tensor = separated_sources_tensor.view(-1, separated_sources_tensor.shape[-1])
        class_samples_np = np.concatenate(class_samples)

        if separated_sources_tensor.shape[0] != self.batch_size:
            separated_sources_tensor = separated_sources_tensor[:-1]
            class_samples_np = class_samples_np[:-1]

        augmented_data0 = separated_sources_tensor.clone().float()

        if self.do_augment:
            augmented_data1 = separated_sources_tensor.clone().float()
            augmented0 = self.transform(augmented_data0.numpy(), sample_rate=self.sample_rate)
            augmented1 = self.transform(augmented_data1.numpy(), sample_rate=self.sample_rate)
            augmented0 = torch.from_numpy(augmented0)
            augmented1 = torch.from_numpy(augmented1)
            return torch.cat((augmented0, augmented1), dim=0), class_samples_np.tolist()
        return augmented_data0, class_samples_np.tolist()
    

class BirdcallWavsDatasetMultiviewSimple(BirdcallMixsWavsDataset):
    def __init__(self, audio_files: list, cfg, classes:list = None, crop_type='loudest', transform_aug=None, n_views = 2, frac=3, do_augment = False, keep_original=False):
        super().__init__(audio_files, cfg, classes, crop_type, False, frac)
        self.transform_aug = transform_aug
        self.n_views = n_views
        self.do_augment = do_augment
        self._transform = lambda x: x
        self.keep_original = keep_original

    # def set_transform(self, transform):
    #     self._transform = transform

    def __getitem__(self, idx):
        audio_fn = os.path.join(self.data_folder, self.audio_files[idx])
        audio, sr = torchaudio.load(audio_fn, normalize=True)
        audio = self._preprocess(audio, sr, self.frame_size_s*self.sample_rate)
        augmeted = []
        
        if self.do_augment:
            for _ in range(self.n_views - int(self.keep_original)):
                augmeted.append(self.transform_aug(audio))
            if self.keep_original:
                augmeted.append(audio)
        else:
            augmeted.append(audio)

        return torch.stack(augmeted, dim=0), self.classes[idx]