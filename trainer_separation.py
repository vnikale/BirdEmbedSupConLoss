import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from loss import apply_mixit, snr_loss, si_snr

import tqdm
import re

class Trainer():
    def __init__(self, model, 
                 criterion, metric,
                 optimizer, 
                 logger_kwargs, 
                 device, 
                 cfg,
                 checkpoint_fn = None,
                 ) -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.logger_kwargs = logger_kwargs
        self.device = device
        self.cfg = cfg
        self.criterion = criterion
        self.metric = metric
        self.writer = SummaryWriter(**logger_kwargs)
        self.save_graph()
        self.checkpoint_fn = checkpoint_fn
        self.epoch_start = 0

        if checkpoint_fn is not None:
            self.load_checkpoint(checkpoint_fn)

            match = re.search(r'(?<=best_model_)\d+', checkpoint_fn)
            if match is None:
                match = re.search(r'(?<=_)\d+(?=_)', checkpoint_fn)
                if match is None:
                    match = re.search(r'(?<=_)\d+', checkpoint_fn)
            self.epoch_start = int(match.group(0)) 

        if 'comment' in logger_kwargs:
            if not os.path.exists(cfg.log_dir+'_txt'):
                os.makedirs(cfg.log_dir+'_txt')
            self.log_fn = os.path.join(cfg.log_dir+'_txt','train_log_'+logger_kwargs['comment']+'.txt')
        else:
            raise ValueError("Please provide a comment in logger_kwargs")

        self.chk_dir = cfg.chk_dir + '_' + logger_kwargs['comment']
        if not os.path.exists(self.chk_dir):
            os.makedirs(self.chk_dir)
        
        if cfg.AMP:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None


    def load_checkpoint(self, checkpoint_fn):
        self.model.load_state_dict(torch.load(checkpoint_fn))
        print(f"Loaded checkpoint from {checkpoint_fn}")

    def save_graph(self):
        self.writer.add_graph(self.model, torch.zeros((1, 1, self.cfg.frame_size_s*self.cfg.sample_rate)).to(self.device))

    def _train(self, train_loader, epoch):
        self.model.train()
        cfg = self.cfg # For brevity
        epoch_si_snr_accum = 0.0   # For accumulating the SI-SNR over the epoch
        epoch_loss = 0.0           # For accumulating the loss over the epoch
        self.optimizer.zero_grad()
        batch_accum_si_snr = 0
        batch_accum_loss = 0
        for batch_idx, audio  in enumerate(train_loader):
            # mix1, mix2 = self._create_mixes(separated_sources)
            separated_sources, mix1, mix2, classes = audio

            if len(separated_sources.shape) > 3:
                separated_sources = separated_sources.squeeze()
            if len(mix1.shape) > 2 or len(mix2.shape) > 2:
                mix1 = mix1.squeeze()
                mix2 = mix2.squeeze()

            mix_of_mixes = mix1 + mix2
            mix_of_mixes = mix_of_mixes.unsqueeze(1)
            mix_of_mixes = mix_of_mixes.to(self.device)

            mixes = torch.stack([mix1, mix2], dim=1)
            mixes = mixes.to(self.device)
            mix1 = mix1.to(self.device)
            mix2 = mix2.to(self.device)

            with autocast(enabled=cfg.AMP):
                estimated_sources = self.model(mix_of_mixes)
                loss, estimated_mixes, _ = apply_mixit(snr_loss, mixes, estimated_sources)
                loss = torch.mean(loss.float())
                si_snr_value = torch.mean(si_snr(mixes, estimated_mixes).float())
                epoch_si_snr_accum += si_snr_value.item()
                batch_accum_si_snr += si_snr_value.item()
                batch_accum_loss += loss.item()
            
            loss = loss / cfg.accumulation_steps

            if torch.isnan(estimated_sources.float().mean()).any() or torch.isnan(separated_sources.float().mean()).any():
                print(torch.isnan(separated_sources.mean()).any())
                print("Error: NaNs in estimated_sources or separated_sources!")
                # os.system("sudo shutdown")
                raise ValueError("Error: NaNs in estimated_sources or separated_sources!")
            
            if batch_idx % cfg.log_freq == 0 or batch_idx == 0:
                # Writing audio to TensorBoard
                for i in range(min(estimated_sources.shape[0], 2)): # Write the first 2 samples from the batch
                    self.writer.add_audio(f'Audio/Original_Mix_of_Mix_{i}', mix_of_mixes[i], global_step=(epoch*len(train_loader) + batch_idx), sample_rate=cfg.sample_rate)
                    self.writer.add_audio(f'Audio/Estimated_Mix_of_Mix_{i}', estimated_mixes[i,0,:]+estimated_mixes[i,1,:], global_step=(epoch*len(train_loader) + batch_idx), sample_rate=cfg.sample_rate)
                    self.writer.add_audio(f'Audio/Estimated_Mix0_{i}', estimated_mixes[i,0,:], global_step=(epoch*len(train_loader) + batch_idx), sample_rate=cfg.sample_rate)
                    self.writer.add_audio(f'Audio/Estimated_Mix1_{i}', estimated_mixes[i,1,:], global_step=(epoch*len(train_loader) + batch_idx), sample_rate=cfg.sample_rate)

                    for j in range(estimated_sources.shape[1]):
                        self.writer.add_audio(f'Audio/Estimated_Source_{i}_{j}', estimated_sources[i,j], global_step=(epoch*len(train_loader) + batch_idx), sample_rate=cfg.sample_rate)
            
            if batch_idx % (cfg.log_freq * 5) == 0 or batch_idx == 0:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}_{batch_idx}.pt')

            epoch_loss += loss.item()
            self.scaler.scale(loss).backward()

            if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                batch_accum_si_snr /= cfg.accumulation_steps
                batch_accum_loss /= cfg.accumulation_steps

                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_accum_loss}, SI-SNR: {batch_accum_si_snr}")
                # self.writer.add_scalar("Training Loss run", loss.item(), global_step=(epoch*len(train_loader) + batch_idx)) # Writing loss to tensorboard
                # self.writer.add_scalar("Training SI-SNR run", si_snr_value.item(), global_step=(epoch*len(train_loader) + batch_idx)) # Writing SI-SNR to tensorboard
                self.writer.add_scalar("Training Loss run", batch_accum_loss, global_step=(epoch*len(train_loader) + batch_idx)) # Writing loss to tensorboard
                self.writer.add_scalar("Training SI-SNR run", batch_accum_si_snr, global_step=(epoch*len(train_loader) + batch_idx)) # Writing SI-SNR to tensorboard

                batch_accum_si_snr = 0
                batch_accum_loss = 0
        
        return epoch_loss / len(train_loader), epoch_si_snr_accum/len(train_loader)
    

    def _validate(self, valid_loader, epoch):
        self.model.eval()
        cfg = self.cfg # For brevity
        valid_si_snr_accum = 0.0   # For accumulating the SI-SNR over the epoch
        valid_loss = 0.0           # For accumulating the loss over the epoch
        with torch.no_grad():
            for batch_idx, audio  in enumerate(valid_loader):
                # mix1, mix2 = self._create_mixes(separated_sources)
                separated_sources, mix1, mix2, classes = audio

                if len(separated_sources.shape) > 3:
                    separated_sources = separated_sources.squeeze()
                if len(mix1.shape) > 2 or len(mix2.shape) > 2:
                    mix1 = mix1.squeeze()
                    mix2 = mix2.squeeze()

                mix_of_mixes = mix1 + mix2
                mix_of_mixes = mix_of_mixes.unsqueeze(1)
                mixes = torch.stack([mix1, mix2], dim=1)
                mix_of_mixes = mix_of_mixes.to(self.device)
                mixes = mixes.to(self.device)
                mix1 = mix1.to(self.device)
                mix2 = mix2.to(self.device)
                
                with autocast(enabled=cfg.AMP):
                    estimated_sources = self.model(mix_of_mixes)
                    loss, estimated_mixes, _ = apply_mixit(snr_loss, mixes, estimated_sources)
                    loss = torch.mean(loss)
                    si_snr_value = torch.mean(si_snr(mixes, estimated_mixes))
                    valid_si_snr_accum += si_snr_value.item()

                if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(valid_loader)):
                    print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(valid_loader)},Validation Loss: {loss.item()}, ValidationSI-SNR: {si_snr_value.item()}")
                    self.writer.add_scalar("Validation Loss run", loss.item(), global_step=(epoch*len(valid_loader) + batch_idx)) # Writing loss to tensorboard
                    self.writer.add_scalar("Validation SI-SNR run", si_snr_value.item(), global_step=(epoch*len(valid_loader) + batch_idx)) # Writing SI-SNR to tensorboard

                valid_loss += loss.item()
        
        return valid_loss / len(valid_loader), valid_si_snr_accum / len(valid_loader)
    
    def _create_mixes(self, separated_sources):
        mix1 = torch.zeros((self.cfg.batch_size, self.cfg.frame_size_s*self.cfg.sample_rate)).to(self.device)
        mix2 = torch.zeros((self.cfg.batch_size, self.cfg.frame_size_s*self.cfg.sample_rate)).to(self.device)
        for i in range(self.cfg.batch_size):
            perm = torch.randperm(separated_sources.shape[1])
            for l in range(self.cfg.num_sources//2):
                mix1[i,:] += separated_sources[i,perm[l],:]
            for l in range(self.cfg.num_sources//2, self.cfg.num_sources):
                mix2[i,:] += separated_sources[i,perm[l],:]
        return mix1, mix2
    

    def train(self, train_loader, valid_loader):
        cfg = self.cfg
        best_valid_si_snr = float('-inf')
        # tqdm_bar = tqdm.tqdm()
        for epoch in range(self.epoch_start, self.epoch_start+cfg.max_epochs):
            train_loss, train_si_snr = self._train(train_loader, epoch)
            valid_loss, valid_si_snr = self._validate(valid_loader, epoch)
            
            log_str = f"Epoch {epoch + 1}/{cfg.max_epochs}, Train Loss: {train_loss}, Train SI-SNR: {train_si_snr}, Valid Loss: {valid_loss}, Valid SI-SNR: {valid_si_snr}"
            print(log_str)
            # tqdm_bar.set_description(log_str)
            # tqdm_bar.refresh()

            with open(self.log_fn, 'a') as f:
               f.write(log_str + '\n')

            self.writer.add_scalar("Validation Loss", valid_loss, global_step=epoch)        
            self.writer.add_scalar("Validation SI-SNR", valid_si_snr, global_step=epoch)        
            
            self.writer.add_scalar("Train Loss", train_loss, global_step=epoch)        
            self.writer.add_scalar("Train SI-SNR", train_si_snr, global_step=epoch)  

            if valid_si_snr > best_valid_si_snr:
                best_valid_si_snr = valid_si_snr
                torch.save(self.model.state_dict(), f'{self.chk_dir }/best_model_{epoch}.pt')
                torch.save(self.optimizer.state_dict(), f'{self.chk_dir }/best_opt_{epoch}.pt')
            else:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}.pt')
                torch.save(self.optimizer.state_dict(), f'{self.chk_dir }/opt_{epoch}.pt')


        self.writer.close()