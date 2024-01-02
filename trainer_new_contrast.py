import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from loss import apply_mixit, snr_loss, si_snr, contrastive_loss, SupConLoss,labeled_contrastive_accuracy, mean_distances
from trainer_separation import Trainer as TrainerSeparation
import tqdm
import re
import math
import numpy as np
import time
import random

class Trainer(TrainerSeparation):
    def __init__(self, model, 
                 criterion, metric,
                 optimizer, 
                 logger_kwargs, 
                 device, 
                 cfg,
                 checkpoint_fn = None,
                 coder = None,
                 only_sources = False,
                 ) -> None:
        self.load_checkpoint = self._load_checkpoint
        super().__init__(model, criterion, metric, optimizer, logger_kwargs, device, cfg, checkpoint_fn)

        assert coder is not None, "class coder must be provided"
        self.coder = coder
        self.only_sources = only_sources        

        if metric is None:
            self.metric = labeled_contrastive_accuracy
        else:
            self.metric = metric
        
        if criterion is None:
            self.contrast_loss = SupConLoss(temperature=0.1, contrast_mode='all', base_temperature=0.1)
        else:
            self.contrast_loss = criterion

    def _load_checkpoint(self, checkpoint_fn):
        self.model.load_state_dict(torch.load(checkpoint_fn), strict=False)
        print(f"Loaded checkpoint from {checkpoint_fn}")

    def _train(self, train_loader, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        cfg = self.cfg # For brevity

        epoch_metric_accum = 0.0 
        epoch_loss = 0.0        
        batch_accum_metric = 0
        batch_accum_loss = 0
        batch_accum_d_pos = 0
        batch_accum_d_neg = 0
        epoch_loss_o = 0.0
        epoch_loss_f = 0.0
        n_views = self.cfg.n_views

        for batch_idx, audio  in enumerate(train_loader):
            if self.only_sources:
                audios, classes = audio
                # audios = audios.squeeze()
                audios = audios.view(-1, self.cfg.frame_size_s*self.cfg.sample_rate)
                # classes = [x[0] for x in classes]

            all_classes = [x for x in classes for _ in range(n_views)]

            all_classes_codes = self.coder.encode(all_classes)
            classes_codes = self.coder.encode(classes)

            b_size = len(classes_codes)

            if self.coder.taxonomy is not None:
                families_all = self.coder.encode_family(self.coder.spec_to_family(all_classes))
                orders_all = self.coder.encode_order(self.coder.spec_to_order(all_classes))
                families = self.coder.encode_family(self.coder.spec_to_family(classes))
                orders = self.coder.encode_order(self.coder.spec_to_order(classes))

                families_all = torch.Tensor(families_all).to(self.device)
                orders_all = torch.Tensor(orders_all).to(self.device)
                families = torch.Tensor(families).to(self.device)
                orders = torch.Tensor(orders).to(self.device)

            audios = audios.unsqueeze(1)
            audios = audios.to(self.device)
            classes_codes = torch.Tensor(classes_codes).to(self.device)
            all_classes_codes = torch.Tensor(all_classes_codes).to(self.device)
            with autocast(enabled=cfg.AMP):
                embs = self.model(audios)
                
                if isinstance(embs, tuple):
                    f = embs[1].view(b_size, n_views, -1)
                    embs = embs[0]
                else:
                    # f = torch.split(embs, [b_size]*n_views, dim=0)
                    # for i in range(n_views):
                    #     f[i] = f[i].unsqueeze(1)
                    f = embs.view(b_size, n_views, -1)

                contrast_mixes_spe = self.contrast_loss(f, classes_codes)
                
                # all_codes = torch.Tensor([x for x in classes_codes for _ in range(n_views)]).to(self.device)

                metric = self.metric(embs.view(b_size, n_views, -1)[:, 0, :], classes_codes)
                mean_pos_dist, mean_neg_dist = mean_distances(embs.view(b_size, n_views, -1)[:, 0, :], classes_codes)

                if self.coder.taxonomy is not None:
                    contrast_mixes_f = self.contrast_loss(f, families)
                    contrast_mixes_o = self.contrast_loss(f, orders)

            if self.coder.taxonomy is not None:
                loss = cfg.l_spe * contrast_mixes_spe + cfg.l_f * contrast_mixes_f + cfg.l_o * contrast_mixes_o
            else:
                loss = contrast_mixes_spe
            

            if batch_idx % cfg.log_freq == 0 or batch_idx == 0:
                # 1. Prepare a single metadata list with all labels
                combined_metadata = []

                flattened_classes = all_classes
                if self.coder.taxonomy is not None:
                    families = self.coder.spec_to_family(flattened_classes)
                    orders = self.coder.spec_to_order(flattened_classes)

                    for i in range(len(flattened_classes)):
                        combined_metadata.append([flattened_classes[i], families[i], orders[i]])
                else:
                    for i in range(len(flattened_classes)):
                        combined_metadata.append([flattened_classes[i]])
                
                # 2. Use self.writer.add_embedding once to save the embeddings with the combined metadata
                self.writer.add_embedding(embs,
                                        global_step=(epoch * len(train_loader) + batch_idx),
                                        tag='Train_Embedding',
                                        metadata=combined_metadata,
                                        metadata_header=['class', 'family', 'order'])
                

            if batch_idx % (cfg.log_freq * 5) == 0 or batch_idx == 0:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}_{batch_idx}.pt')
                torch.save(self.optimizer.state_dict(), f'{self.chk_dir }/optimizer_{epoch}_{batch_idx}.pt')
                
            epoch_loss += loss.item()

            batch_accum_loss += loss.item()
            batch_accum_metric += metric
            batch_accum_d_pos += mean_pos_dist
            batch_accum_d_neg += mean_neg_dist

            epoch_loss_o += contrast_mixes_o.item()
            epoch_loss_f += contrast_mixes_f.item()

            epoch_metric_accum += metric
            # start = time.time()
            self.scaler.scale(loss).backward()

            # print(f"Backward pass took {time.time() - start} seconds")
            if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                batch_accum_metric /= cfg.accumulation_steps
                batch_accum_loss /= cfg.accumulation_steps
                batch_accum_d_pos /= cfg.accumulation_steps
                batch_accum_d_neg /= cfg.accumulation_steps

                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_accum_loss: .4f}, Metric: {batch_accum_metric:.2f}")
                print(f"Mean Positive Distance: {batch_accum_d_pos:.2f}, Mean Negative Distance: {batch_accum_d_neg:.2f}, Mean Difference: {batch_accum_d_neg - batch_accum_d_pos:.2f}")

                self.writer.add_scalar("Training Loss run", batch_accum_loss, global_step=(epoch*len(train_loader) + batch_idx)) 
                self.writer.add_scalar("Training Accuracy run", batch_accum_metric, global_step=(epoch*len(train_loader) + batch_idx)) 
                self.writer.add_scalar("Training Mean Positive Distance", batch_accum_d_pos, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Mean Negative Distance", batch_accum_d_neg, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Mean Difference", batch_accum_d_neg-batch_accum_d_pos, global_step=(epoch*len(train_loader) + batch_idx))
                
                if self.coder.taxonomy is not None:
                    print(f"Loss F: {epoch_loss_f / cfg.accumulation_steps:.4f}, Loss O: {epoch_loss_o / cfg.accumulation_steps:.4f}")
                    self.writer.add_scalar("Training Loss F", epoch_loss_f / cfg.accumulation_steps, global_step=(epoch*len(train_loader) + batch_idx))
                    self.writer.add_scalar("Training Loss O", epoch_loss_o / cfg.accumulation_steps, global_step=(epoch*len(train_loader) + batch_idx))
                    epoch_loss_f = 0.0
                    epoch_loss_o = 0.0

                batch_accum_metric = 0
                batch_accum_loss = 0
                batch_accum_d_pos = 0
                batch_accum_d_neg = 0
                
        torch.cuda.empty_cache()
        return epoch_loss / len(train_loader), epoch_metric_accum/len(train_loader)
    

    def _validate(self, valid_loader, epoch):
        self.model.eval()
        cfg = self.cfg # For brevity
        valid_metric = 0.0
        valid_loss = 0.0           # For accumulating the loss over the epoch
        batch_d_pos = 0
        batch_d_neg = 0
        batch_metric = 0
        batch_loss = 0
        
        with torch.no_grad():
            for batch_idx, audio  in enumerate(valid_loader):
                if self.only_sources:
                    audios, classes = audio
                    # audios = audios.view(-1, self.cfg.frame_size_s*self.cfg.sample_rate)

                    audios = audios.squeeze()
                    # classes = [x[0] for x in classes]

                classes_codes = self.coder.encode(classes)
                b_size = len(classes_codes)

                if self.coder.taxonomy is not None:
                    families = self.coder.encode_family(self.coder.spec_to_family(classes))
                    orders = self.coder.encode_order(self.coder.spec_to_order(classes))
                    families = torch.Tensor(families).to(self.device)
                    orders = torch.Tensor(orders).to(self.device)

                audios = audios.unsqueeze(1)
                audios = audios.to(self.device)
                classes_codes = torch.Tensor(classes_codes).to(self.device)

                with autocast(enabled=cfg.AMP):
                    embs = self.model(audios)

                    if isinstance(embs, tuple):
                        f = embs[1].unsqueeze(1)
                        embs = embs[0]
                    else:
                        f = embs.unsqueeze(1)

                    contrast_mixes_spe = self.contrast_loss(f, classes_codes)
                    metric = self.metric(embs, classes_codes)
                    mean_pos_dist, mean_neg_dist = mean_distances(embs,  classes_codes)

                    # if self.coder.taxonomy is not None:
                    #     contrast_mixes_f = self.contrast_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), families)

                    #     contrast_mixes_o = self.contrast_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), order

                # if self.coder.taxonomy is not None:
                    # loss = cfg.l_spe * contrast_mixes_spe + cfg.l_f * contrast_mixes_f + cfg.l_o * contrast_mixes_o
                # else:
                loss = contrast_mixes_spe
                batch_loss += loss.item()
                valid_loss += loss.item()
                
                valid_metric += metric
                batch_d_pos += mean_pos_dist
                batch_d_neg += mean_neg_dist
                batch_metric += metric

                if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(valid_loader)):
                    print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(valid_loader)}, Loss: {loss: .4f}, Metric: {metric:.2f}")
                    print(f"Mean Positive Distance: {mean_pos_dist:.2f}, Mean Negative Distance: {mean_neg_dist:.2f}, Mean Difference: {mean_neg_dist - mean_pos_dist:.2f}")
                    self.writer.add_scalar("Validation Loss run", batch_loss / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx)) 
                    self.writer.add_scalar("Validation Accuracy run", batch_metric / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx)) 
                    self.writer.add_scalar("Validation Mean Positive Distance", batch_d_pos / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Mean Negative Distance", batch_d_neg / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Mean Difference", (batch_d_neg - batch_d_pos) / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))

                    batch_d_pos = 0
                    batch_d_neg = 0
                    batch_metric = 0
                    batch_loss = 0


                if batch_idx % cfg.log_freq == 0 or batch_idx == 0:
                    # 1. Prepare a single metadata list with all labels
                    combined_metadata = []

                    flattened_classes = classes 
                    if self.coder.taxonomy is not None:
                        families = self.coder.spec_to_family(flattened_classes)
                        orders = self.coder.spec_to_order(flattened_classes)

                        for i in range(len(flattened_classes)):
                            combined_metadata.append([flattened_classes[i], families[i], orders[i]])
                    else:
                        for i in range(len(flattened_classes)):
                            combined_metadata.append([flattened_classes[i]])

                    # 2. Use self.writer.add_embedding once to save the embeddings with the combined metadata
                    self.writer.add_embedding(embs,
                                            global_step=(epoch * len(valid_loader) + batch_idx),
                                            tag='Valid_Embedding',
                                            metadata=combined_metadata,
                                            metadata_header=['class', 'family', 'order'])
                


        return valid_loss / len(valid_loader), valid_metric / len(valid_loader)
    
    def _create_mixes(self, separated_sources):
        mix1 = torch.zeros((self.cfg.batch_size, self.cfg.frame_size_s*self.cfg.sample_rate))
        mix2 = torch.zeros((self.cfg.batch_size, self.cfg.frame_size_s*self.cfg.sample_rate))
        for i in range(self.cfg.batch_size):
            perm = torch.randperm(separated_sources.shape[1])
            for l in range(self.cfg.max_n_birdcalls//2):
                mix1[i,:] += separated_sources[i,perm[l],:]
            for l in range(self.cfg.max_n_birdcalls//2, self.cfg.max_n_birdcalls):
                mix2[i,:] += separated_sources[i,perm[l],:]
        return mix1, mix2
    
    def train(self, train_loader, valid_loader):
        cfg = self.cfg
        best_valid_metric = float('-inf')
        # tqdm_bar = tqdm.tqdm()
        for epoch in range(self.epoch_start, self.epoch_start+cfg.max_epochs):
            train_loss, train_metric = self._train(train_loader, epoch)
            valid_loss, valid_metric = self._validate(valid_loader, epoch)
            
            log_str = f"Epoch {epoch + 1}/{cfg.max_epochs}, Train Loss: {train_loss: .4f}, Train Metric: {train_metric: .4f}, Valid Loss: {valid_loss: .4f}, Valid Metric: {valid_metric: .4f}"
            print(log_str)

            with open(self.log_fn, 'a') as f:
               f.write(log_str + '\n')

            self.writer.add_scalar("Validation Loss", valid_loss, global_step=epoch)        
            self.writer.add_scalar("Validation Metric", valid_metric, global_step=epoch)        
            
            self.writer.add_scalar("Train Loss", train_loss, global_step=epoch)        
            self.writer.add_scalar("Train Metric", train_metric, global_step=epoch)  

            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                torch.save(self.model.state_dict(), f'{self.chk_dir }/best_model_{epoch}.pt')
                torch.save(self.optimizer.state_dict(), f'{self.chk_dir }/best_optimizer_{epoch}.pt')
            else:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}.pt')
                torch.save(self.optimizer.state_dict(), f'{self.chk_dir }/optimizer_{epoch}.pt')

        self.writer.close() 