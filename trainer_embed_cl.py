import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from loss import apply_mixit, snr_loss, si_snr, contrastive_loss, SupConLoss,labeled_contrastive_accuracy, mean_distances
from trainer_separation import Trainer as TrainerSeparation
from torchmetrics import Accuracy
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
        self.save_graph = self._save_graph
        super().__init__(model['classifier'], criterion, metric, optimizer, logger_kwargs, device, cfg, checkpoint_fn)
        
        self.model = model['classifier']
        self.model_emb = model['embedding']

        assert coder is not None, "class coder must be provided"
        self.coder = coder
        self.only_sources = only_sources        

        self.metric2 = Accuracy(task="multiclass", num_classes=len(coder), ignore_index=len(coder), threshold=0.1).to(cfg.device)
        self.contrast_loss = SupConLoss(temperature=0.1, contrast_mode='all', base_temperature=0.1)

        if coder.taxonomy is not None:
            self.metric_o = Accuracy(task="multiclass", num_classes=len(self.coder.taxonomy['ORDER1'].dropna().unique())+1, threshold=0.1).to(cfg.device)
            self.metric_f = Accuracy(task="multiclass", num_classes=len(self.coder.taxonomy['FAMILY'].dropna().unique())+1, threshold=0.1).to(cfg.device)

    def _load_checkpoint(self, checkpoint_fn):
        self.model.load_state_dict(torch.load(checkpoint_fn), strict=False)
        print(f"Loaded checkpoint from {checkpoint_fn}")

    def _save_graph(self):
        self.writer.add_graph(self.model, torch.zeros((1, self.cfg.emb_dim)).to(self.device))

    def encode_to_bce(self, targets, num_classes, encode=None):
        # batch_size = targets.shape[0]
        batch_size = len(targets)
        encoded = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            indices = encode(targets[i])
            encoded[i, indices] = 1
        return encoded.float()


    def _train(self, train_loader, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        cfg = self.cfg # For brevity

        epoch_metric_accum = 0.0 
        epoch_loss = 0.0        
        batch_accum_metric = 0
        batch_accum_loss = 0
        batch_accum_m_o = 0
        batch_accum_m_f = 0

        batch_accum_d_pos = 0
        batch_accum_d_neg = 0
        batch_accum_contrast = 0

        epoch_loss_o = 0.0
        epoch_loss_f = 0.0
        batch_accuracy = 0.0

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
                # families = self.coder.encode_family(self.coder.spec_to_family(classes))
                # orders = self.coder.encode_order(self.coder.spec_to_order(classes))

                families_all_bce = self.encode_to_bce(self.coder.spec_to_family(all_classes), len(self.coder.taxonomy['FAMILY'].dropna().unique())+1, self.coder.encode_family)
                orders_all_bce = self.encode_to_bce(self.coder.spec_to_order(all_classes), len(self.coder.taxonomy['ORDER1'].dropna().unique())+1, self.coder.encode_order)

                families_all = torch.Tensor(families_all).to(self.device).long()
                orders_all = torch.Tensor(orders_all).to(self.device).long()
                # families = torch.Tensor(families).to(self.device)
                # orders = torch.Tensor(orders).to(self.device)

                families_all_bce = torch.Tensor(families_all_bce).to(self.device).long()
                orders_all_bce = torch.Tensor(orders_all_bce).to(self.device).long()

            audios = audios.unsqueeze(1)
            audios = audios.to(self.device)
            classes_codes = torch.Tensor(classes_codes).to(self.device)
            
            all_classes_codes_bce = self.encode_to_bce(all_classes, len(self.coder), self.coder.encode)
            all_classes_codes_bce = torch.Tensor(all_classes_codes_bce).to(self.device).long()
            all_classes_codes_bce0 = all_classes_codes_bce.to(self.device)

            all_classes_codes = torch.Tensor(all_classes_codes).to(self.device).long()

            # mixup augmentation
            if torch.rand(1).cpu().item() < cfg.mixup_p:
                # select random examples in the batch
                idx = torch.randperm(b_size*cfg.n_views)
                
                # compute mixup weight
                lam = torch.distributions.beta.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample().cpu().item()
                
                # compute mixup inputs and targets
                audios = lam * audios + (1 - lam) * audios[idx]
                all_classes_codes_bce = lam * all_classes_codes_bce0 + (1 - lam) * all_classes_codes_bce0[idx]
                families_all_bce = lam * families_all_bce + (1 - lam) * families_all_bce[idx]
                orders_all_bce = lam * orders_all_bce + (1 - lam) * orders_all_bce[idx]
                del idx,lam
                
            with autocast(enabled=cfg.AMP):
                
                with torch.no_grad():
                    embs = self.model_emb(audios)
                if isinstance(embs, tuple):
                    f = embs[1].view(b_size, n_views, -1)
                    embs = embs[0]
                else:
                    f = embs.view(b_size, n_views, -1)

                logit = self.model(embs)
                
                # if self.coder.taxonomy is not None:
                if isinstance(logit, dict):
                    logit_s = logit['species']
                    logit_f = logit['family']
                    logit_o = logit['order']
                    logit = logit_s

                # loss_s = self.criterion(logit, all_classes_codes)
                loss_s = self.criterion(logit, all_classes_codes_bce.float())
                accu = self.metric2(logit, all_classes_codes)

                # f = embs.view(b_size, n_views, -1)
                contrast_mixes_spe = self.contrast_loss(f, classes_codes)
                
                metric = self.metric(logit, all_classes_codes_bce0)
                mean_pos_dist, mean_neg_dist = mean_distances(embs, all_classes_codes)

                # if self.coder.taxonomy is not None:
                if isinstance(logit, dict):
                    # loss_f = self.criterion(logit_f, families_all)
                    # loss_o = self.criterion(logit_o, orders_all)
                    loss_f  = self.criterion(logit_f, families_all_bce.float())
                    loss_o  = self.criterion(logit_o, orders_all_bce.float())

                    metric_f = self.metric_f(logit_f, families_all)
                    metric_o = self.metric_o(logit_o, orders_all)
                    


            # if self.coder.taxonomy is not None:
            if isinstance(logit, dict):
                loss = cfg.l_spe * loss_s + cfg.l_f * loss_f + cfg.l_o * loss_o
            else:
                loss = loss_s
            

            if batch_idx % cfg.log_freq == 0 or batch_idx == 0:
                # 1. Prepare a single metadata list with all labels
                combined_metadata = []

                flattened_classes = all_classes
                # if self.coder.taxonomy is not None:
                if isinstance(logit, dict):
                    families = self.coder.spec_to_family(flattened_classes)
                    orders = self.coder.spec_to_order(flattened_classes)

                    for i in range(len(flattened_classes)):
                        combined_metadata.append([flattened_classes[i], families[i], orders[i]])
                else:
                    for i in range(len(flattened_classes)):
                        combined_metadata.append([flattened_classes[i]])
                
                # 2. Use self.writer.add_embedding once to save the embeddings with the combined metadata
                if isinstance(logit, dict):
                    metadata_header = ['class', 'family', 'order']
                else:
                    metadata_header = None

                self.writer.add_embedding(embs,
                                        global_step=(epoch * len(train_loader) + batch_idx),
                                        tag='Train_Embedding',
                                        metadata=combined_metadata,
                                        metadata_header=metadata_header)
                

            if batch_idx % (cfg.log_freq * 5) == 0 or batch_idx == 0:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}_{batch_idx}.pt')
                
            epoch_loss += loss.detach().cpu().cpu().item()

            batch_accum_loss += loss.detach().cpu().item()
            batch_accum_metric += metric.detach().cpu().item()
            batch_accuracy += accu.detach().cpu().item()

            batch_accum_contrast += contrast_mixes_spe.cpu().item()
            batch_accum_d_pos += mean_pos_dist
            batch_accum_d_neg += mean_neg_dist


            # if self.coder.taxonomy is not None:
            if isinstance(logit, dict):
                batch_accum_m_o += metric_o.detach().cpu().item()
                batch_accum_m_f += metric_f.detach().cpu().item()

                epoch_loss_o += loss_o.detach().cpu().item()
                epoch_loss_f += loss_f.detach().cpu().item()

            epoch_metric_accum += metric.detach().cpu().item()
            # start = time.time()
            if cfg.AMP:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # print(f"Backward pass took {time.time() - start} seconds")
            if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                if cfg.AMP:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_norm)
                if cfg.AMP:
                    self.scaler.step(self.optimizer)
                else:
                    self.optimizer.step()
                if cfg.AMP:
                    self.scaler.update()
                self.optimizer.zero_grad()

                batch_accum_metric /= cfg.accumulation_steps
                batch_accum_loss /= cfg.accumulation_steps
                batch_accuracy /= cfg.accumulation_steps
                batch_accum_contrast /= cfg.accumulation_steps
                batch_accum_d_pos /= cfg.accumulation_steps
                batch_accum_d_neg /= cfg.accumulation_steps

                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_accum_loss: .4f}, Metric: {batch_accum_metric:.2f}")
                print(f"Accuracy: {batch_accuracy:.2f}")
                print(f"Contrast: {batch_accum_contrast:.2f}")
                print(f"Mean Positive Distance: {batch_accum_d_pos:.2f}, Mean Negative Distance: {batch_accum_d_neg:.2f}, Mean Difference: {batch_accum_d_neg - batch_accum_d_pos:.2f}")

                self.writer.add_scalar("Training Loss run", batch_accum_loss, global_step=(epoch*len(train_loader) + batch_idx)) 
                self.writer.add_scalar("Training Metric run", batch_accum_metric, global_step=(epoch*len(train_loader) + batch_idx)) 
                self.writer.add_scalar("Training Accuracy run", batch_accuracy, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Contrast run", batch_accum_contrast, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Mean pos dist run", batch_accum_d_pos, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Mean neg dist run", batch_accum_d_neg, global_step=(epoch*len(train_loader) + batch_idx))
                self.writer.add_scalar("Training Mean Difference", batch_accum_d_neg-batch_accum_d_pos, global_step=(epoch*len(train_loader) + batch_idx))


                # if self.coder.taxonomy is not None:
                if isinstance(logit, dict):
                    batch_accum_m_o /= cfg.accumulation_steps
                    batch_accum_m_f /= cfg.accumulation_steps
                    print(f"Loss F: {epoch_loss_f / cfg.accumulation_steps:.4f}, Loss O: {epoch_loss_o / cfg.accumulation_steps:.4f}")
                    print(f"Metric F: {batch_accum_m_f:.2f}, Metric O: {batch_accum_m_o:.2f}")
                    self.writer.add_scalar("Training Loss F", epoch_loss_f / cfg.accumulation_steps, global_step=(epoch*len(train_loader) + batch_idx))
                    self.writer.add_scalar("Training Loss O", epoch_loss_o / cfg.accumulation_steps, global_step=(epoch*len(train_loader) + batch_idx))
                    self.writer.add_scalar("Training Accuracy F", batch_accum_m_f, global_step=(epoch*len(train_loader) + batch_idx))
                    self.writer.add_scalar("Training Accuracy O", batch_accum_m_o, global_step=(epoch*len(train_loader) + batch_idx))
                    epoch_loss_f = 0.0
                    epoch_loss_o = 0.0
                    batch_accum_m_o = 0
                    batch_accum_m_f = 0

                batch_accum_metric = 0
                batch_accum_loss = 0
                batch_accuracy = 0.0
                batch_accum_contrast = 0
                batch_accum_d_pos = 0
                batch_accum_d_neg = 0
                
            del families_all, orders_all, families_all_bce, orders_all_bce
            del all_classes_codes_bce0, all_classes_codes
            del audios
            del loss, logit, embs, f

            torch.cuda.empty_cache()
        return epoch_loss / len(train_loader), epoch_metric_accum/len(train_loader)
    

    def _validate(self, valid_loader, epoch):
        self.model.eval()
        cfg = self.cfg # For brevity
        valid_metric = 0.0
        valid_loss = 0.0           # For accumulating the loss over the epoch
        batch_metric = 0
        batch_loss = 0
        batch_accum_contrast = 0
        batch_accum_d_pos = 0
        batch_accum_d_neg = 0
        batch_accuracy = 0.0

        with torch.no_grad():
            for batch_idx, audio  in enumerate(valid_loader):
                if self.only_sources:
                    audios, classes = audio
                    # audios = audios.view(-1, self.cfg.frame_size_s*self.cfg.sample_rate)

                    audios = audios.squeeze()
                    # classes = [x[0] for x in classes]

                classes_codes = self.coder.encode(classes)
                # b_size = len(classes_codes)

                # if self.coder.taxonomy is not None:
                #     families = self.coder.encode_family(self.coder.spec_to_family(classes))
                #     orders = self.coder.encode_order(self.coder.spec_to_order(classes))
                #     families = torch.Tensor(families).to(self.device)
                #     orders = torch.Tensor(orders).to(self.device)

                audios = audios.unsqueeze(1)
                audios = audios.to(self.device)
                classes_codes_bce = self.encode_to_bce(classes, len(self.coder), self.coder.encode)
                classes_codes_bce = torch.Tensor(classes_codes_bce).to(self.device).long()
                classes_codes = torch.Tensor(classes_codes).to(self.device).long()
                with autocast(enabled=cfg.AMP):

                    embs = self.model_emb(audios)
                    if isinstance(embs, tuple):
                        f = embs[1].unsqueeze(1)
                        embs = embs[0]
                    else:
                        f = embs.unsqueeze(1)
                
                    logit = self.model(embs)

                    # if self.coder.taxonomy is not None:
                    #     logit_s = logit['species']
                    #     logit_f = logit['family']
                    #     logit_o = logit['order']
                    #     logit = logit_s
                    contrast_mixes_spe = self.contrast_loss(f, classes_codes)
                    # loss_s = self.criterion(logit, classes_codes)
                    loss_s = self.criterion(logit, classes_codes_bce.float())
                    metric = self.metric(logit, classes_codes_bce)
                    accu = self.metric2(logit, classes_codes)
                    mean_pos_dist, mean_neg_dist = mean_distances(embs,  classes_codes)

                    # if self.coder.taxonomy is not None:
                    #     # contrast_mixes_f = self.contrast_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), families)
                    #     # contrast_mixes_o = self.contrast_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), order)
                    #     loss_f = self.criterion(logit_f, families)
                    #     loss_o = self.criterion(logit_o, orders)
                    #     metric_f = self.metric_f(logit_f, families)
                    #     metric_o = self.metric_o(logit_o, orders)

                # if self.coder.taxonomy is not None:
                #     loss = cfg.l_spe * loss_s + cfg.l_f * loss_f + cfg.l_o * loss_o
                #     batch_loss_f += loss_f.cpu().item()
                #     batch_loss_o += loss_o.cpu().item()
                #     batch_metric_f += metric_f
                #     batch_metric_o += metric_o
                # else:
                loss = loss_s
                
                batch_loss += loss.detach().cpu().item()
                valid_loss += loss.detach().cpu().item()

                batch_accuracy += accu.detach().cpu().item()
                valid_metric += metric.detach().cpu().item()
                batch_metric += metric.detach().cpu().item()

                batch_accum_contrast += contrast_mixes_spe.detach().cpu().item()
                batch_accum_d_pos += mean_pos_dist
                batch_accum_d_neg += mean_neg_dist

                if (batch_idx+1) % cfg.accumulation_steps == 0 or (batch_idx + 1 == len(valid_loader)):
                    print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Batch {batch_idx+1}/{len(valid_loader)}, Loss: {loss: .4f}, Metric: {metric:.2f}")
                    print(f"Accuracy: {accu:.2f}")
                    print(f"Contrast: {batch_accum_contrast:.2f}")
                    print(f"Mean Positive Distance: {batch_accum_d_pos:.2f}, Mean Negative Distance: {batch_accum_d_neg:.2f}, Mean Difference: {batch_accum_d_neg - batch_accum_d_pos:.2f}")
                    self.writer.add_scalar("Validation Loss run", batch_loss / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx)) 
                    self.writer.add_scalar("Validation Metric run", batch_metric / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx)) 
                    self.writer.add_scalar("Validation Accuracy run", batch_accuracy / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Contrast run", batch_accum_contrast / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Mean pos dist run", batch_accum_d_pos / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Mean neg dist run", batch_accum_d_neg / cfg.accumulation_steps , global_step=(epoch*len(valid_loader) + batch_idx))
                    self.writer.add_scalar("Validation Mean Difference", batch_accum_d_neg-batch_accum_d_pos, global_step=(epoch*len(valid_loader) + batch_idx))

                    batch_accum_contrast = 0
                    batch_accum_d_pos = 0
                    batch_accum_d_neg = 0

                    batch_metric = 0
                    batch_loss = 0
                    batch_accuracy = 0.0

                    # if self.coder.taxonomy is not None:
                    #     print(f"Loss F: {batch_loss_f / cfg.accumulation_steps:.4f}, Loss O: {batch_loss_o / cfg.accumulation_steps:.4f}")
                    #     print(f"Metric F: {batch_metric_f / cfg.accumulation_steps:.2f}, Metric O: {batch_metric_o / cfg.accumulation_steps:.2f}")
                    #     self.writer.add_scalar("Validation Loss F", batch_loss_f / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    #     self.writer.add_scalar("Validation Loss O", batch_loss_o / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    #     self.writer.add_scalar("Validation Metric F", batch_metric_f / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    #     self.writer.add_scalar("Validation Metric O", batch_metric_o / cfg.accumulation_steps, global_step=(epoch*len(valid_loader) + batch_idx))
                    #     batch_loss_f = 0
                    #     batch_loss_o = 0
                    #     batch_metric_f = 0
                    #     batch_metric_o = 0

                if batch_idx % cfg.log_freq == 0 or batch_idx == 0:
                    # 1. Prepare a single metadata list with all labels
                    combined_metadata = []

                    flattened_classes = classes 
                    # if self.coder.taxonomy is not None:
                    if isinstance(logit, dict):
                        families = self.coder.spec_to_family(flattened_classes)
                        orders = self.coder.spec_to_order(flattened_classes)

                        for i in range(len(flattened_classes)):
                            combined_metadata.append([flattened_classes[i], families[i], orders[i]])
                    else:
                        for i in range(len(flattened_classes)):
                            combined_metadata.append([flattened_classes[i]])

                    # 2. Use self.writer.add_embedding once to save the embeddings with the combined metadata
                    if isinstance(logit, dict):
                        metadata_header = ['class', 'family', 'order']
                    else:
                        metadata_header = None

                    self.writer.add_embedding(embs,
                                            global_step=(epoch * len(valid_loader) + batch_idx),
                                            tag='Valid_Embedding',
                                            metadata=combined_metadata,
                                            metadata_header=metadata_header)
                


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
            else:
                torch.save(self.model.state_dict(), f'{self.chk_dir }/model_{epoch}.pt')

        self.writer.close() 