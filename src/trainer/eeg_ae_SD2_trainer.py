import gc, os
import numpy as np

import torch
from torch import nn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models import eeg_AutoEncoder, Frozen_CLIPImageEmbedder as ImageClip
from trainer.base_trainer import BaseTrainer
from trainer.utils import plot_recon_figures, save
from libs.losses import l1, l2, SignalDiceLoss, ContrastiveLoss
from libs.metric import SignalDice

class EEGAETrainer(BaseTrainer):
    def __init__(self, json_file, sharedFilePath):
        self.json_dict = self.json_load(json_file)
        self.sharedFilePath = sharedFilePath
        self.startEpoch     = 0
        self.globalStep     = 0
        self.eps            = 1e-6
        super().__init__(self.json_dict["ckpt_dir"], self.json_dict["log_dir"], self.json_dict["batch_size"], 
                         sharedFilePath, self.json_dict["num_workers"])


        self.clip = ImageClip()

        self.json_parser()
        self.__checkDirectory__()
        self.losses = {"sdsc":[], "rec":[], "sim":[],"loss":[],  "cos":[]}
        self.accuracy = {"sdsc":[], "mse":[]}
        self.val_losses = {"sdsc":[], "rec":[], "sim":[], "loss":[],  "cos":[]}
        self.val_accuracy = {"sdsc":[], "mse":[]}

        
    def json_parser(self):

        ########################################
        #            Train Setting
        ########################################
        self.name               = self.json_dict["name"]
        self.data_name          = self.json_dict["data_name"]
        self.log_dir            = self.json_dict["log_dir"]
        self.ckpt_dir           = self.json_dict["ckpt_dir"]
        self.eeg_train_path     = self.json_dict["eeg_train_path"]
        self.eeg_test_path      = self.json_dict["eeg_test_path"]
        self.eeg_val_path       = self.json_dict["eeg_val_path"]
        self.img_path           = self.json_dict["img_path"]
        self.split_path         = self.json_dict["split_path"]

        ########################################
        #            Model Setting
        ########################################
        self.in_seq         = self.json_dict["in_seq"]
        self.in_channels    = self.json_dict["in_channels"]
        self.z_channels     = self.json_dict["z_channels"]
        self.out_seq        = self.json_dict["out_seq"]
        self.dims           = self.json_dict["dims"]
        self.shortcut       = bool(self.json_dict["shortcut"])
        self.dropout        = self.json_dict["dropout"]
        self.groups         = self.json_dict["groups"]
        self.layer_mode     = self.json_dict["layer_mode"]
        self.block_mode     = self.json_dict["block_mode"]
        self.down_mode      = self.json_dict["down_mode"]
        self.up_mode        = self.json_dict["up_mode"]
        self.pos_mode       = self.json_dict["pos_mode"]
        self.skip_mode      = self.json_dict["skip_mode"]
        self.sim_mode       = self.json_dict["sim_mode"]
        self.n_layer        = self.json_dict["n_layer"]
        self.n_head         = self.json_dict["n_head"]
        self.dff_factor     = self.json_dict["dff_factor"]
        self.stride         = self.json_dict["stride"]
        self.sdsc_lambda    = self.json_dict["sdsc_lambda"]
        self.sim_lambda     = self.json_dict["sim_lambda"]
        self.img_size       = self.json_dict["img_size"]


        ########################################
        #         Training Parameters
        ########################################
        self.lr        = self.json_dict["learningRate"]
        self.epochs    = self.json_dict["trainEpochs"]
        self.saveIter  = self.json_dict["saveIter"]
        self.validIter = self.json_dict["validIter"]
        self.logIter   = self.json_dict["logIter"]
        self.restart   = self.json_dict["restart"]
    
    def model_define(self, gpu):
        self.MODEL = eeg_AutoEncoder(self.in_seq,  self.in_channels, self.z_channels, self.out_seq,    
                                    self.dims, self.shortcut, self.dropout, self.groups, self.layer_mode, 
                                    self.block_mode, self.down_mode, self.up_mode, self.pos_mode, self.skip_mode, 
                                    self.n_layer, self.n_head, self.dff_factor, self.stride).to(gpu)
        
        self.optim = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.MODEL.parameters()), lr= self.lr, betas=[0.9, 0.999])
        self.sdsc_loss = SignalDiceLoss(False).to(gpu)
        self.cos_loss  = nn.CosineSimilarity(dim=1)
        self.mse  = nn.MSELoss()
        if self.sim_mode == "l2":
            self.sim  = nn.MSELoss()
        elif self.sim_mode == "cos":
            self.sim  = nn.CosineSimilarity()
        elif self.sim_mode == "con":
            self.sim  = ContrastiveLoss()
        else:
            raise AssertionError(f"sim_mode only l2, con or cos, now : {self.sim_mode}")
        
        

        
    def _train(self, gpu, size):
        print(f"Now Initialize Rank: {gpu} | Number Of GPU : {size}")
        self.model_define(gpu)        
        self.initialize(gpu, size)
        torch.cuda.set_device(gpu)

        self.MODEL = nn.SyncBatchNorm.convert_sync_batchnorm(DDP(self.MODEL, find_unused_parameters=False))
        # self.clip  = nn.SyncBatchNorm.convert_sync_batchnorm(DDP(ImageClip('ViT-L/14').to(gpu), find_unused_parameters=True))
        self.clip.model.to(gpu)
        self.makeDatasets(self.eeg_train_path, self.eeg_test_path, self.eeg_val_path, self.img_path, self.img_size)

        if gpu == 0:
            self.makeTensorBoard()
            # with open("./log/my_model.log", "w") as f:
            #     result, _ = summary(self.auto_encoder, input_size=(self.heatmap_size, self.joints, self.imageSize, self.imageSize), batch_size=self.batchSize)
            #     f.write(result)

        print(f"DDP RANK {gpu} RUN...")

        if self.restart:
            dict_model = torch.load(os.path.join(self.ckpt_dir, self.name, f"{self.name}_{self.restart}.pth"))
            self.MODEL.module.load_state_dict(dict_model["net"])
            self.startEpoch = self.restart

        for epoch in range(self.startEpoch, self.epochs):
            self.train_dataset_sampler.set_epoch(epoch),
            self.valid_dataset_sampler.set_epoch(epoch)
            torch.cuda.empty_cache()
            gc.collect()

            for step, data in enumerate(self.loader_train):
                self.MODEL.train()
                self.optim.zero_grad()

                eeg, image, label = data['eeg'], data['image'], data['label']
                eeg = eeg.to(gpu)
                image = image.to(gpu)
                b, c, s = eeg.size()

                
                with torch.autocast(device_type="cuda"):
                    latent, rec = self.MODEL(eeg)
                    clip_embed =  self.clip(image)
                    sim_loss    = self.sim(latent.mean(-1), clip_embed) + self.eps
                    cos_loss    = (1 - self.cos_loss(latent.mean(-1), clip_embed)).mean()
                    rec_loss    = torch.sum(torch.sum(l2(rec, eeg), dim=2)) / (b*c + self.eps)
                    sdsc_loss   = self.sdsc_loss(rec, eeg)
                    loss        =   self.sdsc_lambda * sdsc_loss + self.sim_lambda * sim_loss + cos_loss  + rec_loss

                    mse         = self.mse(rec, eeg)
                    sdsc        = self.sdsc_loss.sdsc(rec, eeg)
                loss.backward()
                nn.utils.clip_grad_norm_(self.MODEL.parameters(), 1.0)
                self.optim.step()
                
                self.losses["sdsc"].append(sdsc_loss.item())
                self.losses["rec"].append(rec_loss.item())
                self.losses["sim"].append(sim_loss.item())
                self.losses["cos"].append(cos_loss.item())
                self.losses["loss"].append(loss.item())

                self.accuracy["mse"].append(mse.item())
                self.accuracy['sdsc'].append(sdsc.item())
                    
                if self.globalStep % self.logIter == 0 and gpu ==0:
                    # LOG Print
                    strings = f"Train Step {self.globalStep} | LOSS {np.mean(self.losses['loss']):.4f} | SDSC LOSS {np.mean(self.losses['sdsc']):.4f} | RECON LOSS {np.mean(self.losses['rec']):.4f} | SIM LOSS {np.mean(self.losses['sim']):.4f} | COSINE LOSS {np.mean(self.losses['cos']):.4f}"
                    strings += f" Accuracy MSE {np.mean(self.accuracy['mse']):.4f} | SDSC {np.mean(self.accuracy['sdsc']):.4f}"
                    print(strings)
                    
                    # Tensorboard logged
                    self.summaryWriter.add_scalar("LOSS", np.mean(self.losses['loss']), self.globalStep)
                    self.summaryWriter.add_scalar("SDSC LOSS", np.mean(self.losses['sdsc']), self.globalStep)
                    self.summaryWriter.add_scalar("SIM LOSS", np.mean(self.losses['sim']), self.globalStep)
                    self.summaryWriter.add_scalar("COS LOSS", np.mean(self.losses["cos"]), self.globalStep)
                    self.summaryWriter.add_scalar("RECON LOSS", np.mean(self.losses['rec']), self.globalStep)

                    self.summaryWriter.add_scalar("MSE", np.mean(self.accuracy['mse']), self.globalStep)
                    self.summaryWriter.add_scalar("SDSC", np.mean(self.accuracy['sdsc']), self.globalStep)

                    # Draw Reconstruction
                    fig = plot_recon_figures(eeg.to('cpu').detach().numpy(), rec.detach().to('cpu').numpy(), self.log_dir, self.globalStep, num_figures=b)

                    self.summaryWriter.add_figure("EEG Recon", fig, self.globalStep)
                    self.losses = {"sdsc":[], "rec":[], "loss":[], "sim":[],  "cos":[]}
                    self.accuracy = {"sdsc":[], "mse":[]}
                
                if self.globalStep % self.validIter == 0:
                    self._valid(gpu)
                    self.checkPoint(gpu)

                self.globalStep += 1

    @torch.no_grad()
    def _valid(self, gpu):
        self.MODEL.eval()
        for step, data in enumerate(self.loader_valid):
            eeg, image, label = data['eeg'], data['image'], data['label']
            eeg = eeg.to(gpu)
            image = image.to(gpu)
            b, _, _ = eeg.size()
                
            with torch.autocast(device_type="cuda"):
                latent, rec = self.MODEL(eeg)
                clip_embed =  self.clip(image)

                sim_loss    = self.sim(latent.mean(-1), clip_embed)
                rec_loss    = torch.mean(torch.sum(l2(rec, eeg), dim=1))
                sdsc_loss   = self.sdsc_loss(rec, eeg)
                cos_loss    = 1 - self.cos_loss(latent.mean(-1), clip_embed).mean()
                loss        =  rec_loss + self.sdsc_lambda * sdsc_loss + self.sim_lambda * sim_loss + cos_loss

                mse         = self.mse(rec, eeg)
                sdsc        = self.sdsc_loss.sdsc(rec, eeg)

            self.val_losses["sdsc"].append(sdsc_loss.item())
            self.val_losses["rec"].append(rec_loss.item())
            self.val_losses["sim"].append(sim_loss.item())
            self.val_losses["cos"].append(cos_loss.item())
            self.val_losses["loss"].append(loss.item())

            self.val_accuracy["mse"].append(mse.item())
            self.val_accuracy['sdsc'].append(sdsc.item())

        if gpu == 0:
            # LOG Print
            strings = f"Valid Step {self.globalStep} | LOSS {np.mean(self.val_losses['loss']):.4f} | SDSC LOSS {np.mean(self.val_losses['sdsc']):.4f} | RECON LOSS {np.mean(self.val_losses['rec']):.4f} | SIM LOSS {np.mean(self.val_losses['sim']):.4f} | COSINE LOSS {np.mean(self.val_losses['cos']):.4f}"
            strings += f" Accuracy MSE {np.mean(self.val_accuracy['mse']):.4f} | SDSC {np.mean(self.val_accuracy['sdsc']):.4f}"
            print(strings)
        
            # Tensorboard logged
            self.summaryWriter.add_scalar("VALID LOSS", np.mean(self.val_losses['loss']), self.globalStep)
            self.summaryWriter.add_scalar("VALID SDSC LOSS", np.mean(self.val_losses['sdsc']), self.globalStep)
            self.summaryWriter.add_scalar("VALID SIM LOSS", np.mean(self.val_losses['sim']), self.globalStep)
            self.summaryWriter.add_scalar("VALID COS LOSS", np.mean(self.val_losses["cos"]), self.globalStep)
            self.summaryWriter.add_scalar("VALID RECON LOSS", np.mean(self.val_losses['rec']), self.globalStep)

            self.summaryWriter.add_scalar("VALID MSE", np.mean(self.val_accuracy['mse']), self.globalStep)
            self.summaryWriter.add_scalar("VALID SDSC", np.mean(self.val_accuracy['sdsc']), self.globalStep)

            # Draw Reconstruction
            fig = plot_recon_figures(eeg.to('cpu').detach().numpy(), rec.to('cpu').detach().numpy(), self.log_dir, self.globalStep, num_figures=8)

            self.summaryWriter.add_figure("VALID EEG Recon", fig, self.globalStep)
            self.val_losses = {"sdsc":[], "rec":[], "loss":[], "sim":[], "cos":[]}
            self.val_accuracy = {"sdsc":[], "mse":[]}


    @torch.no_grad()
    def test(self, folder_name):
        self.losses = {"sdsc":[], "rec":[], "loss":[], "sim":[], "cos":[]}
        self.accuracy = {"sdsc":[], "mse":[]}


        self.makeDatasets(self.eeg_train_path, self.eeg_test_path, self.eeg_val_path, self.img_path, ddp=False)

        self.MODEL = eeg_AutoEncoder(self.in_seq,  self.in_channels, self.z_channels, self.out_seq,    
                                    self.dims, self.shortcut, self.dropout, self.groups, self.layer_mode, 
                                    self.block_mode, self.down_mode, self.up_mode, self.pos_mode, self.skip_mode, 
                                    self.n_layer, self.n_head, self.dff_factor, self.stride).cuda()
        self.sdsc_loss = SignalDiceLoss(False).cuda()

        dict_model = torch.load(os.path.join(self.ckpt_dir, folder_name, f"{self.name}_{self.restart}.pth"))
        self.MODEL.load_state_dict(dict_model["net"])
        self.mse  = nn.MSELoss()

        best_mse = 100
        mse_best_step = 0
        best_sdsc = 0
        sdsc_best_step = 0

        self.MODEL.eval()
        for step, data in enumerate(self.loader_test):
            eeg, label = data
            eeg = eeg.cuda()
            b, _, _ = eeg.size()
                
            
            latent, rec = self.MODEL(eeg)

            rec_loss    = torch.mean(torch.sum(l2(rec, eeg), dim=1)).item()
            sdsc_loss   = self.sdsc_loss(rec, eeg).item()
            loss        =  (rec_loss + self.sdsc_lambda * sdsc_loss)

            mse         = self.mse(rec, eeg).item()
            sdsc        = self.sdsc_loss.sdsc(rec, eeg).item()
            print("step", step, " : ", mse, sdsc)
            

            if best_mse > mse:
                mse_best_step = step
                best_mse = mse
            
            if best_sdsc < sdsc:
                sdsc_best_step = step
                best_sdsc = sdsc

            self.losses["sdsc"].append(sdsc_loss)
            self.losses["rec"].append(rec_loss)
            self.losses["loss"].append(loss)

            self.accuracy["mse"].append(mse)
            self.accuracy['sdsc'].append(sdsc)

        
        # LOG Print
        strings = f"TEST Step {step} | LOSS {np.mean(self.losses['loss']):.4f} | SDSC LOSS {np.mean(self.losses['sdsc']):.4f} | RECON LOSS {np.mean(self.losses['rec']):.4f}"
        strings += f" Accuracy MSE {np.mean(self.accuracy['mse']):.4f} | SDSC {np.mean(self.accuracy['sdsc']):.4f}"
        print(strings)
        print("==================================================================================")
        print(f"BEST MSE : {mse_best_step} step {best_mse} MSE | BEST SDSC {sdsc_best_step} step {best_sdsc}")
        print("==================================================================================")
        with open("./log/result.txt", "w") as  f:
            f.write(strings)
            f.write("\n==================================================================================")
            f.write(f"\nBEST MSE : {mse_best_step} step {best_mse} MSE | BEST SDSC {sdsc_best_step} step {best_sdsc}")
            f.write("\n==================================================================================")

        # Draw Reconstruction
        # fig = plot_recon_figures(eeg.to('cpu').detach().numpy(), rec.to('cpu').detach().numpy(), self.log_dir, self.globalStep, num_figures=8)



    def checkPoint(self, gpu):
        if gpu == 0:
            save(self.ckpt_dir, self.MODEL, self.optim, self.globalStep, self.name)
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(os.path.join(self.ckpt_dir, self.name, f"{self.name}_{self.globalStep}.pth"), map_location=mapLocation, weights_only=True)
        self.MODEL.module.load_state_dict(dict_model["net"])
    
    def runTrain(self):
        gpus = torch.cuda.device_count()
        mp.spawn(self._train, args=(gpus,), nprocs=gpus)