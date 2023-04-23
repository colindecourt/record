import os
import numpy as np
import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
from utils.loss import SmoothCELoss
from datasets.cruw.collate_functions import cr_collate
from evaluation.postprocess import post_process_single_frame_cruw, write_dets_results_single_frame
from cruw.eval import evaluate_rodnet_seq
from cruw.eval.rod.rod_eval_utils import accumulate, summarize


class CruwExecutor(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config_dict, cruw_dataset_obj, save_dir):
        """
        PyTorch lightning base class for training models on CRUW datasets.
        @param model: instance of the model to train
        @param train_dataset: training dataset
        @param val_dataset: validation dataset
        @param config_dict: dictionary with training configuration (lr, optimizer, path to data etc.)
        @param cruw_dataset_obj: CRUW dataset object
        @param save_dir: directory to save data
        """
        super(CruwExecutor, self).__init__()
        
        self.cruw_dataset_obj = cruw_dataset_obj
        self.config = config_dict
        self.train_cfg = config_dict['train_cfg']
        self.radar_cfg = cruw_dataset_obj.sensor_cfg.radar_cfg
        self.model_cfg = config_dict['model_cfg']
        self.n_class = self.cruw_dataset_obj.object_cfg.n_class
        self.batch_size = self.train_cfg['batch_size']
        self.learning_rate = self.train_cfg['lr'] 
        self.in_channels = self.model_cfg['in_channels']
        self.win_size = self.train_cfg['win_size']
        self.model_name = self.model_cfg['name']

        hp_dict = {'model_cfg': config_dict['model_cfg'],
                   'train_cfg': config_dict['train_cfg']}
        self.save_hyperparameters(hp_dict)

        # Model
        self.model = model
        self.loss_fct = self.get_loss()

        # Dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Test/Val results dirs
        self.val_res_dir = os.path.join(save_dir, 'val')
        self.test_res_dir = os.path.join(save_dir, 'test')
        if not os.path.exists(self.val_res_dir):
            os.makedirs(self.val_res_dir)
        if not os.path.exists(self.test_res_dir):
            os.makedirs(self.test_res_dir)
        
        # For testing on val set
        self.evalImgs_all = []
        self.n_frames_all = 0

    def get_loss(self):
        """
        Define the loss function to use according to the configuration file
        @return: loss function object
        """
        loss_type = self.train_cfg['loss']
        if loss_type == 'bce':
            return nn.BCELoss()
        elif loss_type == 'mse':
            return nn.SmoothL1Loss()
        elif loss_type == 'smooth_ce':
            alpha = self.train_cfg['alpha_loss']
            return SmoothCELoss(alpha)
        else:
            raise ValueError

    def train_dataloader(self):
        """
        Define PyTorch training dataloader
        @return: train dataloader for ROD2021 dataset
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=True, num_workers=4, drop_last=True)
    
    def val_dataloader(self):
        """
        Define PyTorch validation dataloader
        @return: validation dataloader for ROD2021 dataset
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=cr_collate,
                          shuffle=False, num_workers=4, drop_last=True)

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/AP": 0, "hp/AR": 0, "hp/val_loss": 0, "hp/train_loss": 0})

    def forward(self, x):
        """
        Pytorch Lightning forward pass (inference)
        @param batch_positions: positional encoding vector (optional - only for UTAE)
        @param x: input tensor with shape (B, C, T, H, W) where T in the number of timesteps
        @return: ConfMap prediction
        """
        confmap_pred = self.model(x)
        return confmap_pred
    
    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        # Get data
        ra_maps = batch['radar_data']  # N, H, W, C
        confmap_gts = batch['anno']['confmaps']
        image_paths = batch['image_paths']

        confmap_pred = self.model(ra_maps)

        loss = self.loss_fct(confmap_pred, confmap_gts)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        # Get data
        ra_maps = batch['radar_data'] # N, H, W, C
        confmap_gts = batch['anno']['confmaps']
        image_paths = batch['image_paths']
        obj_infos = batch['anno']['obj_infos']

        confmap_pred = self.forward(ra_maps)

        loss = self.loss_fct(confmap_pred, confmap_gts)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp/val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        ra_maps = batch['radar_data']
        image_paths = batch['image_paths']
        confmap_gts = batch['anno']

        # Get seq name to write results
        seq_name = batch['seq_names'][0]
        if confmap_gts is not None:
            confmap_gts = batch['anno']['confmaps'].float()
            save_dir = os.path.join(self.val_res_dir)
        else:
            save_dir = os.path.join(self.test_res_dir)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, seq_name.upper()+".txt")
        
        if confmap_gts is not None:
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            start_frame_name = image_paths[0][0][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if frame_id == self.win_size-1 and self.model_name not in ('RECORDNoLstmMulti', 'RECORDNoLstmSingle'):
            for tmp_frame_id in range(frame_id):
                print("Eval frame", tmp_frame_id)
                tmp_ra_maps = ra_maps[:, :, :tmp_frame_id+1]
                confmap_pred = self.forward(tmp_ra_maps)
                res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
                write_dets_results_single_frame(res_final, tmp_frame_id, save_path, self.cruw_dataset_obj)

        confmap_pred = self.forward(ra_maps)

        # Write results
        res_final = post_process_single_frame_cruw(confmap_pred[0].cpu(), self.cruw_dataset_obj, self.config)
        write_dets_results_single_frame(res_final, frame_id, save_path, self.cruw_dataset_obj)

    def evaluate_rodnet_seq_(self, res_path, gt_path, n_frame, subset):
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        eval_imgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, self.cruw_dataset_obj)
        out_eval = accumulate(eval_imgs, n_frame, ols_thrs, rec_thrs, self.cruw_dataset_obj, log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)
        
        self.n_frames_all += n_frame
        self.evalImgs_all.extend(eval_imgs)
        
        self.logger.log_metrics({"AP/"+subset.upper(): stats[0] * 100,
                   "AR/"+subset.upper(): stats[1] * 100})
                
    def evaluate_rodnet_(self):
        ols_thrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        rec_thrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        out_eval = accumulate(self.evalImgs_all, self.n_frames_all, ols_thrs, rec_thrs, self.cruw_dataset_obj, log=False)
        stats = summarize(out_eval, ols_thrs, rec_thrs, self.cruw_dataset_obj, gl=False)
        self.logger.log_metrics({"AP/Overall": stats[0] * 100,
                   "AR/Overall": stats[1] * 100})

        self.logger.log_metrics({"hp/AP": stats[0] * 100,
                                 "hp/AR": stats[1] * 100})


    def on_before_batch_transfer(self, batch, dataloader_idx):
        if self.model_name == 'RECORDNoLstmMulti':
            b, c, t, h, w = batch['radar_data'].shape
            batch['radar_data'] = batch['radar_data'].reshape(b, c*t, h, w)
            return batch
        elif self.model_name == 'RECORDNoLstmSingle':
            b, c, t, h, w = batch['radar_data'].shape
            assert t == 1
            batch['radar_data'] = batch['radar_data'].reshape(b, c, h, w)
            return batch
        else:
            return batch
        
    def configure_optimizers(self):
        opti = self.train_cfg['optimizer']
        scheduler = self.train_cfg['scheduler']
        if opti == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif opti == 'adam_reg':
            assert self.weight_decay is not None
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif opti == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif opti == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0001)
        else: 
            raise ValueError

        if scheduler == 'exp':
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                'interval': 'epoch',
                'frequency': 10
            }
        elif scheduler == 'step':
            # for DANet
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif scheduler is None:
            return optimizer
        else:
            raise ValueError
        return [optimizer], [lr_scheduler]


