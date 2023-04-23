import pytorch_lightning as pl
import torch
from utils.carrada_functions import normalize, get_metrics
from .carrada_trainer import CarradaExecutor

class CarradaExecutorOI(CarradaExecutor):

    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)

        rd_loss = 0
        ra_loss = 0
        loss = 0
        coherence_loss = 0
        rd_losses_log =[]
        ra_losses_log = []

        # Forward
        for t in range(rd_data.shape[2]):
            if t == 0:
                self.model.rd_encoder.__init_hidden__()
                self.model.ra_encoder.__init_hidden__()
                self.model.ad_encoder.__init_hidden__()
            rd_outputs, ra_outputs = self.model(rd_data[:, :, t], ra_data[:, :, t], ad_data[:, :, t])

            # Compute loss
            if self.nb_losses < 3:
                # Case without the CoL
                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, t], axis=1))
                                for c in self.rd_criterion]
                rd_loss += torch.mean(torch.stack(rd_losses))
                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, t], axis=1))
                                for c in self.ra_criterion]
                ra_loss += torch.mean(torch.stack(ra_losses))
                loss += torch.mean(rd_loss + ra_loss)
            else:
                # Case with the CoL
                # Select the wCE and wSDice
                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, t], axis=1))
                                for c in self.rd_criterion[:2]]
                rd_loss += torch.mean(torch.stack(rd_losses))
                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, t], axis=1))
                                for c in self.ra_criterion[:2]]
                ra_loss += torch.mean(torch.stack(ra_losses))
                # Coherence loss
                coherence_loss += self.rd_criterion[2](rd_outputs, ra_outputs)
                loss += torch.mean(rd_loss + ra_loss + coherence_loss)

            # For logging only
            rd_losses_log.append(rd_losses)
            ra_losses_log.append(ra_losses)

        rd_loss /= rd_data.shape[2]
        ra_loss /= rd_data.shape[2]
        loss /= rd_data.shape[2]
        # Log losses
        if self.nb_losses > 2:
            coherence_loss = torch.mean(coherence_loss)
            loss_dict = {
                'train/loss': loss,
                'train/rd_global': rd_loss,
                'train/rd_ce': torch.mean(torch.Tensor(rd_losses_log), dim=0)[0],
                'train/rd_Dice': torch.mean(torch.Tensor(rd_losses_log), dim=0)[1],
                'train/ra_global': ra_loss,
                'train/ra_ce': torch.mean(torch.Tensor(ra_losses_log), dim=0)[0],
                'train/ra_Dice': torch.mean(torch.Tensor(rd_losses_log), dim=1)[1],
                'train/coherence': coherence_loss
            }
        else:
            loss_dict = {
                'train/loss': loss,
                'train/rd_global': rd_loss,
                'train/rd_ce': torch.mean(torch.Tensor(rd_losses_log), dim=0)[0],
                'train/rd_Dice': torch.mean(torch.Tensor(rd_losses_log), dim=0)[1],
                'train/ra_global': ra_loss,
                'train/ra_ce': torch.mean(torch.Tensor(ra_losses_log), dim=0)[0],
                'train/ra_Dice': torch.mean(torch.Tensor(rd_losses_log), dim=1)[1],
            }
        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):
        self.model.rd_encoder.__init_hidden__()
        self.model.ra_encoder.__init_hidden__()
        self.model.ad_encoder.__init_hidden__()
        test_results = dict()
        test_results['range_doppler'] = get_metrics(self.rd_metrics)
        test_results['range_angle'] = get_metrics(self.ra_metrics)

        test_results['global_acc'] = (1/2)*(test_results['range_doppler']['acc']+
                                                    test_results['range_angle']['acc'])
        test_results['global_prec'] = (1/2)*(test_results['range_doppler']['prec']+
                                                    test_results['range_angle']['prec'])
        test_results['global_dice'] = (1/2)*(test_results['range_doppler']['dice']+
                                                    test_results['range_angle']['dice'])

        test_results_log = {
            'val_metrics/rd_acc': test_results['range_doppler']['acc'],
            'val_metrics/rd_prec': test_results['range_doppler']['prec'],
            'val_metrics/rd_miou': test_results['range_doppler']['miou'],
            'val_metrics/rd_dice': test_results['range_doppler']['dice'],
            'val_metrics/ra_acc': test_results['range_angle']['acc'],
            'val_metrics/ra_prec': test_results['range_angle']['prec'],
            'val_metrics/ra_miou': test_results['range_angle']['miou'],
            'val_metrics/ra_dice': test_results['range_angle']['dice'],
            'val_metrics/global_prec': test_results['global_prec'],
            'val_metrics/global_acc': test_results['global_acc'],
            'val_metrics/global_dice': test_results['global_dice']
        }

        self.log_dict(test_results_log, on_epoch=True)
        self.log("hp/val_global_prec", test_results['global_prec'], on_epoch=True)
        self.log("hp/val_global_dice", test_results['global_dice'], on_epoch=True)
        self.rd_metrics.reset()
        self.ra_metrics.reset()

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)

        rd_outputs, ra_outputs = self.forward(rd_data, ra_data, ad_data)
            
        # Compute loss 
        if self.nb_losses < 3:
            # Case without the CoL
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, 0], axis=1))
                            for c in self.rd_criterion]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, 0], axis=1))
                            for c in self.ra_criterion]
            ra_loss = torch.mean(torch.stack(ra_losses))
            loss = torch.mean(rd_loss + ra_loss)
        else:
            # Case with the CoL
            # Select the wCE and wSDice
            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, 0], axis=1))
                            for c in self.rd_criterion[:2]]
            rd_loss = torch.mean(torch.stack(rd_losses))
            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, 0], axis=1))
                            for c in self.ra_criterion[:2]]
            ra_loss = torch.mean(torch.stack(ra_losses))
            # Coherence loss
            coherence_loss = self.rd_criterion[2](rd_outputs, ra_outputs)
            loss = torch.mean(rd_loss + ra_loss + coherence_loss)

        # Compute metrics
        self.rd_metrics.add_batch(torch.argmax(rd_mask[:, :, 0], axis=1).cpu(),
                                torch.argmax(rd_outputs, axis=1).cpu())
        self.ra_metrics.add_batch(torch.argmax(ra_mask[:, :, 0], axis=1).cpu(),
                                torch.argmax(ra_outputs, axis=1).cpu())
        
        if self.nb_losses > 2:
            loss_dict = {
                'val/loss': loss,
                'val/rd_global': rd_loss,
                'val/rd_ce': rd_losses[0],
                'val/rd_Dice': rd_losses[1],
                'val/ra_global': ra_loss,
                'val/ra_ce': ra_losses[0],
                'val/ra_Dice': ra_losses[1],
                'val/coherence': coherence_loss
            }
        else:
            loss_dict = {
                'val/loss': loss,
                'val/rd_global': rd_loss,
                'val/rd_ce': rd_losses[0],
                'val/rd_Dice': rd_losses[1],
                'val/ra_global': ra_loss,
                'val/ra_ce': ra_losses[0],
                'val/ra_Dice': ra_losses[1]      
            }
        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('hp/val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """

        rd_data = batch['rd_matrix'].float()
        ra_data = batch['ra_matrix'].float()
        ad_data = batch['ad_matrix'].float()
        rd_mask = batch['rd_mask'].float()
        ra_mask = batch['ra_mask'].float()
        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type, proj_path=self.project_path)
        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type, proj_path=self.project_path)

        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type, proj_path=self.project_path)

        rd_outputs, ra_outputs = self.forward(rd_data, ra_data, ad_data)

        # Compute metrics
        self.rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                  torch.argmax(rd_outputs, axis=1).cpu())
        self.ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                  torch.argmax(ra_outputs, axis=1).cpu())

    def on_test_epoch_end(self):
        """
        Compute metrics and log it 
        """
        self.model.rd_encoder.__init_hidden__()
        self.model.ra_encoder.__init_hidden__()
        self.model.ad_encoder.__init_hidden__()
        test_results = dict()
        test_results['range_doppler'] = get_metrics(self.rd_metrics)
        test_results['range_angle'] = get_metrics(self.ra_metrics)

        test_results['global_acc'] = (1/2)*(test_results['range_doppler']['acc']+
                                                    test_results['range_angle']['acc'])
        test_results['global_prec'] = (1/2)*(test_results['range_doppler']['prec']+
                                                    test_results['range_angle']['prec'])
        test_results['global_dice'] = (1/2)*(test_results['range_doppler']['dice']+
                                                    test_results['range_angle']['dice'])

        test_results_log = {
            'test_metrics/rd_acc': test_results['range_doppler']['acc'],
            'test_metrics/rd_prec': test_results['range_doppler']['prec'],
            'test_metrics/rd_miou': test_results['range_doppler']['miou'],
            'test_metrics/rd_dice': test_results['range_doppler']['dice'],
            'test_metrics/ra_acc': test_results['range_angle']['acc'],
            'test_metrics/ra_prec': test_results['range_angle']['prec'],
            'test_metrics/ra_miou': test_results['range_angle']['miou'],
            'test_metrics/ra_dice': test_results['range_angle']['dice'],
            'test_metrics/global_prec': test_results['global_prec'],
            'test_metrics/global_acc': test_results['global_acc'],
            'test_metrics/global_dice': test_results['global_dice'],
            'test_metrics/rd_dice_bkg': test_results['range_doppler']['dice_by_class'][0],
            'test_metrics/rd_dice_ped': test_results['range_doppler']['dice_by_class'][1],
            'test_metrics/rd_dice_cycl': test_results['range_doppler']['dice_by_class'][2],
            'test_metrics/rd_dice_car': test_results['range_doppler']['dice_by_class'][3],
            'test_metrics/ra_dice_bkg': test_results['range_angle']['dice_by_class'][0],
            'test_metrics/ra_dice_ped': test_results['range_angle']['dice_by_class'][1],
            'test_metrics/ra_dice_cycl': test_results['range_angle']['dice_by_class'][2],
            'test_metrics/ra_dice_car': test_results['range_angle']['dice_by_class'][3],
            
            'test_metrics/rd_iou_bkg': test_results['range_doppler']['miou_by_class'][0],
            'test_metrics/rd_iou_ped': test_results['range_doppler']['miou_by_class'][1],
            'test_metrics/rd_iou_cycl': test_results['range_doppler']['miou_by_class'][2],
            'test_metrics/rd_iou_car': test_results['range_doppler']['miou_by_class'][3],
            'test_metrics/ra_iou_bkg': test_results['range_angle']['miou_by_class'][0],
            'test_metrics/ra_iou_ped': test_results['range_angle']['miou_by_class'][1],
            'test_metrics/ra_iou_cycl': test_results['range_angle']['miou_by_class'][2],
            'test_metrics/ra_iou_car': test_results['range_angle']['miou_by_class'][3],
            
        }

        self.log_dict(test_results_log, on_epoch=True)
        self.log(name='hp/test_rd_miou', value=test_results['range_doppler']['miou'], on_epoch=True)
        self.log(name="hp/test_ra_miou", value=test_results['range_angle']['miou'], on_epoch=True)
        self.rd_metrics.reset()
        self.ra_metrics.reset()

    def on_validation_start(self):
        self.model.rd_encoder.__init_hidden__()
        self.model.ra_encoder.__init_hidden__()
        self.model.ad_encoder.__init_hidden__()

    def on_test_start(self):
        self.model.rd_encoder.__init_hidden__()
        self.model.ra_encoder.__init_hidden__()
        self.model.ad_encoder.__init_hidden__()