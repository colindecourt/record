import torch
from utils.carrada_functions import normalize, get_metrics
from .carrada_trainer_sv import CarradaExecutorSV

class CarradaExecutorSVOI(CarradaExecutorSV):

    def training_step(self, batch, batch_id):
        """
        Perform one training step (forward + backward) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        @return: loss value to log
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        loss = 0
        losses_log =[]

        # Forward
        for t in range(data.shape[2]):
            if t == 0:
                self.model.encoder.__init_hidden__()
            outputs = self.model(data[:, :, t])

            # Case without the CoL
            losses = [c(outputs, torch.argmax(mask[:, :, t], axis=1))
                            for c in self.criterion]
            loss += torch.mean(torch.stack(losses))

            # For logging only
            losses_log.append(losses)

        loss /= data.shape[2]

        # Log losses
        loss_dict = {
            'train/loss': loss,
            'train/ce': torch.mean(torch.Tensor(losses_log), dim=0)[0],
            'train/dice': torch.mean(torch.Tensor(losses_log), dim=0)[1],
        }
        self.log_dict(loss_dict, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        self.log('hp/train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def on_validation_epoch_end(self):
        test_results = get_metrics(self.metrics)

        if self.view == 'range_doppler':
            test_results_log = {
                'val_metrics/rd_acc': test_results['acc'],
                'val_metrics/rd_prec': test_results['prec'],
                'val_metrics/rd_miou': test_results['miou'],
                'val_metrics/rd_dice': test_results['dice']
            }
        elif self.view == 'range_angle':
            test_results_log = {
                'val_metrics/ra_acc': test_results['acc'],
                'val_metrics/ra_prec': test_results['prec'],
                'val_metrics/ra_miou': test_results['miou'],
                'val_metrics/ra_dice': test_results['dice']
            }

        self.log_dict(test_results_log, on_epoch=True)
        self.log(name='hp/val_global_prec', value=test_results['prec'], on_epoch=True)
        self.log(name="hp/val_global_dice", value=test_results['dice'], on_epoch=True)
        self.metrics.reset()
        self.model.encoder.__init_hidden__()

    def validation_step(self, batch, batch_id):
        """
        Perform a validation step (forward pass) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        outputs = self.forward(data)

        # Compute loss
        # Case without the CoL
        losses = [c(outputs, torch.argmax(mask[:, :, 0], axis=1))
                  for c in self.criterion]
        loss = torch.mean(torch.stack(losses))

        # Log losses
        loss_dict = {
            'val/loss': loss,
            'val/ce': losses[0],
            'val/dice': losses[1],
        }

        # Compute metrics
        self.metrics.add_batch(torch.argmax(mask[:, :, 0], axis=1).cpu(),
                               torch.argmax(outputs, axis=1).cpu())

        self.log_dict(loss_dict, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('hp/val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_id):
        """
        Perform a test step (forward pass + evaluation) on a batch of data.
        @param batch: data batch from the dataloader
        @param batch_id: id of the current batch
        """
        if self.view == 'range_doppler':
            data = batch['rd_matrix'].float()
            mask = batch['rd_mask'].float()
        elif self.view == 'range_angle':
            data = batch['ra_matrix'].float()
            mask = batch['ra_mask'].float()
        data = normalize(data, self.view, norm_type=self.norm_type, proj_path=self.project_path)

        outputs = self.forward(data)

        # Compute metrics
        self.metrics.add_batch(torch.argmax(mask, axis=1).cpu(),
                               torch.argmax(outputs, axis=1).cpu())

    def on_test_epoch_end(self):
        """
        Compute metrics and log it
        """
        self.model.encoder.__init_hidden__()
        test_results = get_metrics(self.metrics)
        if self.view == 'range_doppler':
            test_results_log = {
                'test_metrics/rd_acc': test_results['acc'],
                'test_metrics/rd_prec': test_results['prec'],
                'test_metrics/rd_miou': test_results['miou'],
                'test_metrics/rd_dice': test_results['dice'],

                'test_metrics/rd_dice_bkg': test_results['dice_by_class'][0],
                'test_metrics/rd_dice_ped': test_results['dice_by_class'][1],
                'test_metrics/rd_dice_cycl': test_results['dice_by_class'][2],
                'test_metrics/rd_dice_car': test_results['dice_by_class'][3],

                'test_metrics/rd_iou_bkg': test_results['miou_by_class'][0],
                'test_metrics/rd_iou_ped': test_results['miou_by_class'][1],
                'test_metrics/rd_iou_cycl': test_results['miou_by_class'][2],
                'test_metrics/rd_iou_car': test_results['miou_by_class'][3],

            }
        elif self.view == 'range_angle':
            test_results_log = {
                'test_metrics/ra_acc': test_results['acc'],
                'test_metrics/ra_prec': test_results['prec'],
                'test_metrics/ra_miou': test_results['miou'],
                'test_metrics/ra_dice': test_results['dice'],

                'test_metrics/ra_dice_bkg': test_results['dice_by_class'][0],
                'test_metrics/ra_dice_ped': test_results['dice_by_class'][1],
                'test_metrics/ra_dice_cycl': test_results['dice_by_class'][2],
                'test_metrics/ra_dice_car': test_results['dice_by_class'][3],

                'test_metrics/ra_iou_bkg': test_results['miou_by_class'][0],
                'test_metrics/ra_iou_ped': test_results['miou_by_class'][1],
                'test_metrics/ra_iou_cycl': test_results['miou_by_class'][2],
                'test_metrics/ra_iou_car': test_results['miou_by_class'][3],
            }

        self.log_dict(test_results_log, on_epoch=True)
        self.log(name='hp/test_dice', value=test_results['dice'], on_epoch=True)
        self.log(name="hp/test_miou", value=test_results['miou'], on_epoch=True)
        self.metrics.reset()

    def on_validation_epoch_start(self):
        self.model.encoder.__init_hidden__()

    def on_test_epoch_start(self):
        self.model.encoder.__init_hidden__()
