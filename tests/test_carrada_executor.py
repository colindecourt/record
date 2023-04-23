import yaml
import os
from datasets.carrada.dataset import Carrada
from datasets.carrada.dataloaders import SequenceCarradaDataset, CarradaDataset, HFlip, VFlip
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

CONFIG_PATH = "path_to_configs"

def test_carradaexecutor_record():
    from models import MVRecord
    from executors import MVRECORDExecutor
    config = yaml.load(open(os.path.join(CONFIG_PATH, 'carrada', 'config_mvrecord_carrada.yaml'), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    backbone_cfg = yaml.load(open(config['model_cfg']['backbone_pth']), yaml.FullLoader)
    encoder_ra_cfg = backbone_cfg['encoder_config_ra']
    encoder_rd_cfg = backbone_cfg['encoder_config_rd']
    decoder_cfg = backbone_cfg['decoder_config']
    n_lstm = 0
    for layer in encoder_ra_cfg:
        if layer[0] == 1:
            n_lstm += 1
    if n_lstm == 0:
        in_channels = config['model_cfg']['in_channels']
    else:
        in_channels = 1
    n_frames = 5
    model_instance = MVRecord(encoder_ra_config=encoder_ra_cfg, encoder_rd_config=encoder_rd_cfg,
                              decoder_config=decoder_cfg,
                              n_classes=4, n_frames=n_frames, in_channels=in_channels)
    if model_instance.n_lstm == 0:
        add_temp = False
    else:
        add_temp = True

    dataset = Carrada(config).get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    all_datasets = []
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(paths['carrada'], seq_name[0])
        all_datasets.append(CarradaDataset(seq,
                                        'dense',
                                        path_to_frames,
                                        process_signal=True,
                                        n_frames=n_frames, add_temp=True))
    dataloader = DataLoader(ConcatDataset(all_datasets))

    executor = MVRECORDExecutor(config, model_instance)
    logger = TensorBoardLogger(save_dir=config['train_cfg']['ckpt_dir'], name=config['model_cfg']['name'],
                               default_hp_metric='global_prec')

    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor, train_dataloaders=dataloader, val_dataloaders=dataloader)