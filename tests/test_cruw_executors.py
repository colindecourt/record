import os
import yaml
from cruw import CRUW 
from datasets import ROD2021Dataset
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

CONFIG_PATH = "path_to_configs"

def test_cruw_executors_record():
    from models import Record
    from executors import RECORDExecutor
    path = os.path.join(CONFIG_PATH, 'cruw', 'config_record_cruw.yaml')
    config_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    config_dict['train_cfg']['batch_size'] = 4
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
     # Load datasets
    print('---- Load train dataset ----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False)
    print('---- OK ----')

    print('---- Load val dataset ----')
    valid_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='valid',
                                   is_random_chirp=False)
    print('---- OK ----')
    
    # Load model
    model_config = yaml.load(open(config_dict['model_cfg']['backbone_pth']), Loader=yaml.FullLoader)
    encoder_cfg = model_config['encoder_config']
    decoder_cfg = model_config['decoder_config']
    print('---- Load RECORD ----')
    model = Record(encoder_config=encoder_cfg, decoder_config=decoder_cfg, n_class=3, in_channels=8)
    print('---- OK ----')
    # Define executor
    logger = TensorBoardLogger(save_dir=config_dict['train_cfg']['ckpt_dir'], name=config_dict['model_cfg']['name'],
                               default_hp_metric='train_loss_epoch')
    print('---- Initialize executors for RECORD using RECORDExecutor class')
    executor = RECORDExecutor(model=model, train_dataset=train_dataset, val_dataset=valid_dataset,
                              config_dict=config_dict, cruw_dataset_obj=cruw_dataset,
                              save_dir=logger.log_dir)
    print('---- Executor initialized! ----')

    print('---- Perform one step of training ----')
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor)


def test_cruw_executors_record2d():
    from models import Record
    from executors import RECORDExecutor
    path = os.path.join(CONFIG_PATH, 'cruw', 'config_record_2d.yaml')
    config_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    config_dict['train_cfg']['batch_size'] = 4
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
    # Load datasets
    print('---- Load train dataset ----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False)
    print('---- OK ----')

    print('---- Load val dataset ----')
    valid_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='valid',
                                   is_random_chirp=False)
    print('---- OK ----')

    # Load model
    model_config = yaml.load(open(config_dict['model_cfg']['backbone_pth']), Loader=yaml.FullLoader)
    encoder_cfg = model_config['encoder_config']
    decoder_cfg = model_config['decoder_config']
    print('---- Load RECORD-2D ----')
    model = Record(encoder_config=encoder_cfg, decoder_config=decoder_cfg, n_class=3,
                   in_channels=config_dict['model_cfg']['in_channels'])
    print('---- OK ----')
    # Define executor
    logger = TensorBoardLogger(save_dir=config_dict['train_cfg']['ckpt_dir'], name=config_dict['model_cfg']['name'])
    print('---- Initialize executors for RECORD-2D using RECORDExecutor class')
    executor = RECORDExecutor(model=model, train_dataset=train_dataset, val_dataset=valid_dataset,
                              config_dict=config_dict, cruw_dataset_obj=cruw_dataset,
                              save_dir=logger.log_dir)
    print('---- Executor initialized! ----')

    print('---- Perform one step of training ----')
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor)


def test_cruw_executors_danet():
    from models import DANet
    from executors import DANetExecutor
    path = os.path.join(CONFIG_PATH, 'cruw', 'config_danet.yaml')
    config_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    config_dict['train_cfg']['batch_size'] = 4
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
    # Load datasets
    print('---- Load train dataset ----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False, all_confmaps=True)
    print('---- OK ----')

    print('---- Load val dataset ----')
    valid_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='valid',
                                   is_random_chirp=False, all_confmaps=True)
    print('---- OK ----')

    # Load model
    print('---- Load DANet ----')
    model = DANet(in_channels=8, n_class=3)
    print('---- OK ----')
    # Define executor
    logger = TensorBoardLogger(save_dir=config_dict['train_cfg']['ckpt_dir'], name=config_dict['model_cfg']['name'],
                               default_hp_metric='train_loss_epoch')
    print('---- Initialize executors for DANet using DANetExecutor class')
    executor = DANetExecutor(model=model, train_dataset=train_dataset, val_dataset=valid_dataset,
                              config_dict=config_dict, cruw_dataset_obj=cruw_dataset,
                              save_dir=logger.log_dir)
    print('---- Executor initialized! ----')

    print('---- Perform one step of training ----')
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor)


def test_cruw_executors_danet_c():
    from models import DANet
    from executors import DANetExecutor
    path = os.path.join(CONFIG_PATH, 'cruw', 'config_danet.yaml')
    config_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    config_dict['train_cfg']['batch_size'] = 4
    config_dict['model_cfg']['name'] = 'DANet-C'
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
    # Load datasets
    print('---- Load train dataset ----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False)
    print('---- OK ----')

    print('---- Load val dataset ----')
    valid_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='valid',
                                   is_random_chirp=False)
    print('---- OK ----')

    # Load model
    dummy_input = torch.rand((1, 1, 16, 128, 128))
    print('---- Load DANet-C ----')
    model = DANet(in_channels=8, n_class=3)
    print('---- OK ----')
    # Define executor
    logger = TensorBoardLogger(save_dir=config_dict['train_cfg']['ckpt_dir'], name=config_dict['model_cfg']['name'],
                               default_hp_metric='train_loss_epoch')
    print('---- Initialize executors for DANet-C using DANetExecutor class')
    executor = DANetExecutor(model=model, train_dataset=train_dataset, val_dataset=valid_dataset,
                              config_dict=config_dict, cruw_dataset_obj=cruw_dataset,
                              save_dir=logger.log_dir)
    print('---- Executor initialized! ----')

    print('---- Perform one step of training ----')
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor)


def test_cruw_executors_utae():
    from models import UTAE
    from executors import UTAEExecutor
    path = os.path.join(CONFIG_PATH, 'cruw', 'config_utae.yaml')
    config_dict = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    config_dict['train_cfg']['batch_size'] = 4
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
    # Load datasets
    print('---- Load train dataset ----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False, all_confmaps=False)
    print('---- OK ----')

    print('---- Load val dataset ----')
    valid_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='valid',
                                   is_random_chirp=False, all_confmaps=False)
    print('---- OK ----')

    # Load model
    print('---- Load UTAE ----')
    model = UTAE(input_dim=8)
    print('---- OK ----')
    # Define executor
    logger = TensorBoardLogger(save_dir=config_dict['train_cfg']['ckpt_dir'], name=config_dict['model_cfg']['name'],
                               default_hp_metric='train_loss_epoch')
    print('---- Initialize executors for UTAE using UTAEExecutor class')
    executor = UTAEExecutor(model=model, train_dataset=train_dataset, val_dataset=valid_dataset,
                            config_dict=config_dict, cruw_dataset_obj=cruw_dataset,
                            save_dir=logger.log_dir)
    print('---- Executor initialized! ----')

    print('---- Perform one step of training ----')
    trainer = pl.Trainer(logger=logger, accelerator='gpu', devices=1, max_steps=1)
    trainer.fit(executor)
