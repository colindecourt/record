import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from utils.args import parse_configs, update_config_dict
from datasets.carrada.dataset import Carrada
from datasets.carrada.dataloaders import SequenceCarradaDataset, CarradaDataset
from pytorch_lightning.loggers import TensorBoardLogger
from utils.models_utils import get_models


def parse_args():
    parser = argparse.ArgumentParser(description='MV-RECORD - Evaluate model')
    parser.add_argument('--config', required=True, type=str, help='configuration file path')
    parser.add_argument('--log_dir', required=True, type=str, help='Log directory (e.g. ./logs/)')
    parser.add_argument('--version', required=True, type=str, help='Version of the run to evaluate')
    parser.add_argument('--ckpt', required=True, type=str, help='Ckpt to resume the training')

    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']
dataset_cfg = config_dict['dataset_cfg']

n_frames = model_cfg['win_size']

# Load model
nb_classes = config_dict['model_cfg']['nb_classes']

# Load model
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

log_dir = args.log_dir
name = model_name
version = args.version
ckpt_path = args.ckpt
logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version, default_hp_metric=False)

# Load Pytorch Lightning models
if model_name == 'MV-RECORD':
    from executors import MVRECORDExecutor as Model
    model = Model(config_dict, model=model_instance)
    metric_to_monitor = 'val_metrics/global_prec'
elif model_name in ('RECORD-RD', 'RECORD-RA'):
    from executors import SVRECORDExecutor as Model
    model = Model(config_dict, model=model_instance, view=model_cfg['view'])
    metric_to_monitor = 'val_metrics/rd_prec' if model_cfg['view'] == 'range_doppler' else 'val_metrics/ra_prec'
else:
    raise ValueError

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'


trainer = pl.Trainer(logger=logger, accelerator=accelerator, devices=1,
                     max_epochs=config_dict['train_cfg']['n_epoch'])

print("Start evaluation")
### Test dataset
test_dataset = Carrada(config_dict).get('Test')
seq_dataloader = DataLoader(SequenceCarradaDataset(test_dataset), batch_size=1,
                            shuffle=False, num_workers=0)
all_datasets = []
for _, data in enumerate(seq_dataloader):
    seq_name, seq = data
    dataset_pth = config_dict['dataset_cfg']['carrada']
    batch_size = config_dict['train_cfg']['batch_size']
    path_to_frames = os.path.join(dataset_pth, seq_name[0])
    all_datasets.append(CarradaDataset(seq,
                                    'dense',
                                    path_to_frames,
                                    process_signal=True,
                                    n_frames=n_frames, add_temp=True))
test_dataloader = DataLoader(ConcatDataset(all_datasets), batch_size=train_cfg['batch_size'],
                             shuffle=False, num_workers=4)

trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

