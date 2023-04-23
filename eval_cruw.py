import argparse
import os.path

import yaml
from cruw import CRUW
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_configs, update_config_dict
from evaluation.eval_cruw import eval_on_test, eval_on_val
from utils.models_utils import get_models
from datasets import ROD2021Dataset
from executors import RECORDExecutor as Model

def parse_args():
    parser = argparse.ArgumentParser(description='RECORD - Evaluate model')
    parser.add_argument('--config', required=True, type=str, help='configuration file path')
    parser.add_argument('--log_dir', required=True, type=str, help='Log directory (e.g. ./logs/)')
    parser.add_argument('--version', required=True, type=str, help='Version of the run to evaluate')
    parser.add_argument('--ckpt', required=True, type=str, help='Ckpt to resume the training')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')

    parser = parse_configs(parser)
    args = parser.parse_args()
    return args


args = parse_args()

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)


model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']
test_cfg = config_dict['test_cfg']
dataset_cfg = config_dict['dataset_cfg']

# Load model
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

# Init CRUW dataset utils
dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
               sensor_config_name=config_dict['model_cfg']['sensor_config'])
radar_configs = dataset.sensor_cfg.radar_cfg
range_grid = dataset.range_grid
angle_grid = dataset.angle_grid
data_dir = config_dict['dataset_cfg']['data_dir']

if args.test_on_val or args.test_all:
    valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict,
                                   all_confmaps=False, split='valid')
else:
    valid_dataset = None

log_dir = args.log_dir
name = model_name
version = args.version
ckpt_path = args.ckpt
logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version, default_hp_metric=False)

# Update variables with new config dict
if 'RECORD' in model_name:
    backbone_cfg = yaml.load(open(model_cfg['backbone_pth']), yaml.FullLoader)
    config_dict['model_cfg']['layout'] = backbone_cfg

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']


model_instance = get_models(model_cfg)

executor = Model(model=model_instance, train_dataset=None, val_dataset=valid_dataset, config_dict=config_dict,
                 cruw_dataset_obj=dataset, save_dir=logger.log_dir)

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'

trainer = pl.Trainer(logger=logger, accelerator=accelerator, devices=1,
                     max_epochs=train_cfg['n_epoch'])

print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']


if args.test_on_val:
    assert args.data_split == 'classic'
    print('Set for evaluation: VALIDATION')
    eval_on_val(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=False, ckpt_path=ckpt_path)
elif args.test_all:
    eval_on_val(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=False, ckpt_path=ckpt_path)
    eval_on_test(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=False, ckpt_path=ckpt_path)
else:    
    eval_on_test(trainer=trainer, executor=executor, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=False, ckpt_path=ckpt_path)

