import argparse
import yaml
from cruw import CRUW
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_configs, parse_transforms, update_config_dict, get_models
from datasets import ROD2021Dataset
from evaluation import eval_on_test, eval_on_val
from executors import RECORDOIExecutor as Model


def parse_args():
    parser = argparse.ArgumentParser(description='RECORD model')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--test_on_val', action='store_true', help='Eval only on val set (default is test)')
    parser.add_argument('--test_all', action='store_true', help='Eval on val and on test sets')
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--seed', type=int, help='Seed to use for training the model')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    parser.add_argument('--tb_version', type=str, help='Name to the saved model')
    parser = parse_configs(parser)
    parser = parse_transforms(parser)
    args = parser.parse_args()
    return args


args = parse_args()
deterministic = False

seed = 252 if args.seed is None else args.seed

config_dict = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config_dict = update_config_dict(config_dict, args)

pl.seed_everything(seed=seed, workers=True)

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

# Load datasets
train_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='train')
valid_dataset = ROD2021Dataset(data_dir=data_dir, dataset=dataset, config_dict=config_dict, all_confmaps=True,
                               split='valid')

log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'], name=model_name, version=args.tb_version, default_hp_metric=False)

# Add some entries to the configuration dict to get back the logs
run_dir = logger.experiment.log_dir
config_dict['train_cfg']['run_dir'] = run_dir

checkpoint_callback = ModelCheckpoint(dirpath=None, monitor='val_loss', mode="min", save_last=True, save_top_k=3)
lr_tracker = LearningRateMonitor()

early_stop = EarlyStopping(monitor='val_loss', patience=7, mode='min')
callbacks = [checkpoint_callback, lr_tracker, early_stop]

model_cfg = config_dict['model_cfg']
train_cfg = config_dict['train_cfg']

model = Model(model=model_instance, train_dataset=train_dataset, val_dataset=valid_dataset, config_dict=config_dict,
                 cruw_dataset_obj=dataset, save_dir=logger.log_dir)

if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'
trainer = pl.Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, devices=1,
                     max_epochs=train_cfg['n_epoch'], deterministic=deterministic,
                     accumulate_grad_batches=train_cfg['accumulate_grad'])

print('Start training')
trainer.fit(model, ckpt_path=args.resume_ckpt)


print("Start evaluation")
data_root = config_dict['dataset_cfg']['data_root']

if args.test_on_val:
    print('Set for evaluation: VALIDATION')
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True)
elif args.test_all:
    eval_on_val(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                config_dict=config_dict, all_confmaps=True, ckpt_path='best')
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')
else:    
    eval_on_test(trainer=trainer, executor=model, dataset=dataset, data_root=config_dict['dataset_cfg']['data_root'],
                 config_dict=config_dict, all_confmaps=True, ckpt_path='best')

print('Training finished.')
