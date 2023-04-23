import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from utils import update_config_dict, get_transformations, get_models
from datasets import SequenceCarradaDataset, CarradaDatasetOnline, Carrada, CarradaDatasetRangeDopplerOnline, CarradaDatasetRangeAngleOnline

def parse_args():
    parser = argparse.ArgumentParser(description='MV-RECORD')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--deterministic', action='store_true', help='Apply deterministic CUDA ops for reproducibility')
    parser.add_argument('--resume_ckpt', type=str, help='Path to the checkpoint to resume the training')
    args = parser.parse_args()
    return args


args = parse_args()
deterministic = False

seed = 42
pl.seed_everything(seed=seed, workers=True)

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
config = update_config_dict(config, args)

model_cfg = config['model_cfg']
train_cfg = config['train_cfg']
dataset_cfg = config['dataset_cfg']

n_frames = model_cfg['win_size']

# Load model
model_instance = get_models(model_cfg)
model_name = model_cfg['name']

# Load datasets
if model_name == 'RECORD-RD-OI':
    dataset_loader = CarradaDatasetRangeDopplerOnline
elif model_name == 'RECORD-RA-OI':
    dataset_loader = CarradaDatasetRangeAngleOnline
elif model_name in ('MV-RECORD-OI'):
    dataset_loader = CarradaDatasetOnline
else:
    raise ValueError

# Train dataset
train_dataset = Carrada(config).get('Train')
seq_dataloader = DataLoader(SequenceCarradaDataset(train_dataset), batch_size=1,
                            shuffle=True, num_workers=0)
all_datasets = []

transform_names = config['train_cfg']['transformations'].split(',')
transformations = get_transformations(transform_names=transform_names, sizes=(config['model_cfg']['w_size'], config['model_cfg']['h_size']))
for _, data in enumerate(seq_dataloader):
    seq_name, seq = data
    path_to_frames = os.path.join(dataset_cfg['carrada'], seq_name[0])
    all_datasets.append(dataset_loader(seq,
                                       'dense',
                                       path_to_frames,
                                       process_signal=True, transformations=transformations,
                                       n_frames=n_frames, add_temp=True))
train_dataloader = DataLoader(ConcatDataset(all_datasets), batch_size=train_cfg['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)

# Val dataset
val_dataset = Carrada(config).get('Validation')
seq_dataloader = DataLoader(SequenceCarradaDataset(val_dataset), batch_size=1,
                            shuffle=False, num_workers=0)
all_datasets = []
for _, data in enumerate(seq_dataloader):
    seq_name, seq = data
    path_to_frames = os.path.join(dataset_cfg['carrada'], seq_name[0])
    all_datasets.append(dataset_loader(seq,
                                    'dense',
                                    path_to_frames,
                                    process_signal=True,
                                    n_frames=1, add_temp=False))
val_dataloader = DataLoader(ConcatDataset(all_datasets), batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

# Logger
log_dir = train_cfg['ckpt_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = TensorBoardLogger(save_dir=train_cfg['ckpt_dir'], name=model_name, default_hp_metric=False)

# Add some entries to the configuration dict to get the logs
run_dir = logger.experiment.log_dir
config['train_cfg']['run_dir'] = run_dir

# Load Pytorch Lightning models
if model_name == 'MV-RECORD-OI':
    from executors import MVRECORDOIExecutor as Model
    model = Model(config, model=model_instance)
    metric_to_monitor = 'val_metrics/global_prec'
elif model_name in ('RECORD-RD-OI', 'RECORD-RA-OI'):
    from executors import SVRECORDOIExecutor as Model
    model = Model(config, model=model_instance, view=model_cfg['view'])
    metric_to_monitor = 'val_metrics/rd_prec' if model_cfg['view'] == 'range_doppler' else 'val_metrics/ra_prec'
else:
    raise ValueError

# Callbacks
checkpoint_callback = ModelCheckpoint(dirpath=run_dir, monitor=metric_to_monitor, mode="max",
                                      save_last=True, save_top_k=3)
lr_tracker = LearningRateMonitor()
early_stop = EarlyStopping(monitor=metric_to_monitor, patience=20, mode='max')
callbacks = [checkpoint_callback, lr_tracker, early_stop]



if torch.cuda.is_available():
    print('CUDA available, use GPU')
    accelerator = 'gpu'
else:
    print('WARNING: CUDA not available, use CPU')
    accelerator = 'cpu'
trainer = pl.Trainer(logger=logger, callbacks=callbacks, accelerator=accelerator, devices=1,
                     max_epochs=train_cfg['n_epoch'],
                     accumulate_grad_batches=train_cfg['accumulate_grad'])

print('Start training')
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.resume_ckpt)

print('Test model')
# Test dataset
test_dataset = Carrada(config).get('Test')
seq_dataloader = DataLoader(SequenceCarradaDataset(test_dataset), batch_size=1,
                            shuffle=False, num_workers=4)
all_datasets = []
for _, data in enumerate(seq_dataloader):
    seq_name, seq = data
    path_to_frames = os.path.join(dataset_cfg['carrada'], seq_name[0])
    all_datasets.append(dataset_loader(seq,
                                       'dense',
                                       path_to_frames,
                                       process_signal=True,
                                       n_frames=1, add_temp=False))
test_dataloader = DataLoader(ConcatDataset(all_datasets), batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')
