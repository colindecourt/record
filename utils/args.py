import argparse
from distutils.command.config import config

def parse_args():
    parser = argparse.ArgumentParser(description='RROD - A recurrent radar object detector')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--reproduce', action='store_true', help='If true, fix seed and use a different train/val set.')
    parser.add_argument('--data_split', default='classic', help='How to split the data (classic, train_only or mix)')
    parser.add_argument('--test_on_val', action='store_true')

    parser = parse_configs(parser)
    parser = parse_wandb(parser)
    args = parser.parse_args()
    return args

def parse_wandb(parser):
    parser.add_argument('--project', type=str, default='rrod', help='Weights and biases project (default: RROD)')
    parser.add_argument('--group', type=str, help='Group for grouping runs in Weights and biases')
    parser.add_argument('--id', type=str, help='id of the run to resume')
    parser.add_argument('--resume_ckpt', type=str, help='path to the checkpoint to resume the training')
    return parser

def parse_transforms(parser):
    parser.add_argument('--mirror', type=float, help="Probability of applying mirroring data augmentation")
    parser.add_argument('--reverse', type=float, help="Probability of applying temporal flip data augmentation")
    parser.add_argument('--gaussian', type=float, help="Probability of applying gaussian noise data augmentation")
    return parser


def parse_configs(parser):   
    # dataset_cfg
    parser.add_argument('--data_root', type=str,
                        help='directory to the dataset (will overwrite data_root in config file)')
    parser.add_argument('--data_dir', type=str,
                        help='directory to the prepared dataset (will overwrite data_dir in config file)')

    # model_cfg
    parser.add_argument('--model_name', type=str, help='name of the model to load')
    parser.add_argument('--max_dets', type=int, help='max detection per frome')
    parser.add_argument('--peak_thres', type=float, help='peak threshold')
    parser.add_argument('--ols_thres', type=float, help='OLS thres')
    parser.add_argument('--norm', type=str, help='normalization type')
    parser.add_argument('--n_group', type=int, help='number of group for group normalization')
    parser.add_argument('--in_channels', type=int, help='number of input channels')
    parser.add_argument('--n_chirps', type=int, help='number of chirps to use')
    parser.add_argument('--backbone_pth', type=str, help='path to the backbone layout to use')
    parser.add_argument('--width_mult', type=float, help='width multiplier of the model')
    

    # train_cfg
    parser.add_argument('--ckpt_dir', type=str, help='path to ckeckpoint data')
    parser.add_argument('--n_epoch', type=int, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, help='optimizer to use')
    parser.add_argument('--loss', type=str, help='loss function to use for optimize the model')
    parser.add_argument('--alpha_loss', type=float, help='alpha for loss')
    parser.add_argument('--win_size', type=int, help='window size for RF images')
    parser.add_argument('--train_step', type=int, help='training step within RF snippets')
    parser.add_argument('--train_stride', type=int, help='training stride between RF snippets')
    parser.add_argument('--log', action='store_true', help='if true log image during training')
    parser.add_argument('--train_log_step', type=int, help='step for logging image when training')
    parser.add_argument('--val_log_step', type=int, help='step for logging image when validate')
    parser.add_argument('--test_log_step', type=int, help='step for logging image when testing')
    
    # test_cfg
    parser.add_argument('--test_step', type=int, help='testing step within RF snippets')
    parser.add_argument('--test_stride', type=int, help='testing stride between RF snippets')
    parser.add_argument('--rr_min', type=float, help='range of range min value')
    parser.add_argument('--rr_max', type=float, help='range of range max value')
    parser.add_argument('--ra_min', type=float, help='range of angle min value')
    parser.add_argument('--ra_max', type=float, help='range of angle max value')

    return parser
    
def update_config_dict(config_dict, args):
    # dataset_cfg
    if hasattr(args, 'data_root') and args.data_root is not None:
        data_root_old = config_dict['dataset_cfg']['base_root']
        config_dict['dataset_cfg']['base_root'] = args.data_root
        config_dict['dataset_cfg']['data_root'] = config_dict['dataset_cfg']['data_root'].replace(data_root_old,
                                                                                                  args.data_root)
        config_dict['dataset_cfg']['anno_root'] = config_dict['dataset_cfg']['anno_root'].replace(data_root_old,
                                                                                                  args.data_root)
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        config_dict['dataset_cfg']['data_dir'] = args.data_dir

    # model_cfg
    if hasattr(args, 'model_name') and args.model_name is not None:
        config_dict['model_cfg']['name'] = args.model_name
    if hasattr(args, 'max_dets') and args.max_dets is not None:
        config_dict['model_cfg']['max_dets'] = args.max_dets
    if hasattr(args, 'peak_thres') and args.peak_thres is not None:
        config_dict['model_cfg']['peak_thres'] = args.peak_thres
    if hasattr(args, 'ols_thres') and args.ols_thres is not None:
        config_dict['model_cfg']['ols_thres'] = args.ols_thres
    if hasattr(args, 'norm') and args.norm is not None:
        config_dict['model_cfg']['norm'] = args.norm
    if hasattr(args, 'n_group') and args.n_group is not None:
        if args.n_group == 0:
            config_dict['model_cfg']['n_group'] = None
        else:
            config_dict['model_cfg']['n_group'] = args.n_group
    if hasattr(args, 'in_channels') and args.in_channels is not None:
        config_dict['model_cfg']['in_channels'] = args.in_channels
    if hasattr(args, 'n_chirps') and args.n_chirps is not None:
        config_dict['model_cfg']['n_chirps'] = args.n_chirps
    if hasattr(args, 'backbone_pth') and args.backbone_pth is not None:
        config_dict['model_cfg']['backbone_pth'] = args.backbone_pth
    if hasattr(args, 'width_mult') and args.width_mult is not None:
        config_dict['model_cfg']['width_mult'] = args.width_mult

    # train_cfg
    if hasattr(args, 'ckpt_dir') and args.ckpt_dir is not None:
        config_dict['train_cfg']['ckpt_dir'] = args.ckpt_dir
    if hasattr(args, 'n_epoch') and args.n_epoch is not None:
        config_dict['train_cfg']['n_epoch'] = args.n_epoch
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config_dict['train_cfg']['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        config_dict['train_cfg']['lr'] = args.lr
    if hasattr(args, 'optimizer') and args.optimizer is not None:
        config_dict['train_cfg']['optimizer'] = args.optimizer
    if hasattr(args, 'loss') and args.loss is not None:
        config_dict['train_cfg']['loss'] = args.loss
    if hasattr(args, 'alpha_loss') and args.alpha_loss is not None:
        config_dict['train_cfg']['alpha_loss'] = args.alpha_loss
    if hasattr(args, 'win_size') and args.win_size is not None:
        config_dict['train_cfg']['win_size'] = args.win_size
    if hasattr(args, 'train_step') and args.train_step is not None:
        config_dict['train_cfg']['train_step'] = args.train_step
    if hasattr(args, 'train_stride') and args.train_stride is not None:
        config_dict['train_cfg']['train_stride'] = args.train_stride
    if hasattr(args, 'log') and args.log:
        config_dict['train_cfg']['log'] = True
    if hasattr(args, 'train_log_step') and args.train_log_step is not None:
        config_dict['train_cfg']['train_log_step'] = args.train_log_step
    if hasattr(args, 'test_log_step') and args.test_log_step is not None:
        config_dict['train_cfg']['test_log_step'] = args.test_log_step
    if hasattr(args, 'val_log_step') and args.val_log_step is not None:
        config_dict['val_cfg']['train_log_step'] = args.val_log_step

    # Data augmentation
    if hasattr(args, 'mirror') and args.mirror is not None:
        config_dict['train_cfg']['aug']['mirror'] = args.mirror
    if hasattr(args, 'reverse') and args.reverse is not None:
        config_dict['train_cfg']['aug']['reverse'] = args.reverse
    if hasattr(args, 'gaussian') and args.gaussian is not None:
        config_dict['train_cfg']['aug']['gaussian'] = args.gaussian

    # test_cfg
    if hasattr(args, 'test_step') and args.test_step is not None:
        config_dict['test_cfg']['test_step'] = args.test_step
    if hasattr(args, 'test_stride') and args.test_stride is not None:
        config_dict['test_cfg']['test_stride'] = args.test_stride
    if hasattr(args, 'rr_min') and args.rr_min is not None:
        config_dict['test_cfg']['rr_min'] = args.rr_min
    if hasattr(args, 'rr_max') and args.rr_max is not None:
        config_dict['test_cfg']['rr_max'] = args.rr_max
    if hasattr(args, 'ra_min') and args.ra_min is not None:
        config_dict['test_cfg']['ra_min'] = args.ra_min
    if hasattr(args, 'ra_max') and args.ra_max is not None:
        config_dict['test_cfg']['ra_max'] = args.ra_max

    return config_dict