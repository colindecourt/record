import os
import random
import pickle
import numpy as np
import time
from .parse_pkl import list_pkl_filenames_from_prepared
from .transforms import random_apply

from torch.utils import data
import torch


class ROD2021Dataset(data.Dataset):
    def __init__(self, data_dir, dataset, config_dict, split, is_random_chirp=False, subset=None, all_confmaps=False):
        """
        Dataset for the ROD2021 dataset. Modified from: https://github.com/yizhou-wang/RODNet
        @param data_dir: directory to the prepared data
        @param dataset: CRUW dataset object
        @param config_dict: dictionary with the training configuration (hyperparameters, paths to data etc.)
        @param split: split to load (train, valid or test)
        @param is_random_chirp: load a random chirps among the 4 available (only if num_chirps < 4
        @param subset: sequence to load, if only one is loaded (use in test)
        @param all_confmaps: if False returns only the ConfMap of the LAST timestep (use for RECORD, UTAE and DANet-C)
        """
        # parameters settings
        self.data_dir = data_dir
        self.dataset = dataset
        self.config_dict = config_dict
        self.n_class = dataset.object_cfg.n_class
        self.win_size = config_dict['train_cfg']['win_size']
        self.all_confmaps = all_confmaps
        self.model_name = config_dict['model_cfg']['name']
        self.aug_dict = config_dict['train_cfg']['aug']

        if config_dict['dataset_cfg']['docker']:
            docker_path = "/home/datasets/"

        # data settings
        self.normalize = config_dict['train_cfg']['normalize']
        if self.normalize:
            self.mean_data = torch.Tensor(config_dict['dataset_cfg']['mean_cplx'])
            self.std_data = torch.Tensor(config_dict['dataset_cfg']['std_cplx'])

        # Get "real" split folder
        if split == 'train':
            self.split = 'train'
        else:
            assert split in ('valid', 'test')
            self.split = split
            if config_dict['model_cfg']['name'] == "RECORD-OI":
                self.win_size = 1

        if self.split == 'train':
            self.step = config_dict['train_cfg']['train_step']
            self.stride = config_dict['train_cfg']['train_stride']
        else:
            self.step = config_dict['test_cfg']['test_step']
            self.stride = config_dict['test_cfg']['test_stride']

        self.is_random_chirp = is_random_chirp

        # Dataloader for MNet
        self.n_chirps = config_dict['model_cfg']['n_chirps']
        self.chirp_ids = self.dataset.sensor_cfg.radar_cfg['chirp_ids']

        # dataset initialization
        self.image_paths = []
        self.radar_paths = []
        self.obj_infos = []
        self.confmaps = []
        self.n_data = 0
        self.index_mapping = []

        if subset is not None:
            self.data_files = [subset + '.pkl']
        else:
            self.data_files = list_pkl_filenames_from_prepared(self.data_dir, self.split)

        self.seq_names = [name.split('.')[0] for name in self.data_files]

        self.n_seq = len(self.seq_names)

        for seq_id, data_file in enumerate(self.data_files):
            data_file_path = os.path.join(data_dir, self.split, data_file)
            data_details = pickle.load(open(data_file_path, 'rb'))
            assert split in ('train', 'valid', 'test')

            if split == 'train' or split == 'valid':
                assert data_details['anno'] is not None
            n_frame = data_details['n_frame']
            if not config_dict['dataset_cfg']['docker']:
                self.image_paths.append(data_details['image_paths'])
                self.radar_paths.append(data_details['radar_paths'])
            else:
                old_base_root = data_details['radar_paths'][0][0].split('ROD2021')[0]
                if split == 'test':
                    self.image_paths.append(None)
                else:
                    self.image_paths.append([new_path.replace(old_base_root, docker_path)
                                             for new_path in data_details['image_paths']])
                new_paths = []
                for list_chirps in data_details['radar_paths']:
                    new_paths.append([new_path.replace(old_base_root, docker_path) for new_path in list_chirps])
                self.radar_paths.append(new_paths)



            n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
                1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
            self.n_data += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * self.stride])
            if data_details['anno'] is not None:
                self.obj_infos.append(data_details['anno']['metadata'][:n_frame])
                self.confmaps.append(data_details['anno']['confmaps'][:n_frame])

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data

    def __getitem__(self, index):
        seq_id, data_id = self.index_mapping[index]
        seq_name = self.seq_names[seq_id]
        image_paths = self.image_paths[seq_id]
        radar_paths = self.radar_paths[seq_id]
        if len(self.confmaps) != 0:
            this_seq_obj_info = self.obj_infos[seq_id]
            this_seq_confmap = self.confmaps[seq_id]

        data_dict = dict(
            status=True,
            seq_names=seq_name,
            image_paths=[],
            positions=np.zeros(shape=(1, self.win_size))
        )

        if self.n_chirps < 4 and self.is_random_chirp:
            chirp_id = random.sample(range(0, 4), self.n_chirps)
        elif self.n_chirps == 4:
            chirp_id = np.arange(0, 4).tolist()
        elif self.n_chirps == 1 and self.is_random_chirp:
            chirp_id = random.randint(0, len(self.chirp_ids) - 1)
        elif self.n_chirps == 1 and not self.is_random_chirp:
            chirp_id = 0
        else:
            raise ValueError

        radar_configs = self.dataset.sensor_cfg.radar_cfg
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        # Load radar data
        try:
            if isinstance(chirp_id, int):
                radar_npy_win = torch.zeros((self.win_size, 2, ramap_rsize, ramap_asize), dtype=torch.float32)
                for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                    radar_npy_win[idx, :, :, :] = torch.from_numpy(
                        np.transpose(np.load(radar_paths[frameid][chirp_id]), (2, 0, 1)))
                    if self.split != 'test':
                        data_dict['image_paths'].append(image_paths[frameid])
                    else:
                        data_dict['image_paths'].append(radar_paths[frameid])
            elif isinstance(chirp_id, list):
                radar_npy_win = torch.zeros((self.win_size, self.n_chirps * 2, ramap_rsize, ramap_asize),
                                            dtype=torch.float32)
                for idx, frameid in enumerate(
                        range(data_id, data_id + self.win_size * self.step, self.step)):
                    for cid, c in enumerate(chirp_id):
                        npy_path = radar_paths[frameid][cid]
                        radar_npy_win[idx, cid * 2:cid * 2 + 2, :, :] = torch.from_numpy(
                            np.transpose(np.load(npy_path), (2, 0, 1)))
                    if self.split != 'test':
                        data_dict['image_paths'].append(image_paths[frameid])
                    else:
                        data_dict['image_paths'].append(radar_paths[frameid])
            else:
                raise TypeError
            radar_npy_win = radar_npy_win.transpose(1, 0)
        except:
            # in case load npy fail
            data_dict['status'] = False
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('./tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + radar_paths[frameid][chirp_id] + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            return data_dict

        data_dict['radar_data'] = radar_npy_win
        # Load annotations
        if len(self.confmaps) != 0:
            confmap_gt = this_seq_confmap[data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            confmap_gt = torch.from_numpy(confmap_gt)
            obj_info = this_seq_obj_info[data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = confmap_gt[:self.n_class]
            assert confmap_gt.shape == \
                   (self.n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            data_dict['anno'] = dict(
                obj_infos=obj_info,
                confmaps=confmap_gt.float(),
            )
        else:
            data_dict['anno'] = None

        if self.split == "train":
            # Data augmentation
            data_dict['radar_data'], data_dict['anno']['confmaps'], image_paths = random_apply(data_dict['radar_data'],
                                                                                               data_dict['anno'][
                                                                                                   'confmaps'],
                                                                                               image_paths,
                                                                                               self.aug_dict)

        # Normalize data
        if self.normalize:
            mean_data = self.mean_data.tile(self.n_chirps)
            std_data = self.std_data.tile(self.n_chirps)
            data_dict['radar_data'] = (data_dict['radar_data'] - mean_data[:, None, None, None]) / std_data[:, None, None, None]

        if not self.all_confmaps and data_dict['anno'] is not None:
            data_dict['anno']['confmaps'] = data_dict['anno']['confmaps'][:, -1]
            data_dict['anno']['obj_infos'] = data_dict['anno']['obj_infos']

        if self.model_name == 'UTAE':
            fps = 1 / 30
            pe = np.linspace(0, fps * self.win_size, self.win_size, dtype=np.float32)
            data_dict['positions'] = pe

        return data_dict
