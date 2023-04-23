"""
From: https://github.com/yizhou-wang/RODNet
"""
import os


def list_pkl_filenames(dataset_configs, split):
    data_root = dataset_configs['data_root']
    seqs = dataset_configs[split]['seqs']
    seqs_pkl_names = [name + '.pkl' for name in seqs]
    return seqs_pkl_names


def list_pkl_filenames_from_prepared(data_dir, split):
    seqs_pkl_names = sorted(os.listdir(os.path.join(data_dir, split)))
    return seqs_pkl_names


def list_pkl_filenames_from_sequences(data_dir, sequences):
    seqs_pkl_names = [] 
    for seq in sequences:
        seqs_pkl_names.append(os.path.join(data_dir, seq+'.pkl'))
    return sorted(seqs_pkl_names)