import yaml
import os
from datasets.carrada.dataset import Carrada
from datasets.carrada.dataloaders import SequenceCarradaDataset, CarradaDataset, HFlip, VFlip
from torch.utils.data import DataLoader
import os

CONFIG_PTH = "path_to_configs"

def test_carrada_dataset():
    """Method to test the dataset"""
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', 'config_mvrecord_carrada.yaml'), 'r'), yaml.FullLoader)
    dataset = Carrada(config).get('Train')
    assert '2019-09-16-12-52-12' in dataset.keys()
    # assert '2020-02-28-13-05-44' in dataset.keys()


def test_carrada_sequence():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', 'config_mvrecord_carrada.yaml'), 'r'), yaml.FullLoader)
    dataset = Carrada(config).get('Train')
    dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        if i == 0:
            seq = [subseq[0] for subseq in seq]
            assert seq_name[0] == '2019-09-16-12-52-12'
            assert '000163' in seq
            assert '001015' in seq
        else:
            break


def test_carradadataset():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', 'config_mvrecord_carrada.yaml'), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    n_frames = 3
    dataset = Carrada(config).get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(paths['carrada'], seq_name[0])
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [256, 64]
            assert list(frame['ra_matrix'].shape[2:]) == [256, 256]
            assert list(frame['ad_matrix'].shape[2:]) == [256, 64]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [256, 64]
            assert list(frame['ra_mask'].shape[2:]) == [256, 256]
        break


def test_carrada_subflip():
    config = yaml.load(open(os.path.join(CONFIG_PTH, 'carrada', 'config_mvrecord_carrada.yaml'), 'r'), yaml.FullLoader)
    paths = config['dataset_cfg']
    n_frames = 3
    dataset = Carrada(config).get('Train')
    seq_dataloader = DataLoader(SequenceCarradaDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = os.path.join(paths['carrada'], seq_name[0])
        frame_dataloader = DataLoader(CarradaDataset(seq,
                                                     'dense',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            rd_matrix = frame['rd_matrix'][0].cpu().detach().numpy()
            rd_mask = frame['rd_mask'][0].cpu().detach().numpy()
            rd_frame_test = {'matrix': rd_matrix,
                             'mask': rd_mask}
            rd_frame_vflip = VFlip()(rd_frame_test)
            rd_matrix_vflip = rd_frame_vflip['matrix']
            rd_frame_hflip = HFlip()(rd_frame_test)
            rd_matrix_hflip = rd_frame_hflip['matrix']
            assert rd_matrix[0][0][0] == rd_matrix_vflip[0][0][-1]
            assert rd_matrix[0][0][-1] == rd_matrix_vflip[0][0][0]
            assert rd_matrix[0][0][0] == rd_matrix_hflip[0][-1][0]
            assert rd_matrix[0][-1][0] == rd_matrix_hflip[0][0][0]
        break
