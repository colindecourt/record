import yaml 
from cruw import CRUW 
from datasets import ROD2021Dataset
import os

CONFIG_PATH = "path_to_configs"
def test_cruw_dataloader_classic():
    config_dict = yaml.load(open(os.path.join(CONFIG_PATH, 'cruw', 'config_record_cruw.yaml'), 'r'), Loader=yaml.FullLoader)
    cruw_dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'],
                        sensor_config_name=config_dict['model_cfg']['sensor_config'])
    data_dir = config_dict['dataset_cfg']['data_dir']
    
    print('---- Load train dataset with only the last confidence map----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                   is_random_chirp=False, all_confmaps=False)
    print('---- OK ----')

    print('---- Test output train dataset with only the last confidence map ----')
    out = train_dataset[0]
    out_radar = out['radar_data']
    confmap = out['anno']['confmaps']
    assert out_radar.shape == (config_dict['model_cfg']['in_channels'], 
                                config_dict['train_cfg']['win_size'],
                                128, 128)
    assert confmap.shape == (3, 128, 128)

    print('---- Load train dataset with all confidence maps----')
    train_dataset = ROD2021Dataset(config_dict=config_dict, data_dir=data_dir, dataset=cruw_dataset, split='train',
                                is_random_chirp=False, all_confmaps=True)
    print('---- OK ----')

    print('---- Test output train dataset all confidence maps ----')
    out = train_dataset[0]
    out_radar = out['radar_data']
    confmap = out['anno']['confmaps']
    assert out_radar.shape == (config_dict['model_cfg']['in_channels'], 
                                config_dict['train_cfg']['win_size'],
                                128, 128)
    assert confmap.shape == (3, config_dict['train_cfg']['win_size'], 128, 128)

