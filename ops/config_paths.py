import os
import argparse
import yaml


def config_paths(configs_pth, backbone_pth, ckpt_pth, rod_pth, rod_data, carrada_pth, docker):
    if args.docker:
        docker_flag = True
    else:
        docker_flag = False
    for dir in os.listdir(configs_pth):
        if dir == 'cruw':
            assert rod_pth is not None and rod_data is not None, \
                print('Please provide a path to ROD2021 dataset to write config files!')
            cruw_dir = os.path.join(configs_pth, dir)
            for file in os.listdir(cruw_dir):
                cur_file_pth = os.path.join(cruw_dir, file)
                # Read config
                tmp_config = yaml.load(open(cur_file_pth, 'r'), Loader=yaml.FullLoader)
                # Update config
                tmp_config['dataset_cfg']['base_root'] = rod_pth
                tmp_config['dataset_cfg']['docker'] = docker_flag
                tmp_config['dataset_cfg']['data_root'] = str(os.path.join(rod_pth, 'sequences'))
                tmp_config['dataset_cfg']['anno_root'] = os.path.join(rod_pth, 'annotations')
                tmp_config['dataset_cfg']['data_dir'] = rod_data
                tmp_config['train_cfg']['ckpt_dir'] = os.path.join(ckpt_pth, 'rod2021')
                if tmp_config['model_cfg']['name'] in ('RECORD', 'RECORDNoLstmSingle', 'RECORDNoLstmMulti', 'RECORD-OI'):
                    tmp_config['model_cfg']['backbone_pth'] = os.path.join(backbone_pth,
                                                                           tmp_config['model_cfg']['backbone_pth'].split(
                                                                               '/')[-1])
                # Write config
                with open(cur_file_pth, 'w') as f:
                    doc = yaml.dump(tmp_config, f, default_flow_style=False, sort_keys=False)
        elif dir == 'carrada':
            assert carrada_pth is not None, print('Please provide a path to CARRADA dataset to write config files!')
            carrada_dir = os.path.join(configs_pth, dir)
            for file in os.listdir(carrada_dir):
                cur_file_pth = os.path.join(carrada_dir, file)
                if file == "weights_config":
                    continue
                # Read config
                tmp_config = yaml.load(open(cur_file_pth, 'r'), Loader=yaml.FullLoader)
                # Update config
                tmp_config['dataset_cfg']['warehouse'] = carrada_pth.split('Carrada')[0]
                tmp_config['dataset_cfg']['carrada'] = carrada_pth
                tmp_config['dataset_cfg']['weight_path'] = os.path.join(configs_pth, 'carrada', 'weights_config')
                tmp_config['dataset_cfg']['project_path'] = configs_pth.split('configs')[0]
                tmp_config['train_cfg']['ckpt_dir'] = os.path.join(ckpt_pth, 'carrada')
                tmp_config['model_cfg']['backbone_pth'] = os.path.join(backbone_pth,
                                                                       tmp_config['model_cfg']['backbone_pth'].split(
                                                                           '/')[-1])
                # Write config
                with open(cur_file_pth, 'w') as f:
                    doc = yaml.dump(tmp_config, f, default_flow_style=False, sort_keys=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Configure paths for RECORD models')
    # General
    parser.add_argument('--configs_path', type=str, help='Path to the configuration files')
    parser.add_argument('--backbone_path', type=str, help='Path to model configuration files')
    parser.add_argument('--ckpt_path', type=str, help='Path to save logs and checkpoint files')
    # ROD2021 dataset
    parser.add_argument('--rod_base_root', type=str, help='Path to ROD2021 root folder')
    parser.add_argument('--rod_data', type=str, help='Path to ROD2021 prepared data')
    # CARRADA dataset
    parser.add_argument('--carrada_base_root', type=str, help='Path to CARRADA root folder')

    # Docker
    parser.add_argument('--docker', action='store_true')

    args = parser.parse_args()
    # Update configs
    config_paths(configs_pth=args.configs_path, backbone_pth=args.backbone_path, ckpt_pth=args.ckpt_path,
                 rod_data=args.rod_data, rod_pth=args.rod_base_root, carrada_pth=args.carrada_base_root,
                 docker=args.docker)