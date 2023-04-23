import os
import argparse
from cruw.cruw import CRUW
import shutil

def parse_convert_to_rod2021():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rodnet_res_dir', type=str, help='Path to predictions folder')
    parser.add_argument('--convert_res_dir', type=str, help='Path to converted predictions folder')
    parser.add_argument('--archive_dir', type=str, help='Path to folder to save final archive')
    parser.add_argument('--data_root', type=str, help='Path to the prepared data')
    args = parser.parse_args()
    return args


SEQ_NAMES = ['2019_05_28_CM1S013', '2019_05_28_MLMS005', '2019_05_28_PBMS006', '2019_05_28_PCMS004',
             '2019_05_28_PM2S012', '2019_05_28_PM2S014', '2019_09_18_ONRD004', '2019_09_18_ONRD009',
             '2019_09_29_ONRD012', '2019_10_13_ONRD048']


if __name__ == '__main__':
    args = parse_convert_to_rod2021()
    dataset = CRUW(data_root=args.data_root, sensor_config_name='sensor_config_rod2021')
    seq_names = os.listdir(args.rodnet_res_dir)
    if not os.path.exists(args.convert_res_dir):
        os.makedirs(args.convert_res_dir)
    for seq_name in seq_names:
        seq_name = seq_name.split('.txt')[0]
        if seq_name.upper() in SEQ_NAMES:
            txt_name_in = os.path.join(args.rodnet_res_dir, seq_name.upper()+'.txt')
            txt_name_out = os.path.join(args.convert_res_dir, seq_name.upper() + '.txt')
            with open(txt_name_in, 'r') as f:
                data = f.readlines()
            f_out = open(txt_name_out, 'w')
            for line in data:
                frameid, class_name, rid, aid, conf = line.rstrip().split()
                rid = int(rid)
                aid = int(aid)
                r = dataset.range_grid[rid]
                a = dataset.angle_grid[aid]
                conf = float(conf)
                if conf > 1:
                    conf = 1.0
                f_out.write("%s %.4f %.4f %s %.4f\n" % (frameid, r, a, class_name, conf))
    f_out.close()
    shutil.make_archive(args.archive_dir, format='zip', root_dir=args.convert_res_dir)