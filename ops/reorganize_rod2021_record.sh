# download all zip files and unzip
unzip TRAIN_RAD_H.zip
unzip TRAIN_CAM_0.zip
unzip TEST_RAD_H.zip
unzip TRAIN_RAD_H_ANNO.zip
unzip CAM_CALIB.zip

# make folders for data and annotations
mkdir sequences
mkdir annotations

# rename unzipped folders
mv TRAIN_RAD_H sequences/train
mv TRAIN_CAM_0 train
mv TEST_RAD_H sequences/test
mv TRAIN_RAD_H_ANNO annotations/train

# merge folders and remove redundant
rsync -av train/ sequences/train/
rm -r train

# create valid directory
mkdir sequences/valid/
mkdir annotations/valid/

# move sequence from train to valid directory
mv sequences/train/2019_04_09_PMS1000/ sequences/valid/2019_04_09_PMS1000
mv sequences/train/2019_04_30_MLMS001/ sequences/valid/2019_04_30_MLMS001
mv sequences/train/2019_05_29_MLMS006/ sequences/valid/2019_05_29_MLMS006
mv sequences/train/2019_09_29_ONRD005/ sequences/valid/2019_09_29_ONRD005

mv annotations/train/2019_04_09_PMS1000.txt annotations/valid/2019_04_09_PMS1000.txt
mv annotations/train/2019_04_30_MLMS001.txt annotations/valid/2019_04_30_MLMS001.txt
mv annotations/train/2019_05_29_MLMS006.txt annotations/valid/2019_05_29_MLMS006.txt
mv annotations/train/2019_09_29_ONRD005.txt annotations/valid/2019_09_29_ONRD005.txt

rm *.zip