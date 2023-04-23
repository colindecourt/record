FROM nvcr.io/nvidia/pytorch:21.11-py3


#RUN apt-get update
RUN pip install PyYAML pytorch-lightning==1.8.1 torchinfo scikit-image

COPY . ./record
RUN pip install -e ./record

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR ./record

RUN git clone https://github.com/yizhou-wang/cruw-devkit.git
RUN pip install ./cruw-devkit/
RUN pip install .

RUN python ./ops/config_paths.py --configs_path $PWD/configs/ --backbone_path $PWD/models/configs/ \
    --ckpt_path /home/logs/ --rod_base_root /home/datasets/ROD2021/ \
    --rod_data /home/datasets/cruw_prepared/ --carrada_base_root /home/datasets/Carrada/ --docker

