cd /opt/data/private/zhl/BAM-CBAM-pytorch-master
proxy="http://clash:clash_3058@172.27.0.72:7890"
export http_proxy=$proxy https_proxy=$proxy all_proxy=$proxy
python3 main.py --arch none --backbone resnet50 --datasets ImageNet --n_epochs 90 --lr_scheduler StepLR --lr_decay_epoch 30 --gamma 0.1
python3 main.py --arch bam --backbone resnet50 --datasets ImageNet --n_epochs 90 --lr_scheduler StepLR --lr_decay_epoch 30 --gamma 0.1
python3 main.py --arch cbam --backbone resnet50 --datasets ImageNet --n_epochs 90 --lr_scheduler StepLR --lr_decay_epoch 30 --gamma 0.1
