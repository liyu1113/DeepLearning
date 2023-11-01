cd /opt/data/private/zhl/BAM-CBAM-pytorch-master
proxy="http://clash:clash_3058@172.27.0.72:7890"
export http_proxy=$proxy https_proxy=$proxy all_proxy=$proxy
python3 main.py --arch none --backbone resnet50 --datasets CIFAR100
python3 main.py --arch cbam --backbone resnet50 --datasets CIFAR100
