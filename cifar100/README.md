### for VGG 
nohup python -u vgg.py --depth 13 --no-norm > vgg13.log &
nohup python -u vgg.py --depth 13 > vgg13_bn.log &
nohup python -u vgg.py --depth 16 --no-norm > vgg16.log &
nohup python -u vgg.py --depth 19 --lr 0.01 --no-norm > vgg19.log &

### for ResNet

nohup python -u resnet.py --depth 18 > resnet18.log &
nohup python -u resnet.py --depth 34 > resnet34.log &
nohup python -u resnet.py --depth 50 > resnet50.log &

### for DenseNet

nohup python -u densenet.py --depth 121 > densenet121.log &
nohup python -u densenet.py --depth 169 > densenet169.log &