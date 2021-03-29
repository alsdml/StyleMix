This code is based on the implementations for CutMix(CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features), Puzzle Mix(Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup), AdaIN(Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization)

## Usage
1) train.py : Code to train the model
2) train.sh : Script to run train.py
3) test.py : Code to check CIFAR-100 and CIFAR-10 classification performance (if fgsm option is set to False) or experiment with FGSM Attack (if fgsm option is set to True) in Pyramid200 model.
4) test.sh : Script to run test.py
5) makeStyleDistanceMatrix.py : Code to make styleDistanceMatrix10, 100
6) makeStyleDistanceMatrix.sh : Script to run makeStyleDistance.sh
7) models : The directory containing the pre-trained style transfer encoder and decoder networks.

## Train Model

Modify the contents of train.sh according to each situation and run it.

1. StyleCutMix_Auto_Gamma + CIFAR-100

python train.py \
    --net_type pyramidnet \
    --dataset cifar100 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --prob 0.5 \
    --r 0.7 \
    --delta 3.0 \
    --method StyleCutMix_Auto_Gamma \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

2. StyleCutMix_Auto_Gamma + CIFAR-10

python train.py \
    --net_type pyramidnet \
    --dataset cifar10 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --prob 0.5 \
    --r 0.7 \
    --delta 1.0 \
    --method StyleCutMix_Auto_Gamma \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

3. StyleCutMix + CIFAR-100

python train.py \
    --net_type pyramidnet \
    --dataset cifar100 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --prob 0.5 \
    --r 0.7 \
    --alpha2 0.8 \
    --method StyleCutMix \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

4. StyleCutMix + CIFAR-10

python train.py \
    --net_type pyramidnet \
    --dataset cifar10 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --prob 0.5 \
    --r 0.7 \
    --alpha2 0.8 \
    --method StyleCutMix \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

5. StyleMix + CIFAR-100

python train.py \
    --net_type pyramidnet \
    --dataset cifar100 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --alpha1 0.5 \
    --prob 0.2 \
    --r 0.7 \
    --method StyleMix \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

6. StyleMix + CIFAR-10

python train.py \
    --net_type pyramidnet \
    --dataset cifar10 \
    --depth 200 \
    --alpha 240 \
    --batch_size 64 \
    --lr 0.25 \
    --expname PyraNet200 \
    --epochs 300 \
    --alpha1 0.5 \
    --prob 0.2 \
    --r 0.7 \
    --method StyleMix \
    --save_dir /set/your/save/dir \
    --data_dir /set/your/data/dir \
    --no-verbose

## Test classification performance

Modify the contents of test.sh according to each situation and run it.

1. CIFAR-10

test.sh :
  python test.py \
    --net_type pyramidnet \
    --dataset cifar10 \
    --batch_size 128 \
    --depth 200 \
    --alpha 240 \
    --fgsm False \
    --data_dir /set/your/data/dir \
    --pretrained /set/pretrained/model/dir

2. CIFAR-100

test.sh :
  python test.py \
    --net_type pyramidnet \
    --dataset cifar100 \
    --batch_size 128 \
    --depth 200 \
    --alpha 240 \
    --fgsm False \
    --data_dir /set/your/data/dir \
    --pretrained /set/pretrained/model/dir

## Test FGSM Attack

Modify the contents of test.sh according to each situation and run it.

1. FGSM Attack on CIFAR-10

test.sh :
  python test.py \
    --net_type pyramidnet \
    --dataset cifar10 \
    --batch_size 128 \
    --depth 200 \
    --alpha 240 \
    --fgsm True \
    --eps 1 \
    --data_dir /set/your/data/dir \
    --pretrained /set/pretrained/model/dir

(You can change eps to 1, 2, 4)

2. FGSM Attack on CIFAR-100

test.sh :
  python test.py \
    --net_type pyramidnet \
    --dataset cifar100 \
    --batch_size 128 \
    --depth 200 \
    --alpha 240 \
    --fgsm True \
    --eps 1 \
    --data_dir /set/your/data/dir \
    --pretrained /set/pretrained/model/dir

(You can change eps to 1, 2, 4)

## make StyleDistance Matrix

Modify the contents of makeStyleDistanceMatrix.sh according to each situation and run it.

1. CIFAR-100 (make styleDistanceMatrix100.pt)

makeStyleDistanceMatrix.sh :
  python makeStyleDistanceMatrix.py \
    --dataset cifar100 \
    --batch_size 256 \
    --data_dir /set/your/data/dir

2. CIFAR-10 (make styleDistanceMatrix10.pt)

makeStyleDistanceMatrix.sh :
  python makeStyleDistanceMatrix.py \
    --dataset cifar10 \
    --batch_size 256 \
    --data_dir /set/your/data/dir

