# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pyramidnet as PYRM
from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Test')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: pyramidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--data_dir', default='/write/your/data/dir', type=str, metavar='PATH')
parser.add_argument('--pretrained', default='/set/your/model', type=str, metavar='PATH')
parser.add_argument('--fgsm', type=str2bool, default=False, help='true for fgsm')
parser.add_argument('--eps', default=1, type=int, help='1, 2, 4')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if args.dataset == 'cifar100':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir+'/dataCifar100/', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir+'/dataCifar10/', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))
    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.pretrained))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(args.pretrained))

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True
    mean = torch.tensor([x / 255 for x in [125.3, 123.0, 113.9]], dtype=torch.float32).view(1,3,1,1).cuda()
    std = torch.tensor([x / 255 for x in [63.0, 62.1, 66.7]], dtype=torch.float32).view(1,3,1,1).cuda()

    err1, err5, val_loss  = validate(val_loader, model, args.fgsm, args.eps, mean, std)

    print('Accuracy (top-1 and 5 error):', err1, err5)

def validate(val_loader, model, fgsm, eps, mean, std):
    '''evaluate trained model'''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # check FGSM for adversarial training
        if fgsm:
            input_var = Variable(input, requires_grad=True)
            target_var = Variable(target)

            optimizer_input = torch.optim.SGD([input_var], lr=0.1)
            output = model(input_var)
            loss = criterion(output, target_var)
            optimizer_input.zero_grad()
            loss.backward()

            sign_data_grad = input_var.grad.sign()
            input = input * std + mean + eps / 255. * sign_data_grad
            input = torch.clamp(input, 0, 1)
            input = (input - mean)/std

        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    if fgsm:
        print('Attack (eps : {}) Prec@1 {top1.avg:.2f}'.format(eps, top1=top1))
        print('Attack (eps : {}) Prec@5 {top5.avg:.2f}'.format(eps, top5=top5))
    else:
        print('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses))
    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
