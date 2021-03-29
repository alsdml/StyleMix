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
import utils
import numpy as np
import torchvision.utils
from torchvision.utils import save_image
import warnings
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from function import calc_mean_std
from torch.utils.tensorboard import SummaryWriter
import net_cutmix
import net_mixup
from function import adaptive_instance_normalization, coral
import torch.nn.functional as F
from IPython import embed
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import datetime
warnings.filterwarnings("ignore")
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='StyleMix CIFAR-10, CIFAR-100 training code')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: pyramidnet')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run') # 250
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=18, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=200, type=float,
                    help='number of new channel increases per depth (default: 200)')
parser.add_argument('--expname', default='PyraNet200', type=str,
                    help='name of experiment')
parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='./models/decoder.pth.tar')
parser.add_argument('--prob', default=0.5, type=float)
parser.add_argument('--r', default=0.7, type=float)
parser.add_argument('--alpha1', default=1.0, type=float)
parser.add_argument('--alpha2', default=1.0, type=float)
parser.add_argument('--delta', default=3.0, type=float)
parser.add_argument('--method', type=str, default='StyleCutMix_Auto_Gamma', help='StyleCutMix_Auto_Gamma, StyleCutMix, StyleMix')
parser.add_argument('--save_dir', type=str, default='/write/your/save/dir')
parser.add_argument('--data_dir', type=str, default='/write/your/data/dir')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

def main():

    global args, best_err1, best_err5, styleDistanceMatrix, writer
    args = parser.parse_args()
    writer = SummaryWriter(args.save_dir+'/writer/'+args.method)
    if args.method == 'StyleCutMix_Auto_Gamma' :
        if args.dataset == 'cifar100':
            styleDistanceMatrix = torch.load('styleDistanceMatrix100.pt', map_location='cuda:0')
        elif args.dataset == 'cifar10':
            styleDistanceMatrix = torch.load('styleDistanceMatrix10.pt', map_location='cuda:0')
        else :
            raise Exception('unknown dataset: {}'.format(args.dataset))
        styleDistanceMatrix = styleDistanceMatrix.cpu()
        ind = torch.arange(styleDistanceMatrix.shape[1])
        styleDistanceMatrix[ind, ind] += 2 # Prevent diagonal lines from zero

    global decoder, vgg, pretrained, network_E, network_D
    if args.method.startswith('Style'):
        if args.method.startswith('StyleCutMix'):
            decoder = net_cutmix.decoder
            vgg = net_cutmix.vgg
            print("select network StyleCutMix")
            network_E = net_cutmix.Net_E(vgg)
            network_D = net_cutmix.Net_D(vgg, decoder)
        elif args.method == 'StyleMix':
            decoder = net_mixup.decoder
            vgg = net_mixup.vgg
            print("select network StyleMix")
            network_E = net_mixup.Net_E(vgg)
            network_D = net_mixup.Net_D(vgg, decoder)
        else :
            raise Exception('unknown method: {}'.format(args.method))
        decoder.eval()
        vgg.eval()
        decoder.load_state_dict(torch.load(args.decoder))
        vgg.load_state_dict(torch.load(args.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        vgg.cuda()
        decoder.cuda()
        network_E.eval()
        network_D.eval()
        network_E = torch.nn.DataParallel(network_E).cuda()
        network_D = torch.nn.DataParallel(network_D).cuda()

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
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir+'/dataCifar100/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_dir+'/dataCifar100/', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_dir+'/dataCifar10/', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
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
    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        writer.add_scalar('train_loss', train_loss, epoch+1)
        writer.add_scalar('val_loss', val_loss, epoch+1)
        writer.add_scalar('err1', err1, epoch+1)
        writer.add_scalar('err5', err5, epoch+1)
        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir, args.dataset)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    print("current_LR : ",current_LR)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        prob = np.random.rand(1)
        if prob < args.prob:
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_1 = target
            target_2 = target[rand_index]
            if args.method.startswith('StyleCutMix'):
                if args.method == 'StyleCutMix_Auto_Gamma' :
                    styleDistance = styleDistanceMatrix[target_1, target_2]
                    gamma = torch.tanh(styleDistance/args.delta)
                else :
                    gamma = np.random.beta(args.alpha2, args.alpha2)

                u = nn.Upsample(size=(224, 224), mode='bilinear')
                x1 = u(input)
                x2 = x1[rand_index]
                rs = np.random.beta(args.alpha1, args.alpha1)
                M = torch.zeros(1,1,224,224).float()
                lam_temp = np.random.beta(args.alpha1, args.alpha1)
                bbx1, bby1, bbx2, bby2 = rand_bbox(M.size(), 1.-lam_temp)
                with torch.no_grad():
                    x1_feat = network_E(x1)
                    mixImage = network_D(x1, x2, x1_feat, x1_feat[rand_index], rs, gamma, bbx1, bby1, bbx2, bby2)
                lam = ((bbx2 - bbx1)*(bby2-bby1)/(224.*224.))
                uinv = nn.Upsample(size=(32,32), mode='bilinear')
                output  = model(uinv(mixImage))

                log_preds = F.log_softmax(output, dim=-1) # dimension [batch_size, numberofclass]
                a_loss = -log_preds[torch.arange(output.shape[0]),target_1] # cross-entropy for A
                b_loss = -log_preds[torch.arange(output.shape[0]),target_2] # cross-entropy for B
                if args.method == 'StyleCutMix_Auto_Gamma':
                    gamma = gamma.cuda()
                lam_s = gamma * lam + (1.0 - gamma) * rs
                loss_c = a_loss * (lam) + b_loss * (1. - lam)
                loss_s = a_loss * (lam_s) + b_loss * (1. - lam_s)
                r = args.r
                loss = (r * loss_c + (1.0 - r) * loss_s).mean()
            elif args.method == 'StyleMix':
                u = nn.Upsample(size=(224, 224), mode='bilinear')
                x1 = u(input)
                x2 = x1[rand_index]
                rc = np.random.beta(args.alpha1, args.alpha1)
                rs = np.random.beta(args.alpha1, args.alpha1)
                with torch.no_grad():
                    x1_feat = network_E(x1)
                    mixImage = network_D(x1_feat, x1_feat[rand_index], rc, rs)
                uinv = nn.Upsample(size=(32,32), mode='bilinear')
                output  = model(uinv(mixImage))

                loss_c = rc * criterion(output, target_1)  + (1.0 - rc) * criterion(output, target_2)
                loss_s = rs * criterion(output, target_1)  + (1.0 - rs) * criterion(output, target_2)
                r = args.r
                loss = r * loss_c + (1.0-r) * loss_s
        else:
            output = model(input)
            loss = criterion(output, target)
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    print("Time taken for 1 epoch : ",time.time()-start)
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, save_dir, dataset, filename='checkpoint.pth.tar'):
    directory = save_dir+"/model/"+dataset+"/"+str(args.method)+"/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, save_dir+"/model/"+dataset+"/"+str(args.method)+'/%s/' % (args.expname) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


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
