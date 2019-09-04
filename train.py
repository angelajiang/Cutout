# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import argparse
import os
import pdb
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet


sys.path.insert(0, "/users/ahjiang/src/Cutout/pytorch-cifar")
sys.path.insert(0, "/home/ahjiang/Cutout/pytorch-cifar")
from lib.SelectiveBackpropper import SelectiveBackpropper
#import main as sb
import lib.cifar
import lib.datasets
import lib.svhn

start_time_seconds = time.time()

model_options = ['resnet18', 'wideresnet']
strategy_options = ['nofilter', 'sb', 'kath']
dataset_options = ['cifar10', 'cifar100', 'svhn']
calculator_options = ['relative', 'random', 'hybrid']
fp_selector_options = ['alwayson', 'stale']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=200,
#                    help='number of epochs to train (default: 20)')
parser.add_argument('--hours', type=float, default=12,
                    help='number of hours to train (default: 12)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--output_dir', default="./logs",
                    help='directory to place logs')

parser.add_argument('--kath_oversampling_rate', type=int, default=0,
                    help='oversampling rate for kath')
parser.add_argument('--prob_pow', type=int, default=3,
                    help='dictates SB selectivity')
parser.add_argument('--staleness', type=int, default=2,
                    help='Number of epochs to use stale losses for fp_selector')
parser.add_argument('--lr_sched', default=None,
                    help='path to file with manual lr schedule')
parser.add_argument('--sb', action='store_true', default=False,
                    help='apply selective backprop')
parser.add_argument('--forwardlr', dest='forwardlr', action='store_true',
                    help='LR schedule based on forward passes')
parser.add_argument('--strategy', default='nofilter', choices=strategy_options)
parser.add_argument('--calculator', default='relative', choices=calculator_options)
parser.add_argument('--fp_selector', default='alwayson', choices=fp_selector_options)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.cuda
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


print(args)

test_id = args.dataset + '_' + args.model

# Prepare selective backprop things
if args.sb:
    filename = args.output_dir + "/" + test_id + '_sb.csv'
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        dataset_lib = lib.cifar
    elif args.dataset == 'svhn':
        dataset_lib = lib.svhn
    else:
        print("{} dataset not supported with SB".format(args.dataset))
        exit()

else:
    filename = args.output_dir + "/" + test_id + '.csv'
    dataset_lib = datasets


# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = dataset_lib.CIFAR10(#root='/ssd/datasets/cifar10/',
                                        root="data/",
                                        train=True,
                                        transform=train_transform,
                                        download=True)

    test_dataset = dataset_lib.CIFAR10(#root='/ssd/datasets/cifar10/',
                                       root="data/",
                                       train=False,
                                       transform=test_transform,
                                       download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = dataset_lib.CIFAR100(#root='/ssd/datasets/cifar100/',
                                         root="data/",
                                         train=True,
                                         transform=train_transform,
                                         download=True)

    test_dataset = dataset_lib.CIFAR100(#root='/ssd/datasets/cifar100/',
                                        root="data/",
                                        train=False,
                                        transform=test_transform,
                                        download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = dataset_lib.SVHN(#root='/ssd/datasets/svhn/',
                                     root="data/",
                                     split='train',
                                     transform=train_transform,
                                     download=True)

    extra_dataset = dataset_lib.SVHN(#root='/ssd/datasets/svhn/',
                                     root="data/",
                                     split='extra',
                                     transform=train_transform,
                                     download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = dataset_lib.SVHN(root='/ssd/datasets/svhn/',
                                    split='test',
                                    transform=test_transform,
                                    download=True)

# Data Loader (Input Pipeline)
static_dataset = [a for a in train_dataset]
static_dataset = static_dataset[:512]
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

sb = SelectiveBackpropper(cnn,
                          cnn_optimizer,
                          args.prob_pow,
                          args.batch_size,
                          args.lr_sched,
                          num_classes,
                          len(train_dataset),
                          args.forwardlr,
                          args.strategy,
                          args.kath_oversampling_rate,
                          args.calculator,
                          args.fp_selector,
                          args.staleness)

def test_sb(loader, epoch, sb):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels, ids in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
            loss = nn.CrossEntropyLoss()(pred, labels)
            test_loss += loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_loss /= total
    val_acc = correct / total

    print('test_debug,{},{},{},{:.6f},{:.6f},{},{}'.format(
                epoch,
                sb.logger.global_num_backpropped,
                sb.logger.global_num_skipped,
                test_loss,
                100.*val_acc,
                sb.logger.global_num_skipped_fp,
                time.time() - start_time_seconds))
    cnn.train()
    return 100. * val_acc

def test(loader, epoch, num_images):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    test_loss = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)
            loss = nn.CrossEntropyLoss()(pred, labels)
            test_loss += loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_loss /= total
    val_acc = correct / total

    print('test_debug,{},{},{},{:.6f},{:.6f},{}'.format(
                epoch,
                epoch * num_images,
                0,
                test_loss,
                100.*val_acc,
                time.time()))

    cnn.train()
    return 100. * val_acc


stopped = False 
epoch = -1

while (time.time() - start_time_seconds < args.hours * 3600.):
    epoch += 1

    if args.sb:
        if stopped: break

        sb.trainer.train(train_loader)

        if sb.trainer.stopped:
            stopped = True
            break

        sb.next_epoch()
        sb.next_partition()
        test_acc = test_sb(test_loader, epoch, sb)

    else:
        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        #progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            #progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = 100. * correct / total

            print("Epoch: {} Acc: {:.3f} Loss: {:.3f}".format(epoch,
                                                              accuracy,
                                                              xentropy_loss_avg))

            #progress_bar.set_postfix(
            #    xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            #    acc='%.3f' % accuracy)

        test_acc = test(test_loader, epoch, len(train_loader.dataset))
        #tqdm.write('test_acc: %.3f' % (test_acc))
        scheduler.step(epoch)
        row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
        #csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
#csv_logger.close()
