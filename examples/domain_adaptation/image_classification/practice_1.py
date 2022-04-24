"""
@author: Jin Shin
@contact: shinjin0103@seoultech.ac.kr
"""

'''
import: Inner python packages
Path Maker
'''
import random
import time
import warnings
import sys
import argparse
import shutil
import os
import os.path as osp
import signal
import time
import traceback
ROOT_CODE_DIR = 'C:/Users/shinj/Source/Repos/Transfer-Learning-Library'
root_code_dir = os.path.abspath(ROOT_CODE_DIR)
#print(root_code_dir)
sys.path.append(root_code_dir)
def registry_dirs(cur,count):
    loc_count = count + 1
    dirs = os.listdir(cur)
    if count > 2:
        return
    for dir in dirs:
        cur_dir = str(cur).replace('\\','/') +'/' + dir
        if osp.isdir(cur_dir) and cur_dir.find('.') < 0 :
            sys.path.append(os.path.abspath(cur_dir))
            registry_dirs(os.path.abspath(cur_dir), loc_count)
        else:
            pass
registry_dirs(ROOT_CODE_DIR, 0)

def handler(signum, frame):
    print("caught Ctrl+C signal ")
    exit()

signal.signal(signal.SIGINT, handler)

'''
import: Custom Files
'''
import utils

'''
import: Pytorch Packages
'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F



def main(arg: argparse.Namespace):
    #pass
    #arg.d = 'd'
    print(f'root : {arg.root}')
    print(f'data : {arg.data}')
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    print(args.train_resizing)
    print(args.val_resizing)

    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    
    train_source_dataset, num_classes, args.class_names = \
        utils.get_dataset_only(args.data, args.root, args.source, train_transform)
    
    print('Dataset END')
    print(type(train_source_dataset))
    print('Main Process Finished ')
    return

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
        # dataset parameters
        parser.add_argument('root', metavar='DIR',
                            help='root path of dataset')
        parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                            help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                                 ' (default: Office31)')
        parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
        parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
        parser.add_argument('--train-resizing', type=str, default='default')
        parser.add_argument('--val-resizing', type=str, default='default')
        parser.add_argument('--resize-size', type=int, default=224,
                            help='the image size after resizing')
        parser.add_argument('--no-hflip', action='store_true',
                            help='no random horizontal flipping during training')
        parser.add_argument('--norm-mean', type=float, nargs='+',
                            default=(0.485, 0.456, 0.406), help='normalization mean')
        parser.add_argument('--norm-std', type=float, nargs='+',
                            default=(0.229, 0.224, 0.225), help='normalization std')
        # model parameters
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                            choices=utils.get_model_names(),
                            help='backbone architecture: ' +
                                 ' | '.join(utils.get_model_names()) +
                                 ' (default: resnet18)')
        parser.add_argument('--bottleneck-dim', default=256, type=int,
                            help='Dimension of bottleneck')
        parser.add_argument('--no-pool', action='store_true',
                            help='no pool layer after the feature extractor.')
        parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
        parser.add_argument('--trade-off', default=1., type=float,
                            help='the trade-off hyper-parameter for transfer loss')
        # training parameters
        parser.add_argument('-b', '--batch-size', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 32)')
        parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
        parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                            metavar='W', help='weight decay (default: 1e-3)',
                            dest='weight_decay')
        parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 2)')
        parser.add_argument('--epochs', default=20, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                            help='Number of iterations per epoch')
        parser.add_argument('-p', '--print-freq', default=100, type=int,
                            metavar='N', help='print frequency (default: 100)')
        parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--per-class-eval', action='store_true',
                            help='whether output per-class accuracy during evaluation')
        parser.add_argument("--log", type=str, default='dann',
                            help="Where to save logs, checkpoints and debugging images.")
        parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                            help="When phase is 'test', only test the model."
                                 "When phase is 'analysis', only analysis the model.")
        args = parser.parse_args()
    
        main(args)
        while True:
            time.sleep(5)
            print('Inf Loops...')
            pass
    except Exception as e: 
        print('Exception Main Trial')
        print(e)
        print(e.with_traceback)
        print(e.args)
        print(traceback.format_exc())
        while True:
            print('Inf Loops...(error)')
            pass
        