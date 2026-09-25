import argparse
import random
import numpy as np
import torchaudio
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
from utils.augmentation import ToPILImage, Resize, ToTensor
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import SubsetRandomSampler
from utils.cifar10_dvs import CIFAR10DVS
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import surrogate as surrogate_self
from modules import neuron
from models import spiking_resnet_imagenet, spiking_resnet, spiking_vgg_bn
from modules.margin_resculpted_surrogate import MarginResculptedSurrogate

def get_args():
    parser = argparse.ArgumentParser(description='MRL for SNN')
    parser.add_argument('-seed', default=2026, type=int)
    parser.add_argument('-name', default='', type=str, help='specify a name for the checkpoint and log files')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str,
                        help='should be cifar10, cifar100, dvscifar10, dvsgesture, or imagenet')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-surrogate', default='tri', type=str,
                        help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-pre_train', type=str, help='load a pretrained model. used for imagenet')
    parser.add_argument('-amp', action='store_false', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or AdamW', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-vth', default=1.0, type=float, help='v_threshold of the neuron')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=200, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='vgg11', help='use which SNN model')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate. used for DVSCIFAR10 and dvsgesture')
    parser.add_argument('-weight_decay', type=float, default=5e-4)
    parser.add_argument('-loss_lambda', type=float, default=0.05,
                        help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='loss function setting')
    parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')
    parser.add_argument('-save_init', action='store_true', help='save the initialization of parameters')

    # Adv
    parser.add_argument('-attack', default='clean_nosym', type=str, help='attack mode: empty, fgsm, or pgd')
    parser.add_argument('-eps', default=2, type=float, metavar='N', help='attack eps')
    # PGD
    parser.add_argument('-alpha', default=0.01, type=float, metavar='N', help='pgd attack alpha')
    parser.add_argument('-steps', default=7, type=int, metavar='N', help='pgd attack steps')

    # Margin Resculpted Surrogate
    parser.add_argument('-mrl_gamma', default=0.1, type=float, help='margin resculpted surrogate gamma')


    args = parser.parse_args()
    print(args)

    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)

    return args

def get_data(b, j, T, data_dir, dataset='cifar10'):
    if dataset == 'cifar10' or dataset == 'cifar100':
        c_in = 3
        if dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        trainset = dataloader(root=data_dir, train=True, download=True, transform=transform_train)
        train_data_loader = data.DataLoader(trainset, batch_size=b, shuffle=True, num_workers=j, drop_last=True)

        testset = dataloader(root=data_dir, train=False, download=False, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=b, shuffle=False, num_workers=j, drop_last=True)



    elif dataset == 'imagenet':
        num_classes = 1000
        data_dir = os.path.join(data_dir, 'imagenet')
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=b, shuffle=True,
            num_workers=j, pin_memory=True, drop_last=True)

        test_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=b, shuffle=False,
            num_workers=j, pin_memory=True, drop_last=True)


    return train_data_loader, test_data_loader, c_in, num_classes

def get_net(surrogate='tri', dataset='cifar10', model='vgg5', num_classes=10, drop_rate=0.0, tau=1.5, c_in=3, vth=1.0, gamma=0.0):

    if gamma > 0.0:
        if surrogate == 'sig':
            base_surrogate = surrogate_sj.Sigmoid()
        elif surrogate == 'rec':
            base_surrogate = surrogate_self.Rectangle()
        elif surrogate == 'tri':
            base_surrogate = surrogate_sj.PiecewiseQuadratic()
        surrogate_function = MarginResculptedSurrogate(base_surrogate=base_surrogate, gamma=gamma)
    else:
        if surrogate == 'sig':
            surrogate_function = surrogate_sj.Sigmoid()
        elif surrogate == 'rec':
            surrogate_function = surrogate_self.Rectangle()
        elif surrogate == 'tri':
            surrogate_function = surrogate_sj.PiecewiseQuadratic()

    neuron_model = neuron.BPTTNeuron

    if dataset == 'cifar10' or dataset == 'cifar100' or dataset=='tinyimagenet':
        if 'resnet' in model:
            net = spiking_resnet.__dict__[model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=drop_rate,
                                                  tau=tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1, v_threshold=vth)
            print('using Resnet model.')
        elif 'vgg' in model:
            net = spiking_vgg_bn.__dict__[model](neuron=neuron_model, num_classes=num_classes,
                                                      neuron_dropout=drop_rate, tau=tau, surrogate_function=surrogate_function, c_in=c_in,
                                                      fc_hw=1, v_threshold=vth)
            print('using VGG model.')

    elif dataset == 'imagenet':
        net = spiking_resnet_imagenet.__dict__[model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=drop_rate,
                                                           tau=tau, surrogate_function=surrogate_function, c_in=3)
        print('using NF-Resnet model.')

    else:
        raise NotImplementedError
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()
    return net

def cali_sampler(trainset, cali_num, b, j):
    cali_num_sample = cali_num * b
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    calibrate_indices = indices[:cali_num_sample]
    calibrate_sampler = SubsetRandomSampler(calibrate_indices)
    calibrate_data_loader = data.DataLoader(
        trainset,
        batch_size=b,
        sampler=calibrate_sampler,
        num_workers=j,
    )
    return calibrate_data_loader