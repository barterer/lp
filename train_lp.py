import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time

from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint
from qtorch.auto_low import sequential_lower
import math

import itertools
import train_func as tf
import utils

parser = argparse.ArgumentParser(description='Low Precision Learning')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=500,
                    help='number of epochs for training (default: 500)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for loading datasets. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
args = parser.parse_args()

# low precision configurations
low_formats = [['Representation_Float', '5', 1],['Representation_Float', '6', 2],
               ['Representation_Float', '7', 3],['Representation_Float', '8', 4],
               ['Representation_Float', '6', 1],['Representation_Float', '7', 2],
               ['Representation_Float', '8', 3],['Representation_Float', '9', 4],
               ['Representation_Float', '7', 1],['Representation_Float', '8', 2],
               ['Representation_Float', '9', 3]]

high_formats = [['Representation_Float', '14', 7],['Representation_Float', '16', 9],
                ['Representation_Float', '18', 11],['Representation_Float', '15', 7],
                ['Representation_Float', '17', 9],['Representation_Float', '19', 11],
                ['Representation_Float', '16', 7],['Representation_Float', '18', 9],
                ['Representation_Float', '20', 11]]

lp_configs = list(itertools.product(low_formats,high_formats))

# Pipelines Setup
model_dir = os.path.join(args.save_dir,
               'lp_{}+{}_epo{}_bs{}_lr{}_mom{}_wd{}{}'.format(
                    args.arch, args.data, args.epo, args.bs, args.lr, args.mom, 
                    args.wd, args.tail))
utils.init_pipeline(model_dir)
utils.save_params(model_dir, vars(args))

# training
res = []
for i in range(len(lp_configs)):
    start_time = time.time()
    current_config = lp_configs[i]
    print("Currently trains on : {}".format(current_config))
    low_format,high_format = current_config[0],current_config[1]
    man_low,man_high = low_format[2],high_format[2]
    exp_low = int(low_format[1])-man_low-1
    exp_high = int(high_format[1])-man_high-1
    bit_low = FloatingPoint(exp=exp_low, man=man_low)
    bit_high = FloatingPoint(exp=exp_high, man=man_high)
    low_quant_func = lambda: Quantizer(forward_number=bit_low, backward_number=bit_low,
                            forward_rounding="stochastic", backward_rounding="stochastic")
    high_quant = Quantizer(forward_number=bit_high, backward_number=bit_high,
                            forward_rounding="stochastic", backward_rounding="stochastic")
    config = "l_{}_{}_h_{}_{}".format(low_format[1],low_format[2],high_format[1],high_format[2])
    # model = LPResNet_baby(low_quant_func,high_quant,[2,2,2,2],len(train_set.classes))

    ## Prepare for Training
    transforms = tf.load_transforms(args.transform)
    trainset = tf.load_trainset(args.data, transforms, path=args.data_dir)
    print("Number of classes in {} is: {}".format(args.data,trainset.num_classes))
    trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, num_workers=4)
    if args.pretrain_dir is not None:
        pretrain_model_dir = os.path.join(args.pretrain_dir,"not implemented")                  
        net, _ = tf.load_checkpoint(pretrain_model_dir, args.pretrain_epo)
        utils.update_params(model_dir, pretrain_model_dir)
    else:
        net = tf.load_architectures(args.arch, low_quant_func, high_quant, trainset.num_classes)
    criterion = nn.CrossEntropyLoss()
    weight_quant = quantizer(forward_number=bit_low,
                        forward_rounding="stochastic")
    grad_quant = quantizer(forward_number=bit_low,
                            forward_rounding="stochastic")
    momentum_quant = quantizer(forward_number=bit_high,
                            forward_rounding="stochastic")
    acc_quant = quantizer(forward_number=bit_high,
                            forward_rounding="stochastic")
    optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    optimizer = OptimLP(optimizer,
                        weight_quant=weight_quant,
                        grad_quant=grad_quant,
                        momentum_quant=momentum_quant,
                        acc_quant=acc_quant,
                        grad_scaling=1/1000
    )
    os.makedirs(os.path.join(model_dir, 'checkpoints',config))
    for epoch in range(args.epo):
        for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
            features = net(batch_imgs.cuda())
            loss = criterion(features, batch_lbls.cuda())
            loss = loss * 1000
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            utils.save_state(model_dir, config, epoch, step, loss.item())
        if not (epoch+1) % 10:
            utils.save_ckpt(model_dir, net, epoch,config)
    end_time = time.time()
    print("training complete and takes {}s.".format(end_time-start_time))
