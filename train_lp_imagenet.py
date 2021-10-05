import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import time
from robustness import datasets

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
parser.add_argument('--start_task_id', type=int, default='0',
                    help='starting task id')
parser.add_argument('--tasks', type=int, default='5',
                    help='number of tasks')
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

# imagenet setting
in_path = 'datasets/imagenet64x64'
in_info_path = 'datasets/imagenet64x64'

from robustness.tools.imagenet_helpers import ImageNetHierarchy
in_hier = ImageNetHierarchy(in_path,
                            in_info_path)

superclasses_list = []
for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(reversed(in_hier.wnid_sorted)):
    if ndesc_in >= 5:
        #print(f"WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")
        superclasses_list.append(wnid)
#NEED TO CHANGE
subclass_id_dict = set()
count = 0
used = 0
start = args.start_task_id
i = 0
superclass_ids = []
class_ranges_list = []
superclass_names = []
#label_map_list = []

while i < args.tasks:
    ancestor_wnid = superclasses_list[count]
    # print(f"Superclass | WordNet ID: {ancestor_wnid}, Name: {in_hier.wnid_to_name[ancestor_wnid]}")
    class_ranges, _ = in_hier.get_subclasses([ancestor_wnid],balanced=True)
    class_ranges_no_dup = set()
    for idx, class_id_imgnet in enumerate(class_ranges[0]):
        if not (class_id_imgnet in subclass_id_dict):
            class_ranges_no_dup.add(class_id_imgnet)
            subclass_id_dict.add(class_id_imgnet)
    label = len(class_ranges_no_dup)
    class_ranges = [{list(class_ranges_no_dup)[i]} for i in range(len(class_ranges_no_dup))]
    
#     label = 0
#     label_map = {}
#     for cnt, wnid in enumerate(in_hier.tree[ancestor_wnid].descendants_all):
#         if wnid in in_hier.in_wnids:
# #             print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")
#             if wnid in subclasses_dict:
                
#             else:
#                 label_map[label] = in_hier.wnid_to_name[wnid]
#                 label += 1
#             subclasses_dict.add(wnid)
    if label > 4:
        used += 1
    if label > 4 and used > start:
        superclass_names.append(in_hier.wnid_to_name[ancestor_wnid])
        superclass_ids.append(ancestor_wnid)
        class_ranges_list.append(class_ranges)
#         label_map_list.append(label_map)
        i += 1
    count += 1 

for id in range(args.tasks):
    # Pipelines Setup
    model_dir = os.path.join(args.save_dir,'imagenet_subsets',
                'lp_{}+{}_epo{}_bs{}_lr{}_mom{}_wd{}{}'.format(
                        args.arch, superclass_ids[id], args.epo, args.bs, args.lr, args.mom, 
                        args.wd, args.tail))
    utils.init_pipeline(model_dir)
    utils.save_params(model_dir, vars(args))

    print("Using Superclass {} with WordNetid {}".format(superclass_names[id],superclass_ids[id]))

    transforms = tf.load_transforms(args.transform)
    class_ranges = class_ranges_list[id]
    custom_dataset = datasets.CustomImageNet(in_path,
                                            class_ranges,transform_train=transforms,transform_test=transforms)

    train_loader, test_loader = custom_dataset.make_loaders(workers=10,
                                                            batch_size=5)

    print(f"Train set size: {len(train_loader.dataset.targets)}")
    print(f"Test set size: {len(test_loader.dataset.targets)}")

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

        # transforms = tf.load_transforms(args.transform)
        # trainset = tf.load_trainset(args.data, transforms, path=args.data_dir)
        # print("Number of classes in {} is: {}".format(args.data,trainset.num_classes))
        # trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, num_workers=4)
        trainloader, testloader = custom_dataset.make_loaders(workers=4,batch_size=args.bs)
        if args.pretrain_dir is not None:
            pretrain_model_dir = os.path.join(args.pretrain_dir,"not implemented")                  
            net, _ = tf.load_checkpoint(pretrain_model_dir, args.pretrain_epo)
            utils.update_params(model_dir, pretrain_model_dir)
        else:
            net = tf.load_architectures(args.arch, low_quant_func, high_quant, len(class_ranges))
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

