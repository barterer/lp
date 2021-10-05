import argparse
import os
from tqdm import tqdm
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from robustness import datasets

from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint
from qtorch.auto_low import sequential_lower
import math


import train_func as tf
import utils
import itertools

def calc_acc(test_features, test_labels):
    _, test_pred = torch.max(test_features, 1)
    # test_pred = test_pred.values.detach()
    acc = utils.compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("Test Acc: {}".format(acc))
    return acc

# cifar100 superclass partition
def get_indices_for_superclass(lst,super_data):
    res = []
    for class_id in lst:
        for i in range(len(super_data)):
            if super_data[i][1] == class_id:
                res.append(i)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')  
    parser.add_argument('--save_dir', type=str, default='./saved_stats/',
                    help='base directory for saving evaluation stats. (default: ./saved_stats/)')
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
    error_matrix = np.zeros((88,99))
    # CIFAR10
    base_model_dir = "saved_models/lp_resnet18+cifar10_epo10_bs32_lr0.001_mom0.9_wd0.0005"

    print("Start Evaluating on CIFAR10")
    #print("Start Evaluation on dataset: {}".format(args.data_dir))
    params = utils.load_params(base_model_dir)
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False, path='./data/')
    testloader = DataLoader(testset, batch_size=200)
    res = {}
    for i in range(len(lp_configs)):
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
        ## load model     
        net, epoch = tf.load_checkpoint(base_model_dir, low_quant_func, high_quant, testset.num_classes, None, eval_=True, lp_config=config)
        net = net.cuda().eval()
        # get test features and labels
        test_features, test_labels = tf.get_features(net, testloader, verbose=False)
        acc = calc_acc(test_features, test_labels)
        res[config] = acc
        error_matrix[0,i] = acc
    
    # save_loc = os.path.join(args.save_dir,params['data'])
    # if not os.path.exists(save_loc):
    #     os.makedirs(save_loc)
    # with open(os.path.join(save_loc,"eval_stats.json"), 'w') as fout:
    #     json.dump(res,fout)

    # CIFAR100
    print("Start Evaluating on CIFAR100")
    base_model_dir_list = ["saved_models/cifar100_super/lp_resnet18+","_epo10_bs32_lr0.001_mom0.9_wd0.0005"]

    superclass_dict = {'aquatic mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food containers': [9, 10, 16, 28, 61],
    'fruit and vegetables': [0, 51, 53, 57, 83],
    'household electrical devices': [22, 39, 40, 86, 87],
    'household furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large_carnivores': [3, 42, 43, 88, 97],
    'large man-made outdoor things': [12, 17, 37, 68, 76],
    'large natural outdoor scenes': [23, 33, 49, 60, 71],
    'large omnivores and herbivores': [15, 19, 21, 31, 38],
    'medium-sized mammals': [34, 63, 64, 66, 75],
    'non-insect invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles 1': [8, 13, 48, 58, 90],
    'vehicles 2': [41, 69, 81, 85, 89]}

    transforms = tf.load_transforms("default")
    testset = tf.load_trainset('cifar100', transforms,train=False, path='./data/')


    for key_id,key in enumerate(superclass_dict.keys()):
        classes_ids = superclass_dict[key]
        label_map = {classes_ids[0]:0,classes_ids[1]:1,classes_ids[2]:2,classes_ids[3]:3,classes_ids[4]:4}
        sub_val_ids = get_indices_for_superclass(superclass_dict[key],testset)
        sub_testset = torch.utils.data.Subset(testset, sub_val_ids)
        print("Current Dataset: {}".format(key))
        base_model_dir = base_model_dir_list[0]+key+base_model_dir_list[1]
        params = utils.load_params(base_model_dir)
        test_transforms = tf.load_transforms('test')
        testloader = DataLoader(sub_testset, batch_size=200)
        res = {}
        for i in range(len(lp_configs)):
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
            ## load model     
            net, epoch = tf.load_checkpoint(base_model_dir, low_quant_func, high_quant, 5, None, eval_=True, lp_config=config)
            net = net.cuda().eval()
            # get test features and labels
            test_features, test_labels = tf.get_features(net, testloader, verbose=False)
            test_labels = torch.tensor([label_map[i.item()] for i in test_labels])
            acc = calc_acc(test_features, test_labels)
            res[config] = acc
            error_matrix[1+key_id,i] = acc
        
        save_loc = os.path.join(args.save_dir,"cifar100_super",key)
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        with open(os.path.join(save_loc,"eval_stats.json"), 'w') as fout:
            json.dump(res,fout)

    # DECA
    print("Start Evaluating on DECA")
    base_model_dir_list = ["saved_models/lp_resnet18+","_epo10_bs32_lr0.001_mom0.9_wd0.0005"]
    ds_dir = "/dir/to/deca"
    dataset_list = os.listdir(ds_dir)

    for ds_id, ds_name in enumerate(dataset_list):
        base_model_dir = base_model_dir_list[0]+ds_name+base_model_dir_list[1]
        params = utils.load_params(base_model_dir)
        test_transforms = tf.load_transforms('deca_test')
        testset = tf.load_trainset(ds_name, test_transforms, train=False, path='datasets/my_deca')
        print("{} has test size: {}".format(ds_name, len(testset)))
        testloader = DataLoader(testset, batch_size=100)
        res = {}
        for i in range(len(lp_configs)):
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
            ## load model     
            net, epoch = tf.load_checkpoint(base_model_dir, low_quant_func, high_quant, testset.num_classes, None, eval_=True, lp_config=config)
            net = net.cuda().eval()
            # get test features and labels
            test_features, test_labels = tf.get_features(net, testloader, verbose=False)
            acc = calc_acc(test_features, test_labels)
            res[config] = acc
            error_matrix[21+ds_id,i] = acc
        
        save_loc = os.path.join(args.save_dir,"deca",ds_name)
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        with open(os.path.join(save_loc,"eval_stats.json"), 'w') as fout:
            json.dump(res,fout)

    #imagenet
    print("Start Evaluating on imagenet")
    base_model_dir_list = ["saved_models/imagenet_subsets/lp_resnet18+","_epo10_bs32_lr0.001_mom0.9_wd0.0005"]

    in_path = 'dir/to/imagenet64x64'
    in_info_path = 'dir/to/imagenet64x64'

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
    start = 0
    i = 0
    superclass_ids = []
    class_ranges_list = []
    superclass_names = []
    #label_map_list = []

    while i < 50:
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
        if label > 4:
            used += 1
        if label > 4 and used > start:
            superclass_names.append(in_hier.wnid_to_name[ancestor_wnid])
            superclass_ids.append(ancestor_wnid)
            class_ranges_list.append(class_ranges)
            i += 1
        count += 1 

    for id in range(50):
        print("Evaluating on Superclass {} with WordNetid {}".format(superclass_names[id],superclass_ids[id]))
        base_model_dir = base_model_dir_list[0]+superclass_ids[id]+base_model_dir_list[1]
        transforms = tf.load_transforms("deca_test")
        class_ranges = class_ranges_list[id]
        custom_dataset = datasets.CustomImageNet(in_path,
                                                class_ranges,transform_train=transforms,transform_test=transforms)

        _, testloader = custom_dataset.make_loaders(workers=10,batch_size=100)

        res = {}
        for i in range(len(lp_configs)):
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
            ## load model     
            net, epoch = tf.load_checkpoint(base_model_dir, low_quant_func, high_quant, len(class_ranges), None, eval_=True, lp_config=config)
            net = net.cuda().eval()
            # get test features and labels

            #correct = 0.0

            # if phase=="train": model.train()
            # elif phase=="eval": model.eval()

            
            # with torch.autograd.set_grad_enabled(phase=="train"):
            #     for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            #         input = input.to(device=device)
            #         target = target.to(device=device)
            #         output = model(input)
            #         loss = criterion(output, target)
            #         loss_sum += loss.cpu().item() * input.size(0)
            #         # pred = output.data.max(1, keepdim=True)[1]
            #         _, pred = torch.max(output, 1)
            #         correct += pred.eq(target.data.view_as(pred)).sum()

            #         if phase=="train":
            #             loss = loss * 1000
            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
            #         # correct += torch.sum(pred == target.data)
            #         # loss_sum += loss.cpu().item() * input.size(0)
            # correct = correct.cpu().item()
            # ttl = len(loader.dataset)
            test_features, test_labels = tf.get_features(net, testloader, verbose=False)
            acc = calc_acc(test_features, test_labels)
            res[config] = acc
            error_matrix[38+id,i] = acc
        
        save_loc = os.path.join(args.save_dir,"imagenet_subsets",superclass_ids[id])
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        with open(os.path.join(save_loc,"eval_stats.json"), 'w') as fout:
            json.dump(res,fout)

    print("saving error matrix...")
    np.save("error_matrix_all",error_matrix)

