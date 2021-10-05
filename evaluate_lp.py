import argparse
import os
from tqdm import tqdm
import json

import torch
import numpy as np
from torch.utils.data import DataLoader


os.environ["TORCH_EXTENSIONS_DIR"] = '/tmp/qtorch'

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, help='base directory for saving PyTorch model.')
    parser.add_argument('--epoch', type=int, default=None, help='which epoch for evaluation')
    
    parser.add_argument('--save_dir', type=str, default='./saved_stats/',
                    help='base directory for saving evaluation stats. (default: ./saved_stats/)')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
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

    print("Start Evaluation on dataset: {}".format(args.data_dir))
    params = utils.load_params(args.model_dir)
    test_transforms = tf.load_transforms('test')
    testset = tf.load_trainset(params['data'], test_transforms, train=False, path=args.data_dir)
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
        
        net, epoch = tf.load_checkpoint(args.model_dir, low_quant_func, high_quant, testset.num_classes, args.epoch, eval_=True, lp_config=config)
        net = net.cuda().eval()

        # get test features and labels
        test_features, test_labels = tf.get_features(net, testloader)

        acc = calc_acc(test_features, test_labels)
        res[config] = acc
    
    save_loc = os.path.join(args.save_dir,params['data'])
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    with open(os.path.join(save_loc,"eval_stats.json"), 'w') as fout:
        json.dump(res,fout)

    
