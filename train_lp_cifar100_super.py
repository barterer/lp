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

# cifar100 superclass partition
def get_indices_for_superclass(lst,super_data):
    res = []
    for class_id in lst:
        for i in range(len(super_data)):
            if super_data[i][1] == class_id:
                res.append(i)
    return res

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

transforms = tf.load_transforms(args.transform)
trainset = tf.load_trainset('cifar100', transforms, path=args.data_dir)


for key in superclass_dict.keys():
    classes_ids = superclass_dict[key]
    label_map = {classes_ids[0]:0,classes_ids[1]:1,classes_ids[2]:2,classes_ids[3]:3,classes_ids[4]:4}
    sub_train_ids = get_indices_for_superclass(superclass_dict[key],trainset)
    # sub_val_ids = get_indices_for_superclass(superclass_dict[key],testset)

    sub_trainset = torch.utils.data.Subset(trainset, sub_train_ids)
    # sub_test_set = torch.utils.data.Subset(testset, sub_val_ids)
    print("Current Dataset: {}".format(key))
    # Pipelines Setup
    model_dir = os.path.join(args.save_dir,
                'lp_{}+{}_epo{}_bs{}_lr{}_mom{}_wd{}{}'.format(
                        args.arch, key, args.epo, args.bs, args.lr, args.mom, 
                        args.wd, args.tail))
    utils.init_pipeline(model_dir)
    utils.save_params(model_dir, vars(args))

    # training
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
        print("Number of classes in {} is: {}".format(key,5))
        trainloader = DataLoader(sub_trainset, batch_size=args.bs, drop_last=True, num_workers=4)
        if args.pretrain_dir is not None:
            pretrain_model_dir = os.path.join(args.pretrain_dir,"not implemented")                  
            net, _ = tf.load_checkpoint(pretrain_model_dir, args.pretrain_epo)
            utils.update_params(model_dir, pretrain_model_dir)
        else:
            net = tf.load_architectures(args.arch, low_quant_func, high_quant, 5)
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
                batch_lbls = torch.tensor([label_map[i.item()] for i in batch_lbls])
                features = net(batch_imgs.cuda())
                loss = criterion(features, batch_lbls.cuda())
                loss = loss * 1000
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # utils.save_state(model_dir, config, epoch, step, loss.item())
            if not (epoch+1) % 10:
                utils.save_ckpt(model_dir, net, epoch,config)
        end_time = time.time()
        print("training complete and takes {}s.".format(end_time-start_time))


# cifar10_save_loc = "results/cifar10/"
# if not os.path.exists(cifar10_save_loc):
#     os.makedirs(cifar10_save_loc)
# with open(cifar10_save_loc+"train_stats"+".json", 'w') as fout:
#     json.dump(res,fout)


# ## Cifar100

# ds = torchvision.datasets.CIFAR100
# path = os.path.join("./data", "CIFAR100")
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762)),
# ])

# train_set = ds(path, train=True, download=False, transform=transform_train)
# test_set = ds(path, train=False, download=False, transform=transform_test)


# for key in superclass_dict.keys():
#     sub_train_ids = get_indices_for_superclass(superclass_dict[key],train_set)
#     sub_val_ids = get_indices_for_superclass(superclass_dict[key],test_set)

#     sub_train_set = torch.utils.data.Subset(train_set, sub_train_ids)
#     sub_test_set = torch.utils.data.Subset(test_set, sub_val_ids)
#     print("Current Dataset: {}".format(key))
#     loaders = {
#             'train': torch.utils.data.DataLoader(
#                 sub_train_set,
#                 batch_size=128,
#                 shuffle=True,
#                 num_workers=4,
#                 pin_memory=True
#             ),
#             'test': torch.utils.data.DataLoader(
#                 sub_test_set,
#                 batch_size=128,
#                 num_workers=4,
#                 pin_memory=True
#             )
#     }

#     res = []
#     for i in range(len(lp_configs)):
#         current_config = lp_configs[i]
#         print(current_config)
#         low_format,high_format = current_config[0],current_config[1]
#         man_low,man_high = low_format[2],high_format[2]
#         exp_low = int(low_format[1])-man_low-1
#         exp_high = int(high_format[1])-man_high-1
#         bit_low = FloatingPoint(exp=exp_low, man=man_low)
#         bit_high = FloatingPoint(exp=exp_high, man=man_high)
#         low_quant_func = lambda: Quantizer(forward_number=bit_low, backward_number=bit_low,
#                                 forward_rounding="stochastic", backward_rounding="stochastic")
#         high_quant = Quantizer(forward_number=bit_high, backward_number=bit_high,
#                                 forward_rounding="stochastic", backward_rounding="stochastic")
#         model = LPResNet(low_quant_func,high_quant,[2,2,2,2],len(train_set.classes))
#         res.append(train_lp_model(bit_low,bit_high,model,device,loaders,hyperparam_config,epochs = 5))
# #     loss_file_name = "results/"+key+"/eval_losses"
# #     error_file_name = "results/"+key+"/eval_errors"
# #     os.makedirs(os.path.dirname(loss_file_name), exist_ok=True)
# #     np.save(loss_file_name,np.array([i['eval_loss'] for i in res]))
# #     np.save(error_file_name,np.array([i['eval_error'] for i in res]))
#     train_stats_loc = "results/"+key+"/"
#     if not os.path.exists(train_stats_loc):
#         os.makedirs(train_stats_loc)
#     with open(train_stats_loc+"train_stats"+".json", 'w') as fout:
#         json.dump(res,fout)

# bit_8 = FloatingPoint(exp=4, man=1)
# bit_16 = FloatingPoint(exp=6, man=7)
# low_quant_func = lambda: Quantizer(forward_number=bit_8, backward_number=bit_8,
#                         forward_rounding="stochastic", backward_rounding="stochastic")
# high_quant = Quantizer(forward_number=bit_16, backward_number=bit_16,
#                         forward_rounding="stochastic", backward_rounding="stochastic")
# model = LPResNet(low_quant_func,high_quant,[2,2,2,2],len(train_set.classes))


# model = model.to(device)

# summary(model,(3,224,224),batch_size=64,LP=False)


# # iNaturalist

# from datasets import iNatDataset


# # %%
# with open("/scratch/datasets/inat2019/train2019.json") as f:
#     train_config = json.load(f)
# train_orders_dict={}
# train_images=train_config["images"]
# train_categories=train_config["categories"]
# train_annotations=train_config["annotations"]
# for i in train_annotations:
#     image_id,category_id = i["image_id"],i["category_id"]
#     cur_order = train_categories[category_id]['order']
#     if cur_order in train_orders_dict:
#         train_orders_dict[cur_order].append(image_id)
#     else:
#         train_orders_dict[cur_order] = [image_id]


# # %%
# transform_train = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])


# # %%
# inat_dataset = iNatDataset('train2019.json','/scratch/datasets/inat2019/',4,train_orders_dict,transform_train)


# # %%
# inat_train_loaders,inat_test_loaders = get_balanced_dataloaders(inat_dataset,train_percentage=0.8,num_loaders=1,batch_size=32,num_workers=4)


# # %%
# loaders = {
#             'train':inat_train_loaders[0],
#             'test': inat_test_loaders[0]
#     }


# # %%
# # bit_8 = FloatingPoint(exp=5, man=2)
# # bit_16 = FloatingPoint(exp=6, man=9)
# # low_quant_func = lambda: Quantizer(forward_number=bit_8, backward_number=bit_8,
# #                         forward_rounding="stochastic", backward_rounding="stochastic")
# # high_quant = Quantizer(forward_number=bit_16, backward_number=bit_16,
# #                         forward_rounding="stochastic", backward_rounding="stochastic")
# # model = LPResNet(low_quant_func,high_quant,[2,2,2,2],len(inat_dataset_0.classes))


# # %%
# res = []
# for i in range(len(lp_configs)):
#     current_config = lp_configs[i]
#     print(current_config)
#     low_format,high_format = current_config[0],current_config[1]
#     man_low,man_high = low_format[2],high_format[2]
#     exp_low = int(low_format[1])-man_low-1
#     exp_high = int(high_format[1])-man_high-1
#     bit_low = FloatingPoint(exp=exp_low, man=man_low)
#     bit_high = FloatingPoint(exp=exp_high, man=man_high)
#     low_quant_func = lambda: Quantizer(forward_number=bit_low, backward_number=bit_low,
#                             forward_rounding="stochastic", backward_rounding="stochastic")
#     high_quant = Quantizer(forward_number=bit_high, backward_number=bit_high,
#                             forward_rounding="stochastic", backward_rounding="stochastic")
#     model = LPResNet(low_quant_func,high_quant,[2,2,2,2],len(inat_dataset.classes))
#     res.append(train_lp_model(bit_low,bit_high,model,device,loaders,hyperparam_config,epochs = 5))


# # %%
# inat_save_loc = "results/inat_order{}/".format(4)
# if not os.path.exists(inat_save_loc):
#     os.makedirs(inat_save_loc)
# with open(inat_save_loc+"train_stats"+".json", 'w') as fout:
#     json.dump(res,fout)

# model = model.to(device)


# res = train_lp_model(bit_8,bit_16,model,device,loaders,hyperparam_config,epochs = 5)

# ## Bram's Datasets

# def get_balanced_dataloader(dataset,batch_size=4,num_workers=4):
    
#     # get the samples&nclasses for weight calculation
#     image_samples = np.array(dataset.samples).tolist()
#     image_samples.sort()
#     image_samples = [(i[0],int(i[1]))for i in image_samples]
#     classes = list(OrderedDict.fromkeys([i[1]for i in image_samples]))
#     weights = make_weights_for_balanced_classes(image_samples, len(classes))
#     weights = torch.DoubleTensor(weights)
#     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=sampler,
#                                               num_workers=num_workers,pin_memory=True)


# # %%
# def calc_mean_std(train_ds,val_ds):
#     x_train = np.concatenate([np.transpose(np.asarray(train_ds[i][0]),(1, 2, 0)) for i in range(len(train_ds))])
#     x_val = np.concatenate([np.transpose(np.asarray(val_ds[i][0]),(1, 2, 0)) for i in range(len(val_ds))])
#     x = np.concatenate((x_train,x_val),axis=0)
#     # calculate the mean and std along the (0, 1) axes
#     train_mean = tuple(map(lambda x : round(x,4),tuple(np.mean(x, axis=(0, 1)))))
#     train_std = tuple(map(lambda x : round(x,4),tuple(np.std(x, axis=(0, 1)))))
#     return train_mean,train_std


# # %%
# for i in os.listdir('/scratch/datasets/bw462/deca/my_deca'):
#     if i == 'aircraft' or i == 'cifar100' or i == 'cub' or i == 'dtd':
#         continue
#     args.data = i
#     print(args.data)
#     dataset_dir = '/scratch/datasets/bw462/deca/my_deca/'+args.data
#     train_dir = os.path.join(dataset_dir,'train')
#     val_dir = os.path.join(dataset_dir,'val')
#     transform_calc_mean_std = transforms.Compose([
#         transforms.Resize((112,112)),
#         transforms.ToTensor(),
#     ])
#     train_dataset = torchvision.datasets.ImageFolder(train_dir,transform_calc_mean_std)
#     val_dataset = torchvision.datasets.ImageFolder(val_dir,transform_calc_mean_std)
    
#     train_mean, train_std = calc_mean_std(train_dataset,val_dataset)
    
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(train_mean, train_std),
#     ])
#     train_dataset = torchvision.datasets.ImageFolder(train_dir,transform)
#     val_dataset = torchvision.datasets.ImageFolder(val_dir,transform)
#     train_dataloader = get_balanced_dataloader(train_dataset,batch_size=64,num_workers=4)
#     val_dataloader = get_balanced_dataloader(val_dataset,batch_size=64,num_workers=4)
#     loaders = {
#         'train':train_dataloader,
#         'test':val_dataloader
#     }
#     init_weights_loc = 'initilizations/'+args.data+'/init_weights.pt'
#     os.makedirs(os.path.dirname(init_weights_loc), exist_ok=True)
#     model_exm = models.resnet18(pretrained=True)
#     model_exm.fc = nn.Linear(model_exm.fc.in_features, len(train_dataset.classes))
#     torch.save(model_exm.state_dict(), init_weights_loc)
#     del model_exm
#     res = []
#     for i in range(len(lp_configs)):
#         current_config = lp_configs[i]
#         print(current_config)
#         low_format,high_format = current_config[0],current_config[1]
#         man_low,man_high = low_format[2],high_format[2]
#         exp_low = int(low_format[1])-man_low-1
#         exp_high = int(high_format[1])-man_high-1
#         bit_low = FloatingPoint(exp=exp_low, man=man_low)
#         bit_high = FloatingPoint(exp=exp_high, man=man_high)
#         low_quant_func = lambda: Quantizer(forward_number=bit_low, backward_number=bit_low,
#                                 forward_rounding="stochastic", backward_rounding="stochastic")
#         high_quant = Quantizer(forward_number=bit_high, backward_number=bit_high,
#                                 forward_rounding="stochastic", backward_rounding="stochastic")
#         model = LPResNet(low_quant_func,high_quant,[2,2,2,2],len(train_dataset.classes))
        
#         model.load_state_dict(torch.load(init_weights_loc))
#         res.append(train_lp_model(bit_low,bit_high,model,device,loaders,hyperparam_config,epochs = 5))
#     eval_loss_file_name = "results/"+args.data+"/eval_losses"
#     eval_error_file_name = "results/"+args.data+"/eval_errors"
#     train_loss_file_name = "results/"+args.data+"/train_losses"
#     train_error_file_name = "results/"+args.data+"/train_errors"
#     os.makedirs(os.path.dirname(eval_loss_file_name), exist_ok=True)
#     np.save(eval_loss_file_name,np.array([i[1]['eval_loss'] for i in res]))
#     np.save(eval_error_file_name,np.array([i[1]['eval_error'] for i in res]))
#     np.save(train_loss_file_name,np.array([i[0]['train_loss'] for i in res]))
#     np.save(train_error_file_name,np.array([i[0]['train_error'] for i in res]))
