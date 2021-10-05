from typing import List, Optional, Tuple
from multiprocessing.pool import ThreadPool
from argparse import Namespace
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import json
from collections import OrderedDict
from PIL import Image
import os


ORIGINAL_SIZE = {
    "mnist": torch.Size((1, 28, 28)),
    "fashion": torch.Size((1, 28, 28)),
    "cifar10": torch.Size((3, 32, 32)),
    "svhn": torch.Size((3, 32, 32)),
    "cifar100": torch.Size((3, 32, 32)),
    "fake":  torch.Size((3, 32, 32)),
}

MEAN_STD = {
    "mnist": {(3, 32, 32): (0.10003692801078261, 0.2752173485400458)},
    "fashion": {(3, 32, 32): (0.21899983604159193, 0.3318113789274)},
    "cifar10": {(3, 32, 32): (0.4733630111949825, 0.25156892869250536)},
    "cifar100": {(3, 32, 32): (0.478181, 0.268192)},
    "svhn": {(3, 32, 32): (0.451419, 0.199291)}
}

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "svhn": datasets.SVHN,
    "cifar100": datasets.CIFAR100
}

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def read_json(json_file):
    with open(json_file) as f:
        data_config = json.load(f)
    return data_config

def get_image_ids(order_id,orders_dict):
    ordered_dict = OrderedDict(sorted(orders_dict.items(), key=(lambda v : len(v[1])), reverse=True))
    return list(ordered_dict.items())[order_id][1]

class iNatDataset(Dataset):
    def __init__(self, json_file, root_dir, order_id, orders_dict, transform=None,target_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            order_id (int): the index of order for the dataset.
            orders_dict (dictionary['order':[image_id]]): dictionary by order name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        config_path = os.path.join(root_dir,json_file)
        self.data_config = read_json(config_path)
        self.root_dir = root_dir
        self.image_ids = get_image_ids(order_id,orders_dict)
        classes,class_to_idx = self._find_classes()
        samples = self._make_dataset(class_to_idx)
        self.samples = samples
        self.classes = classes
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
    
    def _make_dataset(self,class_to_idx):
        data_images = self.data_config["images"]
        image_dirs = [self.root_dir+data_images[i]["file_name"] for i in self.image_ids]
        classes = [self.data_config["annotations"][i]['category_id'] for i in self.image_ids]
        targets = [class_to_idx[i] for i in classes]
        res = [(image_dirs[i],targets[i]) for i in range(len(classes))]
        res.sort()
        return res

    def _find_classes(self):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [self.data_config["annotations"][i]['category_id'] for i in self.image_ids]
        classes.sort()
        classes_duplicate_removed = list(OrderedDict.fromkeys(classes))
        class_to_idx = {classes_duplicate_removed[i]: i for i in range(len(classes_duplicate_removed))}
        return classes_duplicate_removed, class_to_idx

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        # data_images = self.data_config["images"]
        # image_idx = self.image_ids[idx]
        # image_info = data_images[image_idx]
        # img_name = os.path.join(self.root_dir,image_info['file_name'])
        path, target = self.samples[idx]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

class InMemoryDataLoader(object):

    def __init__(self, data: Tensor, target: Tensor,
                 batch_size: int, shuffle: bool = True) -> None:
        self.data, self.target = data, target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__index = None

    def __len__(self) -> int:
        return self.data.size(0)

    def __iter__(self):
        randperm = torch.randperm(self.data.size(0)).to(self.data.device)
        self.data = self.data.index_select(0, randperm)
        self.target = self.target.index_select(0, randperm)
        self.__index = 0
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        start = self.__index
        if self.__index >= self.data.size(0):
            raise StopIteration
        end = min(start + self.batch_size, self.data.size(0))
        batch = self.data[start:end], self.target[start:end]
        self.__index = end
        return batch


Padding = Tuple[int, int, int, int]


def get_padding(in_size: torch.Size, out_size: torch.Size) -> Padding:
    assert len(in_size) == len(out_size)
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def load_data_async(dataset_name: str,
                    in_size: Optional[torch.Size] = None):

    original_size = ORIGINAL_SIZE[dataset_name]
    in_size = in_size if in_size is not None else original_size
    padding = get_padding(original_size, in_size)
    mean, std = MEAN_STD[dataset_name][tuple(in_size)]

    if dataset_name == "svhn":
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="train", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    if dataset_name == "svhn":
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="test", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=False, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))

    loader = DataLoader(train_data, batch_size=len(train_data),
                        num_workers=4)
    train_data, train_target = next(iter(loader))
    del loader

    loader = DataLoader(test_data, batch_size=len(test_data),
                        num_workers=4)
    test_data, test_target = next(iter(loader))
    del loader

    return train_data, train_target, test_data, test_target


class DataSetFactory(object):

    def __init__(self, all_datasets: List[str],
                 in_size: Optional[torch.Size] = None) -> None:
        self.full_data = {}
        pool = ThreadPool(processes=len(all_datasets))
        for dataset_name in all_datasets:
            self.full_data[dataset_name] = pool.apply_async(
                load_data_async, (dataset_name, in_size))

    def get_datasets(self, dataset_name: str,
                     device: torch.device,
                     args: Namespace):

        train_data, train_target, test_data, test_target = \
            self.full_data[dataset_name].get()
        train_loader = InMemoryDataLoader(train_data.to(device),
                                          train_target.to(device),
                                          shuffle=True,
                                          batch_size=args.batch_size)
        test_loader = InMemoryDataLoader(test_data.to(device),
                                         test_target.to(device),
                                         shuffle=False,
                                         batch_size=args.test_batch_size)

        return train_loader, test_loader
