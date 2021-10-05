import os
import logging
import json
import numpy as np
import torch

def init_pipeline(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'checkpoints'))
    os.makedirs(os.path.join(model_dir, 'figures'))
    os.makedirs(os.path.join(model_dir, 'plabels'))
    if headers is None:
        headers = ["epoch", "step", "loss"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory 
    to new directoy. """
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_ckpt(model_dir, net, epoch, lp_config=None):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    if lp_config is not None:
        # assert isinstance(lp_config,string),"low precision configuration has to be of type string"
        torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints', lp_config,
        'model-epoch{}.pt'.format(epoch)))
    else:    
        torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints', 
        'model-epoch{}.pt'.format(epoch)))

def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def clustering_accuracy(labels_true, labels_pred):
    """Compute clustering accuracy."""
    from sklearn.metrics.cluster import supervised
    from scipy.optimize import linear_sum_assignment
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)