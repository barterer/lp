# Code for How Low Can We Go: Trading Memory for Error in Low-Precision Training 

## Datasets
- DECA datasets (download [here](https://drive.google.com/file/d/1og7w_E3_CQh_S0kuIEMIlfd7KOEl7_SB/view?usp=sharing))
- ImageNet 64x64 (download [here](http://image-net.org/small/download.php))

## Requirements
The code is verified to run on Python 3.7.3, together with the following libraries:

- torch 1.8.0

- torchvision 0.4.0

- qtorch 0.2.0 

## User Guide

### Data Collection

To train on a specific dataset(e.g. CIFAR-10) on the 99 low precision configurations:

``python train_lp.py --arch resnet18 --data cifar10  --epo 10 --bs 32 --lr 0.001 ``

To train on CIFAR100 partitions or DECA datasets:

``python train_lp_cifar100_super.py --arch resnet18  --epo 10 --bs 32 --lr 0.001 ``

``python train_lp_deca.py --arch resnet18 --transform deca_train  --epo 10 --bs 32 --lr 0.001 --data_dir dir/to/deca ``

To train on ImageNet partitions (`start_task_id` refers to the "task_id" of the first task to run and tasks refers to the number of tasks to run. Given the indices in the first column of Appendix Table 1, The "task_id"=index-38):

``python train_lp_imagenet.py --arch resnet18 --start_task_id 0 --tasks 5 --epo 10 --bs 32 --lr 0.001 --transform deca_train ``

To evaluate on all trained datasets and produce the error matrix:

``python evaluate_lp_all.py ``

To calculate the memory matrix, use the Jupyter notebook `notebooks/calculate_memory.ipynb`.

### Meta-Learning Simulation

To draw Figure 1, use `notebooks/cifar10_pf_and_pf_pie_chart.ipynb`.

To draw the dataset Kendall-tau correlation plot in Figure 6, use `notebooks/dataset_correlation.ipynb`.

To draw the meta-test illustration in Figure 7, use `notebooks/peppp_meta_test_illustration.ipynb`.

To reproduce the meta-training and meta-LOOCV results, use `notebooks/pf_meta_training.ipynb` and `notebooks/pf_meta_loocv.ipynb`. To get different meta-LOOCV settings in Table 1, make the change in every function that performs one active learning approach, in `pf_meta_loocv.ipynb`.

To reproduce the experiments in Section 4.3 that tune the learning rate, use `notebooks/cifar100_LR_tuned_experiments.ipynb`.

## Acknowledgments

The authors thank <https://github.com/ryanchankh/mcr2> for some utility functions, thank <https://github.com/sksq96/pytorch-summary> for the code on memory calculation, and thank <https://github.com/georgehc/mnar_mc> for the implementation of weighted SoftImpute.
