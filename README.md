# DFN-tensorflow
This repo is the tensorflow implementation of Discriminative Feature Network (DFN)

The original paper for DFN can be found at https://arxiv.org/abs/1804.09337.

## Condensed Abstract

Most existing methods of semantic segmentation still suffer from two aspects of challenges: intra-class inconsistency and inter-class indistinction. To tackle these two problems, we propose a Discriminative Feature Network (DFN), which contains two sub-networks: Smooth Network and Border Network. Specifically, to handle the intra-class inconsistency problem, we specially design a Smooth Network with Channel Attention Block and global average pooling to select the more discriminative features. Furthermore, we propose a Border Network to make the bilateral features of boundary distinguishable with deep semantic boundary supervision. 

## Getting Started Guide

### Install Required Packages
This repo of code is written using Tensorflow. Please install all required packages before using this code.
```bash
pip install -r requirements.txt
```

### Dataset Collecttion
The project can only solve problems of binary segmentation at present.So you can only collect a dataset with labels of binary segmentation and split into 3 directories labeled 'train', 'val' and 'test', each of which has 2 subfolders named 'main' and 'segmentation', while 'main' should store original images and 'segemntation' should store segmentation images.

### Data Augmentation
```bash
python data_augment.py --dir data/train
```

### Training
```bash
python main.py --batch_size 1
```

### Testing
```bash
python main.py --batch_size 1 --is_training False
```
The results would be saved in the folder test-outputs.

### Evaluation
```bash
python evaluation.py --gt_dir data/test/segmentation --pred_dir test-outputs --result_txt results.txt
```
The IOU results would be written in the results.txt.

### Questions or Comments

Please direct any questions or comments to me; I am happy to help in any way I can. You can email me directly at 20164228014@stu.suda.edu.cn.
