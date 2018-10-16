# NTS-Net

This is a PyTorch implementation of the ECCV2018 paper "Learning to Navigate for Fine-grained Classification" (Ze Yang, Tiange Luo, Dong Wang, Zhiqiang Hu, Jun Gao, Liwei Wang).you can find original repository from github.com/yangze0930/NTS-Net

## Requirements
- python 3+
- pytorch 0.4+
- numpy
- datetime

## Datasets
Download the [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) datasets and put it in the root directory named **CUB_200_2011**, You can also try other fine-grained datasets.
## your own dataset
I recoomand taht your dataset should be like  as below:
```
dataset
|---images
|-------image01
|-------image02
|-------....
|---X_train.txt
|---X_test.txt
|---y_train.txt
|---y_test.txt
```
## Train the model
If you want to train the NTS-Net, just run ``python train.py``. You may need to change the configurations in ``config.py``. The parameter ``PROPOSAL_NUM`` is ``M`` in the original paper and the parameter ``CAT_NUM`` is ``K`` in the original paper. During training, the log file and checkpoint file will be saved in ``save_dir`` directory. You can change the parameter ``resume`` to choose the checkpoint model to resume.

## Test the model
The test.py doesn't work on your own dataset, I will make it soon.
And also,I am glad to see that anybody could contribute to this repository.
If you want to test the NTS-Net on ``CUB`` dataset, just run ``python test.py``. You need to specify the ``test_model`` in ``config.py`` to choose the checkpoint model for testing.

## Train your own model

If you want to train NTS-Net on your own dataset, you need to prepare the following:
1. Divide your data set into two parts, one is the training set and the other is the test set.
2. There are four files in the data folder: X_train.txt, y_train.txt, X_test.txt, y_test.txt

X_train.txt shows as follows
```
/data/path/to/your/own/data/image01
/data/path/to/your/own/data/image02
/data/path/to/your/own/data/image03
/data/path/to/your/own/data/image04
```

Y_rain.txt shows as follows
```
1
1
2
2
```

Your label_map file should be placed in the data folder,
```
label01,1
label02,2
label03,3
label04,4
```

3. Modify ``FC_NUMS`` in config.py to the total number of classes in your dataset, and ``DATASET_PATH`` as the path to your dataset.

4. Before training, please calculate the mean and variance on the dataset. just run
```
Python core/utils.py
```

Then modify the ``MEAN`` and ``STD`` in config.py

5. Confirm your model save location and modify the ``save_dir`` field in config.py to adjust the value of the ``batch_size`` and ``dataloader_num_workers`` parameters for your machine. You can modify it in config.py.

6. modify the ``own_dataset`` as ``True`` in config.py, run ``python train.py`` to start training your own dataset.

note:

If your machine is multi-GPU, you can modify it in train.py
``Os.environ['CUDA_VISIBLE_DEVICES'] = '0'``, refer to the official pytorch documentation

## Model
the official repository provides a checkpoint model trained by author ``yangze0930``,please check [here](https://drive.google.com/file/d/1F-eKqPRjlya5GH2HwTlLKNSPEUaxCu9H/view?usp=sharing) to get model.If you test on provided model, you will get a 87.6% test accuracy as paper shows.

## TODO
- [ ] add densenet as feature extractor, it will be soon
- [ ] edit test.py to test on own trained model and dataset, it will be soon
- [ ] add squeezenet and mobilenet as feature extractor once I have enough time

## Reference
If you are interested in our work and want to cite it, please acknowledge the following paper:

```
@inproceedings{Yang2018Learning,
author = {Yang, Ze and Luo, Tiange and Wang, Dong and Hu, Zhiqiang and Gao, Jun and Wang, Liwei},
title = {Learning to Navigate for Fine-grained Classification},
booktitle = {ECCV},
year = {2018}
}
```
