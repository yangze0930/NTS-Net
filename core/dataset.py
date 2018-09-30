import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms

import sys
sys.path.append('.')

from config import INPUT_SIZE, MEAN, STD


class CUB():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            self.train_image_paths = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_image_paths = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = scipy.misc.imread(self.train_image_paths[index]), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = scipy.misc.imread(self.test_image_paths[index]), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class OwnDataset():
    '''Define your own data reading class here, 
    we promise to divide the data set into two parts, train, test two sub data sets
    
    the folder structure as follows:

    ./dataset
        /train
            X_train.txt
            y_train.txt
        /test
            X_test.txt
            y_test.txt
    '''
    def __init__(self, root_path=None, label_path=None, is_train=True, data_len=None):
        self.root = root_path
        self.label_path = label_path
        self.is_train = is_train

        X_train, y_train, X_test, y_test = self.read_txt()        
        if self.is_train:
            self.train_image_paths = [os.path.join(self.root, train_file) for train_file in
                              X_train[:data_len]]
            self.train_label = [int(x) for x in y_train][:data_len]
        if not self.is_train:
            self.test_image_paths = [os.path.join(self.root, test_file) for test_file in
                             X_test[:data_len]]
            self.test_label = [int(x) for x in y_test][:data_len]

    def read_txt(self):
        """read txt file containing label and IMAGE_PATH,and return list result
        """
        train_image_paths = [x.decode().replace('\r\n', '') for x in open(os.path.join(self.label_path, 'X_train.txt'), 'rb').readlines()]
        train_labels = [int(x) for x in open(os.path.join(self.label_path, 'y_train.txt'), 'rb').readlines()]
        
        test_image_paths = [x.decode().replace('\r\n', '') for x in open(os.path.join(self.label_path, 'X_test.txt'), 'rb').readlines()]
        test_labels = [int(x) for x in open(os.path.join(self.label_path, 'y_test.txt'), 'rb').readlines()]
        
        return train_image_paths, train_labels, test_image_paths, test_labels

    def __getitem__(self, index):
        if self.is_train:
            img, target = scipy.misc.imread(self.train_image_paths[index]), self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(MEAN, STD)(img)

        else:
            img, target = scipy.misc.imread(self.test_image_paths[index]), self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(MEAN, STD)(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

if __name__ == '__main__':
    # test on CUB dataset
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
    
    # test on your own dataset
    dataset = OwnDataset(root_path='path/to/dataset', label_path='data',is_train=True)
    print(len(dataset.train_image_paths))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = OwnDataset(root_path='path/to/dataset', label_path='data',is_train=False)
    print(len(dataset.test_image_paths))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
    
    
    
