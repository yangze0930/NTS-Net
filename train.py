#from datasets.ShuffleMNIST.dataset import ShuffleMNIST
import os
import numpy as np
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir
from core import model, dataset
from core.utils import init_log, progress_bar, create_dir

import torchvision
from torchvision.utils import save_image
from torchvision import transforms

from ShuffleMNIST import dataset as Shuffdata


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')

#se agrego la funcion create_dir en utils
save_dir = create_dir('save_dir')
logging = init_log(save_dir)
_print = logging.info

# read dataset
#trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
#                                          shuffle=True, num_workers=8, drop_last=False)
#testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
#testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
#                                         shuffle=False, num_workers=8, drop_last=False)

#read dataset
batch_size_train = 64
batch_size_test = 1000


dataset_train =  torchvision.datasets.MNIST('/home/alessio/alonso/datasets', train=True, download=True,
                             transform=torchvision.transforms.ToTensor())

dataset_test =  torchvision.datasets.MNIST('/home/alessio/alonso/datasets', train=False, 
                                           download=True,transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train,drop_last=True, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size_test, shuffle = True,drop_last=True)

shuffled_train = Shuffdata.ShuffleMNIST(train_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True,is_train=True)
shuffled_test = Shuffdata.ShuffleMNIST(test_loader, anchors = [], num=4, radius = 42, wall_shape = 112, sum = True, is_train = False)

print('There are {} images and {} labels in the train set.'.format(len(shuffled_train.train_img),
        len(shuffled_train.train_label)))
print('There are {} images and {} labels in the test set.'.format(len(shuffled_test.test_img),
        len(shuffled_test.test_label)))

#Configuring shuffled DataLoader
from torch.utils.data.sampler import RandomSampler

#se cambian estos nombres a train loader para que sean los que se llaman en la red
train_sampler = RandomSampler(shuffled_train, replacement=True, num_samples= 51200, generator=None)
test_sampler = RandomSampler(shuffled_test, replacement=True, num_samples= 5760, generator=None)

trainloader = torch.utils.data.DataLoader(shuffled_train, batch_size=batch_size_train
                                                   ,drop_last=False, sampler = train_sampler)

testloader = torch.utils.data.DataLoader(shuffled_test, batch_size=batch_size_train
                                                  ,drop_last=False, sampler = test_sampler)

# define model
net = model.attention_net(topN=PROPOSAL_NUM)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
raw_parameters = list(net.pretrained_model.parameters())
part_parameters = list(net.proposal_net.parameters())
concat_parameters = list(net.concat_net.parameters())
partcls_parameters = list(net.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]
net = net.cuda()
net = DataParallel(net)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Eevitar warning y que si se concidere el primer valor de la taza de aprendizaje
# lr_scheduler.MultiStepLR()

for epoch in range(start_epoch, 500):
    for scheduler in schedulers:
        scheduler.step()

    # begin training
    _print('--' * 50)
    net.train()
    for batch_idx, (data_, target_) in enumerate(trainloader):
        #cambiamos sintaxis de esta parte
        img, label = data_, target_.to(device)
        batch_size = img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()


        #transformacion de los datos
        if len(img.shape) == 3:
                img = np.stack([img] * 3, 2)        
        img = np.transpose(img, (0,1,2,3))
        img = torch.as_tensor(img)
        data_ = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = data_.to(device)

        raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        progress_bar(i, len(trainloader), 'train')

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for batch_idx_t, (data_t, target_t) in enumerate(trainloader):
            with torch.no_grad():
                img, label = data_t, target_t.to(device)

                #transformacion de los datos
                img = np.array(data_t)
                #print(img.shape)
                if len(img.shape) == 3:
                    img = np.stack([img] * 3, 2)

                img = np.transpose(img, (0,2,1,3))
                img = torch.as_tensor(img)
                data_t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                img = data_t.to(device)

                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size
                progress_bar(i, len(trainloader), 'eval train set')

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        _print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                _, concat_logits, _, _, _ = net(img)
                # calculate loss
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size
                progress_bar(i, len(testloader), 'eval test set')

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        _print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')
