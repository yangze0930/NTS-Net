import os
from datetime import datetime

import torch.utils.data
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR

from config import (BATCH_SIZE, DATASET_PATH, LR, PROPOSAL_NUM, SAVE_FREQ, WD,
                    dataloader_num_workers, own_dataset, resume, save_dir)
from core import dataset, model
from core.utils import init_log, progress_bar


def read_own_dataloader():
    trainset = dataset.OwnDataset(root_path=DATASET_PATH, label_path='data', is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=dataloader_num_workers, drop_last=False)
    testset = dataset.OwnDataset(root_path=DATASET_PATH, label_path= 'data', is_train=False, data_len=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=dataloader_num_workers, drop_last=False)
    return trainloader, testloader

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_epoch = 1
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info
    print('Finished makdir save_model and create log instance')
    print('Read dataset')
    
    # read dataset    
    if own_dataset:
        trainloader, testloader = read_own_dataloader()
    else:
        trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=8, drop_last=False)
        testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=8, drop_last=False)
    # define model
    print('define model')
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
    print('start training')
    for epoch in range(start_epoch, 500):
        for scheduler in schedulers:
            scheduler.step()
        ##########################  train the model  ###############################
        _print('--' * 50)
        net.train()
        for i, data in enumerate(trainloader):
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()

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

        ##########################  evaluate net and save model  ###############################
        if epoch % SAVE_FREQ == 0:
            ##########################  evaluate net on test set  ###############################
            train_loss = 0
            train_correct = 0
            total = 0
            net.eval()
            for i, data in enumerate(trainloader):
                with torch.no_grad():
                    img, label = data[0].cuda(), data[1].cuda()
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

            ##########################  evaluate net on test set  ###############################
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

            ##########################  save model  ###############################
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
            print('model has been saved')

    print('finishing training')

    # release all allocated GPU memory
    torch.cuda.empty_cache()
