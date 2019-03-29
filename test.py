import os
from torch.autograd import Variable
import torch.utils.data
from torch.nn import DataParallel
from config import BATCH_SIZE, PROPOSAL_NUM, test_model
from core import model, dataset
from core.utils import progress_bar

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if not test_model:
    raise NameError('please set the test_model file to choose the checkpoint!')
# read dataset
trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)
testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8, drop_last=False)
# define model
net = model.attention_net(topN=PROPOSAL_NUM)
ckpt = torch.load(test_model)
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()
net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

# evaluate on train set
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
        progress_bar(i, len(trainloader), 'eval on train set')

train_acc = float(train_correct) / total
train_loss = train_loss / total
print('train set loss: {:.3f} and train set acc: {:.3f} total sample: {}'.format(train_loss, train_acc, total))


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
        progress_bar(i, len(testloader), 'eval on test set')

test_acc = float(test_correct) / total
test_loss = test_loss / total
print('test set loss: {:.3f} and test set acc: {:.3f} total sample: {}'.format(test_loss, test_acc, total))

print('finishing testing')
