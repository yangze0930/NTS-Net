from __future__ import print_function
import os
import sys
import time
import logging
import platform
import torch
from core.dataset import CUB, Car2000

import sys
sys.path.append('.')

from config import DATASET_PATH, BATCH_SIZE, dataloader_num_workers

__all__ = ['progress_bar', 'format_time', 'init_log', 'test_cuda', 'test_cudnn']

# for training on windows, there's no stty size 
# could raise ValueErrors for unpack not enough value
system_name = platform.system()
term_height, term_width = 0, 0
if system_name == '':
    raise SystemError('python cannot identify which system you use')
if system_name == 'Windows':
    # to get windows powershell window size
    # use (Get-Host).UI.RawUI.MaxWindowSize.Width 
    # or (Get-Host).UI.RawUI.MaxWindowSize.Height
    term_height, term_width = 73, 120 # these values belong to my powershell window size on my windows system
elif system_name == 'Linux':
    _, term_width = os.popen('stty size', 'r').read().split()

term_width = int(term_width)

TOTAL_BAR_LENGTH = 40.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(msg + ' | ')
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))


    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def test_cuda():
    import torch
    x = torch.Tensor([1.0])
    x_cuda = x.cuda()
    print(x_cuda)

def test_cudnn():
    import torch
    from torch.backends import cudnn
    x = torch.Tensor([1.0])
    x_cuda = x.cuda()
    print('result of cudnn test examples is:',cudnn.is_acceptable(x_cuda))

def compute_mean_variance():
    """Compute the mean and std value for a certain dataset,
    
    make sure parameters of Car2000 are same with train_loader in tran.py
    """
    print('Compute mean and variance for training data.')
    train_data = Car2000(
        root_path=DATASET_PATH, label_path='data', is_train=True, data_len=None)
    train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=dataloader_num_workers,pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print('mean:\n')
    print(mean)
    print('std:\n')
    print(std)

if __name__ == '__main__':
    # test_cudnn()
    compute_mean_variance()
