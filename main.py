import argparse
import time
import cv2
import os
from os.path import isdir, join
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils import data
from torchvision import transforms
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
from vltools import Logger
from dataloader import JPEGLoader
from model import train_net, load_vgg16pretrain, weights_init

parser = argparse.ArgumentParser(description='PyTorch Implementation of SeeNet.')
parser.add_argument('--bs', type=int, help='batch size', default=16)
# optimizer parameters
parser.add_argument('--lr', type=float, help='base learning rate', default=5e-3)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--stepsize', type=float, help='step size (epoch)', default=8)
parser.add_argument('--gamma', type=float, help='gamma', default=0.1)
parser.add_argument('--wd', type=float, help='weight decay', default=2e-4)
parser.add_argument('--maxepoch', type=int, help='max epoch', default=40)
# general parameters
parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
parser.add_argument('--save_freq', type=int, help='save frequency', default=100)
parser.add_argument('--cuda', type=str, help='cuda', default='3')
parser.add_argument('--checkpoint', type=str, help='checkpoint prefix', default=None)
parser.add_argument('--imgsize', type=int, help='image size fed into network', default=224)
# datasets
parser.add_argument('--tmp', type=str, default='tmp', help='root of saving images')
args = parser.parse_args()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
 
 
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, bs, maxlen=100):
        self.reset()
        self.maxlen = maxlen
        self.bs = bs

    def reset(self):
        self.memory = []
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val):
        if self.count >= self.maxlen:
            self.memory.pop(0)
            self.count -= 1
        self.memory.append(val)
        self.val = val
        self.sum = sum(self.memory)
        self.count += 1
        self.avg = self.sum / self.count

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
             '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

TMP_DIR = "/media/data1/SeeNet_result/"+args.tmp
if not isdir(TMP_DIR):
    os.mkdir(TMP_DIR)
if not isdir(join(TMP_DIR, 'checkpoint')):
    os.mkdir(join(TMP_DIR, 'checkpoint'))
log = Logger(TMP_DIR+'/log.txt')

transform = transforms.Compose([transforms.Resize(256), 
                                transforms.RandomCrop(args.imgsize)])

training_dataset = JPEGLoader('data/train_cls.txt', 'data/VOCdataset/VOCdevkit/VOC2012/JPEGImages', transform=transform)
training_dataloader = data.DataLoader(training_dataset, batch_size=args.bs, 
                                      shuffle=True, num_workers=8, pin_memory=True)

model = train_net()
model.apply(weights_init)
load_vgg16pretrain(model)
model = DataParallel(model).cuda()

weight = []
bias = []

for name, p in model.named_parameters():
    if 'weight' in name:
        weight.append(p)
    else:
        bias.append(p)

optimizer = optim.SGD([{"params": weight, "lr": args.lr, "weight_decay": 0},
                       {"params": bias, "lr": 2*args.lr, "weight_decay": args.wd}], momentum=args.momentum) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
criterion = nn.BCELoss()

losses = AverageMeter(args.bs)
times = AverageMeter(1)
record = {'avg': [], 'val': []}

def plot_atts(att_lst, label_lst, size, savedir):
    pylab.rcParams['figure.figsize'] = size, size/2
    plt.figure()
    for i in range(0, len(att_lst)):
        s = plt.subplot(1,len(att_lst),i+1)
        
        if label_lst[i] == 'Source':
            s.set_xlabel(label_lst[i], fontsize=18)
            plt.imshow(att_lst[i])
        else:
            s.set_xlabel(cats[int(label_lst[i])], fontsize=18)
            plt.imshow(att_lst[i], cmap = colormap(int(label_lst[i])))
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.savefig(savedir)
    plt.close()

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)

def train_epoch(epoch):
    if not isdir(join(TMP_DIR, 'epoch%d'%epoch)):
        os.mkdir(join(TMP_DIR, 'epoch%d'%epoch))
    model.train()
    
    for batch_idx, (img, labels) in enumerate(training_dataloader):
        start_time = time.time()
        bs = img.size(0)
        img, labels = img.cuda(), labels.cuda()
        prob1, prob2, prob3, score_b1, score_b2 = model(img, labels)
        background = torch.zeros(bs, 20, dtype=torch.float).cuda()
        
        #print(prob1[0, :]);print(prob2[0, :]);print(prob3[0, :])
        
        loss1 = criterion(prob1, labels.float())
        loss2 = criterion(prob2, labels.float())
        loss3 = criterion(prob3, background)
        
        loss = loss1 + loss2 + loss3
        
        losses.update(loss)
        record['avg'].append(losses.avg)
        record['val'].append(losses.val)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        times.update(time.time() - start_time)
        
        if batch_idx % args.print_freq == 0:
            log.info("Tr|Ep %03d Bt %03d/%03d: sec/bt: %.2fsec, loss=%.3f (avg=%.3f)" \
                     % (epoch, batch_idx, len(training_dataloader), times.val, losses.val, losses.avg))
        
        if batch_idx % args.save_freq == 0:
            rn = np.random.choice(bs, 1, replace=False)
            for i in rn:
                i_image = img[i, :, :, :].cpu().detach().numpy()                    
                i_image = (np.transpose(i_image, [1, 2, 0]) + np.array([116.62341813, 111.51273588, 103.14803339])).astype('uint8')
                image_gray = cv2.cvtColor(i_image, cv2.COLOR_BGR2GRAY)
                i_label = torch.nonzero(labels[i, :]).reshape(-1)
                att_maps = []
                
                for j in i_label:
                    att1, att2 = score_b1[i, j, :, :].cpu().detach().numpy(), score_b2[i, j, :, :].cpu().detach().numpy()
                    att1 = cv2.resize(att1, (args.imgsize, args.imgsize), interpolation=cv2.INTER_CUBIC)
                    att2 = cv2.resize(att2, (args.imgsize, args.imgsize), interpolation=cv2.INTER_CUBIC)
                    att1[att1 < 0] = 0
                    att2[att2 < 0] = 0
                    att1 = att1 / (np.max(att1) + 1e-8)
                    att2 = att2 / (np.max(att2) + 1e-8)
                    att = np.maximum(att1, att2)
                    #att = att * 0.8 + image_gray / 255. * 0.2
                    att_maps.append(att)
                    
                res_lst = [i_image[:,:,::-1],] + att_maps
                label_lst = ['Source',] + i_label.cpu().numpy().tolist()
                plot_atts(res_lst, label_lst, 16, join(TMP_DIR, 'epoch%d'%epoch, 'iter%d.jpg'%batch_idx))    
        
    torch.save(model.state_dict(), TMP_DIR+'/checkpoint/epoch%d'%epoch)
    log.info('checkpoint has been created!')


def main():
    for epoch in range(args.maxepoch):
        scheduler.step()           # will adjust learning rate
        train_epoch(epoch)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].plot(record['avg'])
        axes[0].legend(['Loss_avg'], loc="upper right")
        axes[0].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[0].set_xlabel("Iter")
        axes[0].set_ylabel("Loss_avg")

        axes[1].plot(record['val'])
        axes[1].grid(alpha=0.5, linestyle='dotted', linewidth=2, color='black')
        axes[1].legend(["Loss_val"], loc="upper right")
        axes[1].set_xlabel("Iter")
        axes[1].set_ylabel("Loss_val")

        plt.tight_layout()
        plt.savefig(TMP_DIR+'/record.pdf')
        plt.close(fig)


if __name__ == '__main__':
    main()    
