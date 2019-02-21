import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision
import math
import scipy.io as sio
import numpy as np

class train_net(nn.Module):
    def __init__(self):
        super(train_net, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(2, 2)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_3 = nn.ReLU()
        
        self.conv1_b1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu1_b1 = nn.ReLU()
        self.conv2_b1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu2_b1 = nn.ReLU()
        self.dropout_b1 = nn.Dropout(p=0.7)           # Not sure about this layer
        self.score_b1 = nn.Conv2d(512, 20, 1, 1, 0)         
        self.GAP_b1 = nn.AvgPool2d(14, 1)             # Global Average Pooling
        
        self.conv1_b2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu1_b2 = nn.ReLU()
        self.conv2_b2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu2_b2 = nn.ReLU()
        self.score_b2 = nn.Conv2d(512, 20, 1, 1, 0)
        self.GAP_b2 = nn.AvgPool2d(14, 1)             # Global Average Pooling
        
        self.conv1_b3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu1_b3 = nn.ReLU()
        self.conv2_b3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu2_b3 = nn.ReLU()
        self.score_b3 = nn.Conv2d(512, 20, 1, 1, 0)
        self.GAP_b3 = nn.AvgPool2d(14, 1)             # Global Average Pooling
        
    def forward(self, x, labels):
        
        x = self.maxpool1(self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x)))))
        x = self.maxpool2(self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x)))))
        x = self.maxpool3(self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(x)))))))
        x = self.maxpool4(self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(x)))))))
        features = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(x))))))
        
        score_b1 = self.score_b1(self.dropout_b1(self.relu2_b1(self.conv2_b1(self.relu1_b1(self.conv1_b1(features))))))
        prob1 = torch.sigmoid(self.GAP_b1(score_b1).squeeze())
        
        features_b2 = mask_b2(features, score_b1, labels)
        score_b2 = self.score_b2(self.relu2_b2(self.conv2_b2(self.relu1_b2(self.conv1_b2(features_b2)))))
        prob2 = torch.sigmoid(self.GAP_b2(score_b2).squeeze())
        
        features_b3 = mask_b3(features, score_b1, labels)
        score_b3 = self.score_b3(self.relu2_b3(self.conv2_b3(self.relu1_b3(self.conv1_b3(features_b3)))))
        prob3 = torch.sigmoid(self.GAP_b3(score_b3).squeeze())
        
        return prob1, prob2, prob3, score_b1, score_b2
    
def mask_b2(features, score, labels, maxt=0.8, mint=0.05):
    
    mask = score.clone()
    mask[mask < 0] = 0
    features = features.clone()
    
    for i in range(20): 
        if torch.all(labels[:, i] == 0):
            mask[:, i, :, :] = 0
        else:
            bs_label = torch.nonzero(labels[:, i]).reshape(-1)
            ma, _ = mask[bs_label, i, :, :].reshape(-1, 14*14).max(dim=1)
            ma = ma.reshape(-1, 1, 1)
            mi, _ = mask[bs_label, i, :, :].reshape(-1, 14*14).min(dim=1)
            mi = mi.reshape(-1, 1, 1)
            tmp = (mask[bs_label, i, :, :] - mi) / (ma - mi + 1e-8)
            mask[:, i, :, :] = 0
            mask[bs_label, i, :, :] = tmp
        
    mask, _ = mask.max(dim=1)
    pos = torch.nonzero(mask > maxt).transpose(1, 0)
    neg = torch.nonzero(mask < mint).transpose(1, 0)
    features[pos[0], i, pos[1], pos[2]] = 0
    features[neg[0], i, neg[1], neg[2]] = -1 * features[neg[0], i, neg[1], neg[2]]
    
    return features

def mask_b3(features, score, labels, thres=0.3):
        
    mask = score.clone()
    mask[mask < 0] = 0
    features = features.clone()
    
    for i in range(20):
        if torch.all(labels[:, i] == 0):
            mask[:, i, :, :] = 0
        else:
            bs_label = torch.nonzero(labels[:, i]).reshape(-1)
            ma, _ = mask[bs_label, i, :, :].reshape(-1, 14*14).max(dim=1)
            ma = ma.reshape(-1, 1, 1)
            mi, _ = mask[bs_label, i, :, :].reshape(-1, 14*14).min(dim=1)
            mi = mi.reshape(-1, 1, 1)
            tmp = (mask[bs_label, i, :, :] - mi) / (ma - mi + 1e-8)
            mask[:, i, :, :] = 0
            mask[bs_label, i, :, :] = tmp
        
    mask, _ = mask.max(dim=1)
    pos = torch.nonzero(mask > thres).transpose(1, 0)
    features[pos[0], i, pos[1], pos[2]] = 0
    
    return features
    
def load_vgg16pretrain(model, vggmodel='vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params =  model.state_dict()
    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size  == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()