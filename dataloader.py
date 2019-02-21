import torch.utils.data as data
import numpy as np
import os
from os.path import join
from PIL import Image
import torch
from random import shuffle
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class JPEGLoader(data.Dataset):
    
    def __init__(self, txt_path, img_dir, transform=None):
        f = open(txt_path, 'r')
        self.info = f.readlines()
        f.close()
        #shuffle(self.info)
        
        self.img_dir = img_dir
        
        self.transform = transform
        
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        data = self.info[idx].split()
        
        img = Image.open(join(self.img_dir, data[0]+".jpg"))
        if self.transform:
            img = self.transform(img)
            
            img = np.asarray(img).astype(np.float32) - np.array([116.62341813, 111.51273588, 103.14803339])
            img = torch.Tensor(np.transpose(img, [2, 0, 1]))
        
        labels = torch.zeros(20, dtype=torch.int64)
        for idx in data[1:]:
            labels[int(idx)] = 1
    
        return img, labels