import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2

caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#remove the following two lines if testing with cpu
#caffe.set_mode_gpu()
# choose which GPU you want to use
#caffe.set_device(0)
caffe.SGDSolver.display = 0

# load net
deploy_file = 'deploy_seenet.prototxt'
model_file = 'seenet_final.caffemodel'
net = caffe.Net(deploy_file, model_file, caffe.TEST)

# images for testing
im_lst = [('samples/2007_000039.jpg', [19,]),
          ('samples/2007_000063.jpg', [8, 11]),
          ('samples/2007_000738.jpg', [0,]),
          ('samples/2007_001185.jpg', [4, 7, 10, 14])]

cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormaps = ['#000000', '#7F0000', '#007F00', '#7F7F00', '#00007F', '#7F007F', '#007F7F', '#7F7F7F', '#3F0000', '#BF0000', '#3F7F00',
             '#BF7F00', '#3F007F', '#BF007F', '#3F7F7F', '#BF7F7F', '#003F00', '#7F3F00', '#00BF00', '#7FBF00', '#003F7F']

def colormap(index):
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', [colormaps[0], colormaps[index+1], '#FFFFFF'], 256)

def resize(im, size):
    h, w = im.shape[:2]
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
    im -= np.array((104.007, 116.669, 122.679))
    im = im.transpose((2, 0, 1))
    return im, h, w

def forward(net, im, label):
    im, height, width = resize(im, test_size)
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    net.blobs['label'].reshape(1, 1, 1, 20)
    net.blobs['label'].data[0,0,0,label] = 1
    net.forward()
    att1 = net.blobs['score_b1'].data[0][label]
    att2 = net.blobs['score_b2'].data[0][label]
    att1 = cv2.resize(att1, (width, height), interpolation=cv2.INTER_CUBIC)
    att2 = cv2.resize(att2, (width, height), interpolation=cv2.INTER_CUBIC)
    att1[att1 < 0] = 0
    att2[att2 < 0] = 0
    att1 = att1 / (np.max(att1) + 1e-8)
    att2 = att2 / (np.max(att2) + 1e-8)
    att = np.maximum(att1, att2)
    return att

#Visualization
def plot_atts(att_lst, label_lst, size):
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
    plt.savefig('img%d'%img_id)
    plt.close()

# input image
test_size = 256
with_flip = True

img_id = 3 # 0-3
im_name, im_labels = im_lst[img_id]
img = cv2.imread(im_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # for visualizing
att_maps = []
img = img.astype('float')
for label in im_labels:
    att = forward(net, img, label)
    if with_flip:
        img_flip = img[:,::-1,:]
        att_flip = forward(net, img_flip, label)
        att = np.maximum(att, att_flip[:,::-1])
    att = att * 0.8 + img_gray / 255. * 0.2
    att_maps.append(att)

res_lst = [img[:,:,::-1].astype(np.uint8),] + att_maps
label_lst = ['Source',] + im_labels
plot_atts(res_lst, label_lst, 20)