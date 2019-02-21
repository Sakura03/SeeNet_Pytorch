from PIL import Image
import numpy as np

means = np.zeros(3, dtype=np.float64)

with open('./data/train_cls.txt', 'r') as f:
    lines = f.readlines()
num = 0
for i in range(len(lines)):
    name = lines[i].split()[0]
    try:
        img = Image.open('data/Pascal VOC dataset/VOCdevkit/VOC2012/JPEGImages/'+name+".jpg")
        img = np.array(img)
        img_mean = img.mean(axis=(0, 1))
        means += img_mean
        print(img_mean)
    except:
        print(name+' is massing')
        num+=1

means /= (len(lines)-num)
print(num)
print(means)



























