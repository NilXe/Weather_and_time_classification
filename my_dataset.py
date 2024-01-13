import json
import os

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

train_transform02 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

class my_Datasets01(Dataset):  # 需要继承data.Dataset
    def __init__(self, path, transform=train_transform02):

        self.path = path
        with open(path+'\\data.json', 'r') as f:  
            data = json.load(f)  
        self.name1 = data['annotations']
        self.trans = transform

    def __getitem__(self, index):
        # 拿到的图片和标签
        data = self.name1[index]
        
        # 图片和标签的路径
        img_path = data['path']
        p_label = data['p_label']
        w_label = data['w_label']
        # 读取图像，并将标签升维
        img_o = Image.open(os.path.join(self.path, img_path))
        
        img_o = self.trans(img_o)

        p_label = torch.tensor(p_label)
        w_label = torch.tensor(w_label)

        return img_o, p_label, w_label

    def __len__(self):
        return len(self.name1)

class my_Datasets02(Dataset):  # 需要继承data.Dataset
    def __init__(self, path, transform=train_transform02):

        self.path = path
        with open(path, 'r') as f:  
            data = json.load(f)  
        self.name1 = data['annotations']
        self.trans = transform

    def __getitem__(self, index):
        # 拿到的图片和标签
        data = self.name1[index]
        
        # 图片和标签的路径
        img_path = data['path']
        label = data['label']
        # 读取图像，并将标签升维
        img_o = Image.open(os.path.join(img_path))
        
        img_o = self.trans(img_o)
        label = torch.tensor(label)

        return img_o, label

    def __len__(self):
        return len(self.name1)


# 测试数据Dataset
class my_test_Dataset(Dataset):
    def __init__(self, path, transform=train_transform02):

        self.path = path
        self.name = os.listdir(os.path.join(self.path, 'test_images'))
        self.trans = transform

    def __getitem__(self, index):
        # 拿到的图片
        data = self.name[index]
        # 图片和标签的路径
        img_path = os.path.join(self.path+'\\test_images', data)
        name = 'test_images\\' + data
        # 读取图像，并将标签升维
        img_o = Image.open(os.path.join(img_path))
        
        img_o = self.trans(img_o)

        return name, img_o

    def __len__(self):
        return len(self.name)
    


if __name__ == '__main__':

    path01 = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data.json'
    path02 = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data.json'

    my_dataloader = my_Datasets02(path=path02)
    my_data = DataLoader(my_dataloader, batch_size=64, shuffle=True)
    for n, (img, label) in enumerate(my_data):
        if n == 0:
            print(label)
            break


    # test_data_pth = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\test_dataset'
    # my_test_Dataset = my_test_Dataset(test_data_pth)
    # my_test_data = DataLoader(my_test_Dataset, batch_size=4, shuffle=True)
    # for i, (name, img) in enumerate(my_test_data):
    #     if i == 0:
    #         print(name)
    #         print(img.shape)
    #         break



    # my_Datasets = my_Datasets01(data_path)

    # my_data = DataLoader(my_Datasets, batch_size=4, shuffle=True)
    # for i, (input, p_label, w_label) in enumerate(my_data):

    #     if i == 1:
    #         print('input:', input.shape)
    #         print('p_label:', p_label)
    #         print('w_label:', w_label)
    #         break

    # print('over')
