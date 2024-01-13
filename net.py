import torch

# from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision.models as models
import torchvision.models as models  
resnet50 = models.resnet50(pretrained=True)
from my_dataset import *
from torch.utils.data import DataLoader
import torchvision.models as models  


resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\resnet50-11ad3fa6.pth"))

# 引入rest50模型
# net = models.resnet50()
# net.load_state_dict(torch.load("./model/resnet50-11ad3fa6.pth"))


class WeatherModel(nn.Module):
    def __init__(self, net, num_classes):
        super(WeatherModel, self).__init__()
        self.num_classes = num_classes
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, self.num_classes)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x

class PeriodModel(nn.Module):
    def __init__(self, net, num_classes):
        super(PeriodModel, self).__init__()
        self.num_classes = num_classes
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, self.num_classes)
        self.output = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x

class net01(nn.Module):
    def __init__(self, net, num_classes):
        super(net01, self).__init__()
        self.num_classes = num_classes
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, self.num_classes)
        self.output = nn.Softmax(dim=1)
        

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x
    

# if __name__ == '__main__':

    # a = torch.ones((1, 3, 224, 224))
    # net = WeatherModel(resnet50, num_classes=4)
    # b = net(a)

    # # writer = SummaryWriter(r"C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\logs")
    # writer.add_graph(net, a)

    # print(b.shape)
    # writer.close()

    # # tensorboard --logdir=logs
    # print('over')


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset'

    # train_data = my_Datasets01(path)
    # train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    #
    # w_model = WeatherModel(resnet50, num_classes=3).to(device)
    # p_model = PeriodModel(resnet50, num_classes=4).to(device)
    #
    # for n, (img, P_label, W_label) in enumerate(train_dataloader):
    #
    #     if n == 0:
    #         img = img.to(device)
    #
    #         P_label = P_label.to(device)
    #         W_label = W_label.to(device)
    #
    #         w_output = w_model(img)
    #         P_output = p_model(img)
    #         print(img.shape)
    #
    #         print('w_output:', w_output.shape, w_output, W_label.shape, W_label)
    #         print('p_output', P_output.shape, P_output, P_label.shape, P_label)
    #
    #         break

    

    