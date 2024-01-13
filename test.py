import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import net
from my_dataset import *
import torchvision.models as models 

resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load(r"C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\resnet50-11ad3fa6.pth"))

# 计算F-score
def calculate_f1_score(y_true, y_pred):
    """
    计算多分类任务的 F1-score
    :param y_true: 实际标签
    :param y_pred: 预测标签，可以是概率值
    :return: F1-score
    """
    
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    f1 = f1_score(y_true, y_pred, average='weighted')

    return float(f1)

test_data_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\test_dataset'

train_data = my_test_Dataset(test_data_path)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)   
#
# p_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\P_model_0.79469.pth'
# w_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\W_model_0.86101.pth'

p_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\P_model_0.94299.pth'
w_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\W_model_0.98199.pth'


# 定义模型
P_model = net.PeriodModel(resnet50, num_classes=4)
W_model = net.WeatherModel(resnet50, num_classes=3)
# 加载模型
if p_model_path != '' and w_model_path != '':
    P_model.load_state_dict(torch.load(p_model_path)['model'])
    W_model.load_state_dict(torch.load(w_model_path)['model'])
    epoch = torch.load(p_model_path)['epoch']
    print(epoch)
    print('model load')

# 模型微调
device = torch.device("cuda")

P_model.to(device)
W_model.to(device)

p_label_dic = {0: 'Morning', 1: 'Afternoon', 2: 'Dawn', 3: 'Dusk'}
w_label_dic = {0: 'Cloudy', 1: 'Sunny', 2: 'Rainy'}


data_dic = {}
lst = []
for n, (name, img) in enumerate(train_dataloader):

    img = img.to(device)
    print(name)
    P_output = P_model(img)
    W_output = W_model(img)

    P_pred = p_label_dic.get(torch.argmax(P_output, dim=1).cpu().numpy()[0])
    W_pred = w_label_dic.get(torch.argmax(W_output, dim=1).cpu().numpy()[0])
    

    lst.append({'filename': name[0], 'period': P_pred, 'weather': W_pred})

data_dic['annotations'] = lst


# 将字典转换为JSON字符串  
json_data = json.dumps(data_dic)  

# 将JSON字符串写入文件  
with open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\test.json', 'w') as f:  
    f.write(json_data)

print('over')

# {
#     “annotations”: 
#      [
#         {
#             “filename”: “test_images\00008.jpg”,
#             “period”: “Morning”,
#             “weather”: “Cloudy”
#         }
#      ]
# }
