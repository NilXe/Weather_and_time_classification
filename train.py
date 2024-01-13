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

path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset'

train_data = my_Datasets01(path)
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)   

# p_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\P_model.pth'
# w_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\W_model.pth'

# p_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\P_model_0.79469.pth'
# w_model_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\W_model_0.86101.pth'


# 定义模型
P_model = net.PeriodModel(resnet50, num_classes=4)
W_model = net.WeatherModel(resnet50, num_classes=3)
# 加载模型
# if p_model_path != '' and w_model_path != '':
#     P_model.load_state_dict(torch.load(p_model_path)['model'])
#     W_model.load_state_dict(torch.load(w_model_path)['model'])
#     epoch = torch.load(p_model_path)['epoch']
#     print(epoch)
#     print('model load')

# 模型微调
device = torch.device("cuda")

P_model.to(device)
W_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer_P = optim.Adam(P_model.parameters(), lr=0.001)
optimizer_W = optim.Adam(W_model.parameters(), lr=0.001)



epochs = 100
best_f1_score = 0.95


p_loss_lst = []
w_loss_lst = []
loss_lst = []

p_f1_score_lst = [] 
w_f1_score_lst = []
f1_score_lst = []

e_lst = []

for e in range(1, epochs+1):

    e_lst.append(e)

    P_model.train()
    W_model.train()

    print(f'------Epoch:{e}/{epochs}-------')

    epoch_loss = 0
    epoch_f1_score = 0
    epoch_loss_P = 0
    epoch_loss_W = 0
    epoch_f1_score_P = 0
    epoch_f1_score_W = 0

    for n, (img, P_label, W_label) in enumerate(train_dataloader):
        img = img.to(device)
        P_label = P_label.to(device)
        W_label = W_label.to(device)

        P_output = P_model(img)
        W_output = W_model(img)

        P_loss = criterion(P_output, P_label)
        W_loss = criterion(W_output, W_label)

        P_f_score = calculate_f1_score(P_label, P_output)
        W_f_score = calculate_f1_score(W_label, W_output)

        # 反向传播
        optimizer_P.zero_grad()
        optimizer_W.zero_grad()
        P_loss.backward()
        W_loss.backward()
        optimizer_P.step()
        optimizer_W.step()

        epoch_loss_P += P_loss
        epoch_loss_W += W_loss

        epoch_f1_score_P += P_f_score
        epoch_f1_score_W += W_f_score
        
    epoch_loss_P = round(float(epoch_loss_P/len(train_dataloader)), 5)
    epoch_loss_W = round(float(epoch_loss_W/len(train_dataloader)), 5)
    p_loss_lst.append(epoch_loss_P)
    w_loss_lst.append(epoch_loss_W)

    epoch_f1_score_P = round(float(epoch_f1_score_P/len(train_dataloader)), 5)
    epoch_f1_score_W = round(float(epoch_f1_score_W/len(train_dataloader)), 5)
    p_f1_score_lst.append(epoch_f1_score_P)
    w_f1_score_lst.append(epoch_f1_score_W)

    epoch_loss = round(float((epoch_loss_P+epoch_loss_W)/2), 5)
    epoch_f1_score = round(float((epoch_f1_score_P+epoch_f1_score_W)/2), 5)

    loss_lst.append(epoch_loss)
    f1_score_lst.append(epoch_f1_score)

    print(f'epoch_loss_P:{epoch_loss_P}, epoch_f1_score_P:{epoch_f1_score_P}')
    print(f'epoch_loss_W:{epoch_loss_W}, epoch_f1_score_W:{epoch_f1_score_W}')
    print(f'Epoch {e+1}/{epochs+1}, Loss: {epoch_loss}, F1 Score: {epoch_f1_score}')

    if epoch_f1_score > best_f1_score or e == epochs:
        torch.save({'model':P_model.state_dict(), 
                    'epoch': e}, rf'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\P_model_{epoch_f1_score_P}.pth')
        torch.save({'model': W_model.state_dict(),
                    'epoch':e}, rf'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\model\W_model_{epoch_f1_score_W}.pth')
        print('Model saved!')
    
plt.figure()
plt.plot(e_lst, loss_lst, label='Loss')

plt.plot(e_lst, f1_score_lst, label='F1 Score')

for i in range(len(e_lst)):
    plt.text(e_lst[i], f1_score_lst[i], f1_score_lst[i], ha='center', va='bottom', fontsize=5)

plt.plot(e_lst, p_loss_lst, label='P_Loss')
plt.plot(e_lst, w_loss_lst, label='W_Loss')
plt.plot(e_lst, p_f1_score_lst, label='P_F1 Score')
plt.plot(e_lst, w_f1_score_lst, label='W_F1 Score')
plt.title('Loss and F1 Score')
plt.ylabel('Loss and F1 Score')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\main\W_P.png')
plt.show()



    


