# @Time    : 2021/01/22 25:16
# @Author  : SY.M
# @FileName: run.py

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
import tqdm
import os
import numpy as np

from loss import Myloss
from models.transformer import Transformer
from utils.data_process import MyDataset

test_interval = 5  # 测试间隔 单位：epoch
draw_key = 1  # 大于等于draw_key才会保存图像

# 超参数设置
EPOCH = 20
BATCH_SIZE = 100
LR = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(f'use device: {DEVICE}')

models = 512
hiddens = 1024
q = 8
v = 8
h = 8
N = 8
dropout = 0.2
pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
# 优化器选择
optimizer_name = 'Adagrad'

train_dataset = MyDataset('train')
test_dataset = MyDataset('test')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

DATA_LEN = train_dataset.train_len  # 训练集样本数量
inputs = train_dataset.input_len  # 时间部数量
channels = train_dataset.channel_len  # 时间序列维度
outputs = train_dataset.output_len  # 分类类别
hz = train_dataset.hz  # hz

net = Transformer(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens,
                  q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)

# 创建Transformer模型

# 创建loss函数 此处使用 交叉熵损失
loss_function = Myloss()
if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
sensitivity_on_train = []
sensitivity_on_test = []
specificity_on_train = []
specificity_on_test = []
precision_on_train = []
precision_on_test = []
recall_on_train = []
recall_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0
net.train()
max_accuracy = 0
max_sensitivity = 0
max_specificity = 0
max_precision = 0
max_recall = 0





def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        TP=0
        TN=0
        FN=0
        FP=0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            c=(label_index == y.long())
            for i in range(label_index.shape[0]):
                if((c[i]==1).item()==1):
                    if(y.long()[i]==1):
                        TP+=1
                    elif(y.long()[i]==0):
                        TN+=1
                    else:
                        print("TPTNerror")
                elif((c[i]==1).item()==0):
                    if(y.long()[i]==1):
                        FN+=1
                    elif(y.long()[i]==0):
                        FP+=1
                    else:
                        print("FPFNerror")
                else:
                    print("error")
        print('TP,TN,FP,FN')
        print(TP,TN,FP,FN)
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
            sensitivity_on_test.append(round((100 * TP /(TP+FN)), 2))
            specificity_on_test.append(round((100 * TN/(TN+FP)), 2))
            precision_on_test.append(round((100 * TP/(TP+FP)), 2))
            recall_on_test.append(round((100 * TP /(TP+FN)), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
            sensitivity_on_train.append(round((100 * TP /(TP+FN)), 2))
            specificity_on_train.append(round((100 * TN/(TN+FP)), 2))
            precision_on_train.append(round((100 * TP/(TP+FP)), 2))
            recall_on_train.append(round((100 * TP /(TP+FN)), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))
        acc=round((100 * correct / total), 2)
        sensitivity=round((100 * TP /(TP+FN)), 2)
        specificity=round((100 * TN/(TN+FP)), 2)
        precision=round((100 * TP/(TP+FP)), 2)
        recall=round((100 * TP /(TP+FN)), 2)
        return acc,sensitivity,specificity,precision,recall




def train():
    for index in range(EPOCH):
        loss_temp = []
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))

            loss_temp.append(loss.item())
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        #                 torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        print(f'Epoch:{index + 1}:\t\tloss:{np.mean(loss_temp)}')
        if ((index + 1) % test_interval) == 0:
            current_accuracy, current_sensitivity, current_specificity, current_precision, current_recall = test(
                test_dataloader)
            test(train_dataloader, 'train_set')
            print('当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')

            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
            if current_sensitivity > max_sensitivity:
                max_sensitivity = current_sensitivity
            if current_specificity > max_specificity:
                max_specificity = current_specificity
            if current_precision > max_precision:
                max_precision = current_precision
            if current_recall > max_recall:
                max_recall = current_recall


if __name__ == '__main__':
    train()