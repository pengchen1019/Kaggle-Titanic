import torch
import torch.nn as nn
import pandas as pd
import numpy as np
device = torch.device("cuda")

class DataProcessing(object):
    def __init__(self):
        pass

    def get_data(self):
        data_train = pd.read_csv('train.csv')
        label = data_train[['Survived']]
        data_test = pd.read_csv('test.csv')
        # 读取指定列
        gender = pd.read_csv('gender_submission.csv', usecols=[1])
        return data_train, label, data_test, gender

    def data_processing(self, data_):
        # 训练集测试集都进行相同的处理
        data = data_[['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Cabin', 'Embarked']]
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        data['Cabin'] = pd.factorize(data.Cabin)[0]
        data.fillna(0, inplace=True)
        data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]
        data['p1'] = np.array(data['Pclass'] == 1).astype(np.int32)
        data['p2'] = np.array(data['Pclass'] == 2).astype(np.int32)
        data['p3'] = np.array(data['Pclass'] == 3).astype(np.int32)
        data['e1'] = np.array(data['Embarked'] == 'S').astype(np.int32)
        data['e2'] = np.array(data['Embarked'] == 'C').astype(np.int32)
        data['e3'] = np.array(data['Embarked'] == 'Q').astype(np.int32)
        del data['Pclass']
        del data['Embarked']
        return data

    def data(self):
        # 读数据
        train_data, label, test_data, gender = self.get_data()
        # 处理数据
        # 训练集输入数据
        train = np.array(data_processing.data_processing(train_data))
        # 训练集标签
        train_label = np.array(label)
        # 测试集
        test = np.array(data_processing.data_processing(test_data))
        # 测试集标签
        test_label = np.array(gender)

        train = torch.from_numpy(train).float()
        train_label = torch.tensor(train_label).float()
        test = torch.tensor(test).float()
        test_label = torch.tensor(test_label)

        return train, train_label, test, test_label


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 7),
            nn.Sigmoid(),
            nn.Linear(7, 7),
            nn.Sigmoid(),
            nn.Linear(7, 1),
        )
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)
        self.mls = nn.MSELoss()

    def forward(self, inputs):
        # 前向传播
        return self.fc(inputs)

    def train(self, inputs, y):
        # 训练
        out = self.forward(inputs)
        loss = self.mls(out, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # print(loss)

    def test(self, x, y):
        # 测试
        # 将variable张量转为numpy
        # out = self.fc(x).data.numpy()
        count = 0
        out = self.fc(x)
        sum = len(y)
        for i, j in zip(out, y):
            i = i.detach().numpy()
            j = j.detach().numpy()
            loss = abs((i - j)[0])
            if loss < 0.3:
                count += 1
        # 误差0.3内的正确率
        print(count/sum)


if __name__ == '__main__':
    data_processing = DataProcessing()
    train_data, train_label, test_data, test_label = data_processing.data()
    net = MyNet()
    count = 0
    for i in range(20000):
        # 为了减小电脑压力,分批训练 100个训练一次  ## 2018.12.22补充：正确的做法应该是用batch
        for n in range(len(train_data)//100 + 1):
            batch_data = train_data[n*100: n*100 + 100]
            batch_label = train_label[n*100: n*100 + 100]
            net.train(train_data, train_label)
    net.test(test_data, test_label)  # 输出结果：0.7488038277511961