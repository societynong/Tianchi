import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd.variable import Variable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle as pkl
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
def get_score(y_pred,y):
    tmp = y_pred - y
    tmp = np.multiply(tmp,tmp)
    score = tmp.mean()
    return score


class zhengqi(nn.Module):
    def __init__(self):
        super(zhengqi,self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(38),
            nn.Linear(38,20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20,1)
        )

    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    model = RandomForestRegressor(n_estimators=666, max_depth=28, max_features=0.7, n_jobs=3)
    warnings.filterwarnings('ignore')

    df = pd.read_table('zhengqi_train.txt')
    dataset = df.values
    corr = np.corrcoef(dataset.T)[-1,:-1]
    useful_index = np.where(np.abs(corr)>0.1)[0]
    x = df.values[:,useful_index] #
    y = df.values[:,-1]
    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40 , shuffle= False)
    # scaler = MinMaxScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    #
    #

    #
    # paras = {'n_estimators': [10, 20, 50, 100, 200],
    #          'max_depth': [10, 20, 30],
    #          'max_features': [0.5, 0.6, 0.7, 0.8]}
    #

    # # z = zhengqi().cuda()
    # # opz = torch.optim.Adam(params=z.parameters(),lr=1e-4)
    # # dataset = Data.TensorDataset(Variable(torch.Tensor(x_train)),Variable(torch.Tensor(y_train[:,np.newaxis])))
    # # loader = Data.DataLoader(dataset=dataset,batch_size=200,shuffle = True)
    # # crit = torch.nn.MSELoss()
    # # EPOCH = 3000
    # # bestscore = 10
    # # besterr = 10
    # # y2 = pd.read_table('result1.txt')
    # # y2 = y2.values
    # # for e in range(EPOCH):
    # #     for step,(b_x,b_y) in enumerate(loader):
    # #         b_x = b_x.cuda()
    # #         b_y = b_y.cuda()
    # #         out = z(b_x)
    # #         lmd = torch.tensor(0.2).cuda()
    # #         l2_reg = torch.tensor(0.).cuda()
    # #         for param in z.parameters():
    # #             l2_reg += torch.norm(param)
    # #         l2_reg = l2_reg
    # #         loss = crit(out,b_y) + lmd*l2_reg
    # #         opz.zero_grad()
    # #         loss.backward()
    # #         opz.step()
    # #     y_pred = z(torch.Tensor(x_test).cuda())
    # #     score = mean_squared_error(y_test,y_pred.cpu().data.numpy())
    # #
    # #     x_test1 = pd.read_table('zhengqi_test.txt')
    # #     x_test1 = x_test1.values
    # #     y_pred = z(torch.Tensor(x_test1).cuda())
    # #     err = mean_squared_error(y2,y_pred.cpu().data.numpy())
    # #     if(score < bestscore ):
    # #         bestscore = score
    # #         besterr = err
    # #         with open('result.txt', 'w') as f:
    # #             y_pred = y_pred.cpu().data.numpy().reshape(-1)
    # #             for y in y_pred:
    # #                 f.write(str(y) + '\n')
    # #         torch.save(z,'zhengqimodel.pkl')
    # #
    # #     print('EPOCH [{}/{}]: LOSS:{},SCORE:{}，ERROR{}'.format(e,EPOCH,loss.cpu().data.numpy(),score,err))
    # #     print('model saved and score:{},err:{}'.format(bestscore, besterr))
    # # z = torch.load('zhengqimodel.pkl')
    #
    #
    # model.fit(x_train,y_train)
    # y_pred_offline = model.predict(x_test)
    # score = mean_squared_error(y_pred_offline,y_test)
    # print("offline score:{:.4f}".format(score))
    model.fit(x,y)
    df_online = pd.read_table('zhengqi_test.txt').values[:,useful_index]
    df_online = scaler.transform(df_online)
    y_pred = model.predict(df_online)
    with open('result.txt', 'w') as f:
        # y_pred = y_pred.cpu().data.numpy().reshape(-1)
        for y in y_pred:
            f.write(str(y) + '\n')
    print('ok')