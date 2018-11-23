import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle as pkl
import warnings

def get_score(y_pred,y):
    tmp = y_pred - y
    tmp = np.multiply(tmp,tmp)
    score = tmp.mean()
    return score

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    useful_index = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 16, 20, 23, 24,27, 31, 36, 37])
    df = pd.read_table('zhengqi_train.txt')
    # col = df.columns.tolist()
    # corr = np.array(df[col].corr()['target'])[:-1]


    df_tr = df.iloc[:1500,:]
    df_te = df.iloc[1500:, :]

    # x_train = df_tr.drop('target',axis=1)
    # y_train = df_tr.target

    x_train = df.drop('target',axis=1)
    y_train = df.target

    x_train = np.array(x_train)[:,useful_index]
    y_train = np.array(y_train)

    x_test = pd.read_table('zhengqi_test.txt')

    # x_test = df_te.drop('target',axis=1)
    y_test = np.array(df_te.target)

    x_test = np.array(x_test)[:,useful_index]
    y_test = np.array(y_test)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    paras={'n_estimators':[10,20,50,100,200,250],
          'max_depth':[10,20,30],
          'max_features':[0.6,0.7,0.8,0.9]}

    rfr=RandomForestRegressor()

    rfr_cv=GridSearchCV(estimator=rfr,param_grid=paras,cv=5,n_jobs=-1)


    # rfr_cv.fit(x_train,y_train);
    # with open('model.pickle','wb') as f:
    #     rfr_cv = pkl.dump(rfr_cv,f)

    with open('model.pickle','rb') as f:
        rfr_cv = pkl.load(f)
    y_pred = rfr_cv.best_estimator_.predict(x_test)
    # score = mean_squared_error(y_pred,y_test)
    # plt.figure()
    # plt.plot(y_test,c='b')
    # plt.plot(y_pred,c='r')
    # plt.show()
    # print('score:{:.4f}'.format(score))

    df = pd.read_table('zhengqi_test.txt')
    y_pred = rfr_cv.best_estimator_.predict(x_test)
    with open('result.txt', 'w') as f:
        for y in y_pred:
            f.write(str(y) + '\n')
    print('ok')