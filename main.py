import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle as pkl
import warnings



if __name__ == "__main__":

    warnings.filterwarnings('ignore')


    df_tr = pd.read_table('zhengqi_train.txt')
    df_te = pd.read_table('zhengqi_test.txt')

    x_train = df_tr.drop('target',axis=1)
    y_train = df_tr.target

    x_test = df_te

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    paras={'n_estimators':[10,20,50,100,200],
          'max_depth':[10,20,30],
          'max_features':[0.5,0.6,0.7,0.8]}

    rfr=RandomForestRegressor()

    rfr_cv=GridSearchCV(estimator=rfr,param_grid=paras,cv=5,n_jobs=-1)
    with open('model.pickle','rb') as f:
        rfr_cv = pkl.load(f)
    y_pred = rfr_cv.best_estimator_.predict(x_test)
    # score = accuracy_score(y_test,y_pred)
    # print(score)
    # rfr_cv.fit(x_train,y_train);
    print(y_pred)
    print('ok')