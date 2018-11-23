import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_table('zhengqi_train.txt')

data_features = df.drop('target',axis=1)
data_target = df.target
print(data_target.head(10))