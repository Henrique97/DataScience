import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from decimal import Decimal
from sklearn import neighbors
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv("aps_failure_test_set.csv")

def get_avg(df):
    df = df[df['class']=='pos']
    
    for k in df.columns.values[1:]:
        df[k].replace('na',np.nan,inplace= True)
        df[k].dropna(inplace = True)
    
    
    medias_neg = pd.DataFrame(columns = df.columns.values[1:])
    
    
    for k in df.columns.values[1:]:
        medias_neg.loc[:,k] = [df[k].map(float, na_action='ignore').mean()]
    
    medias_neg.to_csv("medias_pos.csv")


def imputer(df):
    dfp = pd.read_csv("medias_pos.csv")
    #dfn = pd.read_csv("medias_neg.csv")
    df_copy = df.copy()
    df_copy = df_copy[df_copy['class']=='pos']
    
    for k in df.columns.values[1:]:
        df_copy[k].replace('na',dfp[k].values[0],inplace= True)
   
    return df_copy


def joiner(x,y):
    dfp = pd.read_csv(x)
    dfn = pd.read_csv(y)
    return pd.concat([dfp,dfn])

def get_avg_all(df):
    
    for k in df.columns.values[1:]:
        df[k].replace('na',np.nan,inplace= True)
        df[k].dropna(inplace = True)
    
    
    medias_all = pd.DataFrame(columns = df.columns.values[1:])
    
    
    for k in df.columns.values[1:]:
        medias_all.loc[:,k] = [df[k].map(float, na_action='ignore').mean()]
    
    medias_all.to_csv("medias_all_all.csv")


def imputer_all(df):
    dfp = pd.read_csv("medias_all_all.csv")
    df_copy = df.copy()
    
    for k in df.columns.values[1:]:
        df_copy[k].replace('na',dfp[k].values[0],inplace= True)
   
    return df_copy
