import pandas as pd
import numpy as np
import arff
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
from itertools import combinations, groupby
from collections import Counter
import pandas.core.algorithms as algos
import time
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


df = pd.read_csv("d2_3_ds_merged.csv")
df = df.drop(['experts::0','experts::1','experts::2','experts::3','experts::4','experts::5','os_artifacts_area','walls_artifacts_area','speculum_artifacts_area',\
    'os_specularities_area', 'cervix_artifacts_area', 'walls_specularities_area'], axis = 1)

#df = pd.read_csv("X_scaled.csv")
#df1 = pd.read_csv("Y_to_n.csv")
#df = pd.concat([df, df1], axis = 1)

def binarize(df):
    for col in list(df) :
        if col not in ['consensus']:
            df[col] = pd.qcut(df[col], 3, labels = ['0','1','2'])
        attrs = []
        values = df[col].unique().tolist()
        values.sort()
        for val in values : attrs.append("%s:%s"%(col,val))
        lb = LabelBinarizer().fit_transform(df[col])
        if(len(attrs)==2) :
            v = list(map(lambda x: 1 - x, lb))
            lb = np.concatenate((lb,v),1)
        df2 = pd.DataFrame(data=lb, columns=attrs)
        df = df.drop(columns=[col])
        df = pd.concat([df,df2], axis=1, join='inner')

    df.to_csv("df_binarized_final.csv", index = False)
    return df
    
def get_rules(df, sp):
    frequent_itemsets = apriori(df, min_support=sp, use_colnames=True)
   #print(frequent_itemsets)
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=sp)
    except ValueError:
        return np.array([])
    rules.to_csv("rules" + str(sp) + ".csv")
    return rules

#bdf = binarize(df)
bdf = pd.read_csv("df_binarized_final.csv")


def get_avg():
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Lift',
                              markerfacecolor='r', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Support',
                              markerfacecolor='g', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Confidence',
                              markerfacecolor='b', markersize=5)]
    
    
    for u in np.arange(0.33, 0.16, -0.01):
        rules = get_rules(bdf, u)
        rule_l = []
        print(rules.shape[0],u)
        rules = rules.sort_values(by=['conviction','lift','confidence'])
        lis = []
        #lis.append(['conviction','lift','support','confidence'])
        top = [rules.tail(10),rules.tail(50),rules.tail(100)]
        for t in top:
            lis.append([t['conviction'].mean(), t['lift'].mean(), t['support'].mean(), t['confidence'].mean()])
        
        plt.scatter(u,np.float32(lis[0][1]), c = 'red', label = 'red', s=10)
        plt.scatter(u,np.float32(lis[0][2]), c = 'green', label = 'green', s=10)
        plt.scatter(u,np.float32(lis[0][3]), c = 'blue', label = 'blue', s=10)
        plt.legend(handles=legend_elements)
        print("---" + str(u) + "---")
        print(lis)
    plt.show()
