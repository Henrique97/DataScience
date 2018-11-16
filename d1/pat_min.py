import pandas as pd
import numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
from itertools import combinations, groupby
from collections import Counter
import time
from matplotlib.lines import Line2D


#BINARIZE
def binarize():
    df = pd.read_csv("aps_clustering.csv")
    df = df.drop(['ae_000','af_000','ag_000','ag_001','ag_002','ag_003','ag_009','ai_000','aj_000','ak_000','al_000','am_0','ar_000','as_000','at_000','au_000','ay_000','ay_001',\
        'ay_002','ay_003','ay_004','ay_005','ay_009','az_007','az_008','az_009','ba_009','cd_000','ch_000','cj_000','cl_000','cn_000','cn_001','cn_002','cs_009','cy_000','da_000',\
        'df_000','dg_000','dh_000','di_000','dj_000','dk_000','dl_000','dm_000','dq_000','dr_000','dx_000','dy_000','dz_000','ea_000','ee_009','ef_000','eg_000',\
        'ab_000', 'ag_008', 'ba_008', 'bf_000', 'bm_000', 'bn_000', 'bo_000', 'bp_000', 'bq_000', 'br_000', 'cf_000', 'cm_000', 'cn_009', 'cr_000', 'db_000', 'eb_000'], axis = 1)
    for col in list(df):
        try:
            if col in ['ag_007', 'ay_006','bc_000','bk_000','bl_000', 'cn_008', 'ee_008']:
                df[col] = pd.qcut(df[col], 3,labels=['a','b','c'], duplicates = 'drop')
    
            elif col in ['bz_000', 'co_000', 'cs_008', 'ct_000', 'do_000' ,'dp_000']:
                df[col] = pd.qcut(df[col], 4,labels=['a','b','c','d'], duplicates = 'drop')
            elif col in ['ax_000','ce_000','cu_000','cv_000','cx_000','cz_000','dc_000' ]:
                df[col] = pd.qcut(df[col], 5,labels=['a','b','c','d','e'], duplicates = 'drop')
            else:
                if col not in ['class']:
                    df[col] = pd.qcut(df[col], 6,labels=['a','b','c','d','e','f'], duplicates = 'drop')
        except ValueError:
            print(" |"+col+'| ', end=' ')
            continue          
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
    df.to_csv("d1_3_symbols_r.csv", index = False)
    
def get_apriori(df, ms):
    t1 = time.time()
    frequent_itemsets = apriori(df, min_support=ms, use_colnames=True)
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=ms)
    except ValueError:
        return np.array([])
    t2 = time.time()
    return rules

def get_avg():
   legend_elements = [Line2D([0], [0], marker='o', color='w', label='Lift',
                             markerfacecolor='r', markersize=5),
                      Line2D([0], [0], marker='o', color='w', label='Support',
                             markerfacecolor='g', markersize=5),
                      Line2D([0], [0], marker='o', color='w', label='Confidence',
                             markerfacecolor='b', markersize=5)]
   
   
   for u in np.arange(0.24, 0.15, -0.01):
       rules = get_apriori(df, u)
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

#binarize()
df = pd.read_csv("d1_3_symbols_v4.csv")
df = df.drop(['Unnamed: 0'], axis = 1)