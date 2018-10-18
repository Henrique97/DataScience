# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:58:26 2018

@author: Tomas
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("green.csv")

X = df.drop(['experts::0','experts::1','experts::2','experts::3','experts::4',\
                  'experts::5', 'consensus'], axis = 1)

Y = pd.concat([df['experts::0'], df['experts::1'], df['experts::2'], df['experts::3'], \
                 df['experts::4'], df['experts::5'], df['consensus']], axis = 1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7)

typeA = "KNN"

if typeA == "Naive Bayes":
    print("Naive Bayes")
    clf = OneVsRestClassifier(GaussianNB())

if typeA == "KNN":
    print("KNN")
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
    
clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
cont = 0
for i in range(0, pred.shape[0]):
    for j in range(0, pred.shape[1]):
        if pred[i][j] == Y_test.values[i][j]:
            cont = cont + 1

print("Accuracy (NB): ", cont/(pred.shape[0]*pred.shape[1]))

labels = pd.unique(Y_train.values[0])

preddf = pd.DataFrame(data = pred, columns = ['experts::0','experts::1','experts::2','experts::3',\
                                              'experts::4', 'experts::5', 'consensus'] )


keys = preddf.columns.values 
for xx in keys:
   print("\n\n",xx)
   cm = confusion_matrix(Y_test[xx].values, preddf[xx].values)
    
   plt.imshow(cm, interpolation='nearest')
   classNames = ['Negative','Positive']
   plt.title('Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   tick_marks = np.arange(len(classNames))
   plt.xticks(tick_marks, classNames)
   plt.yticks(tick_marks, classNames)
   s = [['TN','FP'], ['FN', 'TP']]
   for i in range(2):
       for j in range(2):
           plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), color = 'r')
   print("sensitivity", cm[1][1]/(cm[1][1] + cm[1][0]))
   print("specificity", cm[0][0]/(cm[0][0] + cm[0][1]))
   plt.show()
   
   print("\n ROC chart")

   temp = []
   for i in range(0, len(Y_test[xx].values)):
       if Y_test[xx].values[i] == 0:
           temp.append([1, 0])
       else:
           temp.append([0, 1])
   Y_testcm = pd.DataFrame(temp).values
   
   i=0
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   y_score = clf.fit(X_train, Y_train[xx].values).predict_proba(X_test)
   for i in range(2):
       fpr[i], tpr[i], _ = roc_curve(Y_testcm[:, i], y_score[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])
   
   # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], _ = roc_curve(Y_testcm.ravel(), y_score.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
   
   plt.figure()
   lw = 2
   plt.plot(fpr[1], tpr[1], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver operating characteristic example')
   plt.legend(loc="lower right")
   plt.show()
   
   

    

