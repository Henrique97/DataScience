import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

methods = ['KNN','Naive Bayes', 'DecisionTree']

df = pd.read_csv("all_mean_in_class.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
df = pd.get_dummies(df, columns=['class'])

dft = pd.read_csv("all_mean_in_class_test.csv")
dft = dft.drop(['Unnamed: 0'], axis = 1)
dft = pd.get_dummies(dft, columns=['class'])

X = df.drop(['class_neg','class_pos'], axis = 1)
Y = df['class_pos']
X, Y = SMOTE(ratio=1).fit_sample(X, Y)

Xt = dft.drop(['class_neg', 'class_pos'], axis = 1)
Yt = dft['class_pos']

def classificar(X, Y, Xt, Yt, method = 'LinearSVC'):
   
   if method == 'Decision Tree': 
       clf = DecisionTreeClassifier()
   elif method == 'Naive Bayes':
       clf = GaussianNB()
   elif method == 'LinearSVC':
       #clf = CalibratedClassifierCV(LinearSVC()) 
       clf = LogisticRegression()
   elif re.search(r"KNN\d", method).group() != '':
       clf = neighbors.KNeighborsClassifier(method[3:])
   clf.fit(X, Y)
   pred = clf.predict(Xt)
    
   pred = pred.ravel()
   Yt = Yt.ravel()
    
   cont = 0
   for k in range(0,len(pred)):
       if pred[k] == Yt[k]:
           cont = cont + 1

   print(method)
   print("Accuracy: ", cont/len(pred))
   
#CLASS COMP ACC
   cont0 = 0
   cont1 = 0
   s0, s1 = 0, 0
   for k in range(0,len(pred)):
       if Yt[k]==1:
           s1 += 1
           if pred[k] == Yt[k]:
               cont1 = cont1 + 1
       elif Yt[k]==0:
           s0 += 1
           if pred[k] == Yt[k]:
               cont0 = cont0 + 1 
               
   print("\nAccuracy0: ", cont0/s0)
   print("Accuracy1: ", cont1/s1)
   print("Avg: ", (cont0/s0 + cont1/s1)/2)
   print("")
    
    
   cm = confusion_matrix(Yt, pred)

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
           plt.text(j - 0.2, i, str(s[i][j])+" = "+str(cm[i][j]), color = 'r')
   print("sensitivity", cm[1][1]/(cm[1][1] + cm[1][0]))
   print("specificity", cm[0][0]/(cm[0][0] + cm[0][1]))
   plt.show()
    
   
   print("\n ROC chart")
   
   temp = []
   for i in range(0,len(Yt)):
       if Yt[i] == 0:
           temp.append([1, 0])
       else:
           temp.append([0, 1])
   Y_test = pd.DataFrame(temp).values
   
   i=0
   n_classes = 2
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
  # if (method == 'LinearSVC'):
  # y_score = clf.fit(X, Y).score(Xt)    
 #  else:
   y_score = clf.fit(X, Y).predict_proba(Xt)
       
   for i in range(n_classes):
       fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])
   
   # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
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
   plt.title('Receiver operating characteristic')
   plt.legend(loc="upper left")
   plt.show()

classificar(X, Y, Xt, Yt)
