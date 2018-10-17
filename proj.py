import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from decimal import Decimal
from sklearn import neighbors
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("aps_failure_training_set_means.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
df = pd.get_dummies(df, columns=['class'])

dft = pd.read_csv("aps_failure_test_set_means.csv")
dft = dft.drop(['Unnamed: 0'], axis = 1)
dft = pd.get_dummies(dft, columns=['class'])


X = df.drop(['class_neg','class_pos'], axis = 1)
Y = df['class_pos']

Xt = dft.drop(['class_neg', 'class_pos'], axis = 1)
Yt = dft['class_pos']

X, Y = SMOTE(ratio=1).fit_sample(X, Y)
Xt, Yt = SMOTE(ratio=1).fit_sample(Xt, Yt)
#xtt = pd.concat([X, Xt], axis = 0)
#ytt = pd.concat([Y, Yt], axis = 0)
#
#cross_val_score = cross_val_score(clf, xtt, ytt, cv=10)
#print("cross val score: ", cross_val_score)
#print("mean score: ", cross_val_score.mean())

#clf = neighbors.KNeighborsClassifier(5)
clf = GaussianNB()
clf.fit(X, Y)
pred = clf.predict(Xt)

pred = pred.ravel()
Yt = Yt.ravel()

cont = 0
for k in range(0,len(pred)):
    if pred[k] == Yt[k]:
        cont = cont + 1

print("Naive Bayes")
print("Accuracy (NB): ", cont/len(pred))


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
        

plt.show()

print("\n ROC chart")

temp = []
for i in range(0,len(Yt)):
    if Yt[i] == 1:
        temp.append([1, 0])
    else:
        temp.append([0, 1])
Y_test = pd.DataFrame(temp).values

i=0
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
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
plt.title('Receiver operating characteristic example')
plt.legend(loc="upper left")
plt.show()




#
#df = pd.read_csv("aps_failure_test_set.csv")
#keys = df.columns.values
#rows = df.shape[0]    
#columns = df.shape[1]
#
#keys = keys[2:]
#final = pd.DataFrame()
#means = []
#for x in keys:
#    temp = df[x]
#    golo = temp.loc[temp != 'na'].apply(float)
#    meang = golo.mean()
#    means.append(meang)
#            
#for t in range(0, columns-2):
#    for i in range(0, rows):
#        if df.at[i, keys[t]] == 'na':
#            df.at[i, keys[t]] = means[t]
#
#df.to_csv('aps_failure_test_set_means.csv')
#
#
#df = pd.read_csv("aps_failure_training_set.csv")
#X = df.drop(['class'], axis = 1)
#Y = df['class']
#
#clf = GaussianNB()
#clf.fit(X, Y)
#
#tdf = pd.read_csv("aps_failure_test_set.csv")
#
#Xt = tdf.drop(['class'], axis = 1)
#Yt = tdf['class']
#imp = Imputer(missing_values='na', strategy='mean', axis=0)
#imp = imp.fit(Xt)
#
#
#
#pred = clf.predict(Xt)
#
#pred = pred.ravel()
#
#
#cont = 0
#for k in range(0,len(pred)):
#    if pred[k] == Yt[k]:
#        cont = cont + 1
#    
#print("Accuracy (NB): ", cont/len(pred))
#
#df = pd.read_csv("aps_failure_training_set_means.csv")
#keys = df.columns.values
#rows = df.shape[0]    
#columns = df.shape[1]
#
