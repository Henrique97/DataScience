import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

LW = 2
RANDOM_STATE = 42
repeater=3

class DummySampler(object):

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)

# Load the dataset
dftraining = pd.read_csv("all_mean_in_class_training.csv")
dftraining  = pd.get_dummies(dftraining, columns=['class'])
dftraining = dftraining.drop(['Unnamed: 0','class_neg'], axis=1)

dftest = pd.read_csv("all_mean_in_class_test.csv")
dftest  = pd.get_dummies(dftest, columns=['class'])
dftest = dftest.drop(['Unnamed: 0','class_neg'], axis=1)

Xtraining = dftraining.drop(['class_pos'], axis = 1)
Ytraining = dftraining['class_pos']

Xtesting = dftest.drop(['class_pos'], axis = 1)
Ytesting = dftest['class_pos']


Xtraining,ytraining = Xtraining.values, Ytraining.values

Xtesting, ytesting = Xtesting.values,Ytesting.values

Xtraining = StandardScaler().fit_transform(Xtraining)
Xtesting = StandardScaler().fit_transform(Xtesting)

classifiers = [['3nn',KNeighborsClassifier(3)], \
               ['Rf',RandomForestClassifier()], \
               ['DT', DecisionTreeClassifier()], \
               ['NBG', GaussianNB()]]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Allmean_tpr = dict()
for classifier in classifiers:
    Allmean_tpr[classifier[0]]=0.0
Allmean_fpr = np.linspace(0, 1, 100)
for i in range(0,repeater):
    for classifier in classifiers:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        Xtrainer, Ytrainer = SMOTE(ratio=1).fit_sample(Xtraining, ytraining)
        probas_ = classifier[1].fit(Xtrainer, Ytrainer).predict_proba(Xtesting)
        fpr, tpr, thresholds = roc_curve(ytesting, probas_[:, 1])
        mean_tpr = interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr) 
        mean_tpr[-1] = 1.0
        Allmean_tpr[classifier[0]]+=mean_tpr
        
for classifier in classifiers:   
    Allmean_tpr[classifier[0]] /= repeater
    Allmean_auc = auc(Allmean_fpr, Allmean_tpr[classifier[0]])
    plt.plot(Allmean_fpr, Allmean_tpr[classifier[0]], linestyle='--',
             label=f'{classifier[0]} (area = %0.2f)'.format(classifier[0]) % Allmean_auc, lw=LW)
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=LW, color='k',
         label='Luck')

# make nice plotting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Com SMOTE')

plt.legend(loc="lower right")

plt.show()
