import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import datasets, neighbors
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier                          
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
#print(__doc__)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

LW = 2
RANDOM_STATE = 42
repeater = 200
sens = []
spec = []
accu = []

class DummySampler(object):

    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


cv = StratifiedKFold(n_splits=3)

# Load the dataset
df = pd.concat([pd.read_csv("hinselmann.csv"),pd.read_csv("green.csv"),pd.read_csv("schiller.csv")])
df = pd.get_dummies(df, columns=['consensus'])
X = df.drop(['experts::0','experts::1','experts::2','experts::3','experts::4',\
                  'experts::5', 'consensus_0.0','consensus_1.0'], axis = 1)
Y = df['consensus_1.0']

X,y = X.values, Y.values


#X=MinMaxScaler().fit_transform(X)

classifiers = [['DT-Gini impurity', tree.DecisionTreeClassifier(criterion='gini')],['DT-Info Gain', tree.DecisionTreeClassifier(criterion='entropy')]]

samplers = [
    ['', DummySampler()]
]

pipelines = [
    ['{}{}'.format(samplers[0][0], classifier[0]),
     make_pipeline(samplers[0][1], classifier[1])]
    for classifier in classifiers
]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Allmean_tpr = dict()
Allmean_sens = dict()
Allmean_spec = dict()
Allmean_accu = dict()
for name, pipeline in pipelines:
    Allmean_tpr[name]=0.0
    Allmean_sens[name]=0.0
    Allmean_spec[name]=0.0
    Allmean_accu[name]=0.0
    
Allmean_fpr = np.linspace(0, 1, 100)
for i in range(0,repeater):
    for name, pipeline in pipelines:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, y):
            classif = pipeline.fit(X[train], y[train])
            probas_ = classif.predict_proba(X[test])
            pred = classif.predict(X[test])
            cm1 = confusion_matrix(y[test],pred)
            Allmean_spec[name] += cm1[1,1]/(cm1[1,0]+cm1[1,1])
            Allmean_sens[name] += cm1[0,0]/(cm1[0,0]+cm1[0,1])
            total1=sum(sum(cm1))
            Allmean_accu[name]+=(cm1[0,0]+cm1[1,1])/total1
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            
        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        Allmean_tpr[name]+=mean_tpr

print("--------Sem Smote--------")
for name, pipeline in pipelines:   
    Allmean_tpr[name] /= repeater
    print("-----",name,"-----")
    print('Sensitivity : ', Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    print('Specificity : ', Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    print ('Accuracy : ', Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    spec.append(Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    sens.append(Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    accu.append(Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    Allmean_auc = auc(Allmean_fpr, Allmean_tpr[name])
    plt.plot(Allmean_fpr, Allmean_tpr[name], linestyle='--',
             label='{} (area = %0.2f)'.format(name) % Allmean_auc, lw=LW)

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
plt.title('Variação de criterion')

plt.legend(loc="lower right")

plt.show()

##############################################################################
sens = []
spec = []
accu = []

classifiers = [['DT-none', tree.DecisionTreeClassifier(class_weight=None)],['DT-balanced', tree.DecisionTreeClassifier(class_weight='balanced')]]

samplers = [
    ['', DummySampler()]
]

pipelines = [
    ['{}{}'.format(samplers[0][0], classifier[0]),
     make_pipeline(samplers[0][1], classifier[1])]
    for classifier in classifiers
]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Allmean_tpr = dict()
Allmean_sens = dict()
Allmean_spec = dict()
Allmean_accu = dict()
for name, pipeline in pipelines:
    Allmean_tpr[name]=0.0
    Allmean_sens[name]=0.0
    Allmean_spec[name]=0.0
    Allmean_accu[name]=0.0
    
Allmean_fpr = np.linspace(0, 1, 100)
for i in range(0,repeater):
    for name, pipeline in pipelines:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, y):
            classif = pipeline.fit(X[train], y[train])
            probas_ = classif.predict_proba(X[test])
            pred = classif.predict(X[test])
            cm1 = confusion_matrix(y[test],pred)
            Allmean_spec[name] += cm1[1,1]/(cm1[1,0]+cm1[1,1])
            Allmean_sens[name] += cm1[0,0]/(cm1[0,0]+cm1[0,1])
            total1=sum(sum(cm1))
            Allmean_accu[name]+=(cm1[0,0]+cm1[1,1])/total1
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            
        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        Allmean_tpr[name]+=mean_tpr

print("--------Sem Smote--------")
for name, pipeline in pipelines:   
    Allmean_tpr[name] /= repeater
    print("-----",name,"-----")
    print('Sensitivity : ', Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    print('Specificity : ', Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    print ('Accuracy : ', Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    spec.append(Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    sens.append(Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    accu.append(Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    Allmean_auc = auc(Allmean_fpr, Allmean_tpr[name])
    plt.plot(Allmean_fpr, Allmean_tpr[name], linestyle='--',
             label='{} (area = %0.2f)'.format(name) % Allmean_auc, lw=LW)

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
plt.title('Variação de class_weight')

plt.legend(loc="lower right")

plt.show()

###############################################################################
sens = []
spec = []
accu = []

classifiers = [['DT-1', tree.DecisionTreeClassifier(max_depth=1)],
                ['DT-2', tree.DecisionTreeClassifier(max_depth=2)],
                ['DT-3', tree.DecisionTreeClassifier(max_depth=3)],
                ['DT-4', tree.DecisionTreeClassifier(max_depth=4)],
                ['DT-5', tree.DecisionTreeClassifier(max_depth=5)],
                ['DT-7', tree.DecisionTreeClassifier(max_depth=7)],
                ['DT-None', tree.DecisionTreeClassifier()]]

samplers = [
    ['', DummySampler()]
]

pipelines = [
    ['{}{}'.format(samplers[0][0], classifier[0]),
     make_pipeline(samplers[0][1], classifier[1])]
    for classifier in classifiers
]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Allmean_tpr = dict()
Allmean_sens = dict()
Allmean_spec = dict()
Allmean_accu = dict()
for name, pipeline in pipelines:
    Allmean_tpr[name]=0.0
    Allmean_sens[name]=0.0
    Allmean_spec[name]=0.0
    Allmean_accu[name]=0.0
    
Allmean_fpr = np.linspace(0, 1, 100)
for i in range(0,repeater):
    for name, pipeline in pipelines:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, y):
            classif = pipeline.fit(X[train], y[train])
            probas_ = classif.predict_proba(X[test])
            pred = classif.predict(X[test])
            cm1 = confusion_matrix(y[test],pred)
            Allmean_spec[name] += cm1[1,1]/(cm1[1,0]+cm1[1,1])
            Allmean_sens[name] += cm1[0,0]/(cm1[0,0]+cm1[0,1])
            total1=sum(sum(cm1))
            Allmean_accu[name]+=(cm1[0,0]+cm1[1,1])/total1
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            
        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        Allmean_tpr[name]+=mean_tpr

print("--------Sem Smote--------")
for name, pipeline in pipelines:   
    Allmean_tpr[name] /= repeater
    print("-----",name,"-----")
    print('Sensitivity : ', Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    print('Specificity : ', Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    print ('Accuracy : ', Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    spec.append(Allmean_spec[name]/(repeater*cv.get_n_splits(X,y)))
    sens.append(Allmean_sens[name]/(repeater*cv.get_n_splits(X,y)))
    accu.append(Allmean_accu[name]/(repeater*cv.get_n_splits(X,y)))

    Allmean_auc = auc(Allmean_fpr, Allmean_tpr[name])
    plt.plot(Allmean_fpr, Allmean_tpr[name], linestyle='--',
             label='{} (area = %0.2f)'.format(name) % Allmean_auc, lw=LW)

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
plt.title('Variação de max_depth')

plt.legend(loc="lower right")

plt.show()

fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,6,7], accu[0:7])

ax.set(xlabel='max_depth', ylabel='accuracy',
       title='Accuracy')
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,6,7], spec[0:7])

ax.set(xlabel='max_depth', ylabel='specificity',
       title='Specificity')
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.plot([1,2,3,4,5,6,7], sens[0:7])

ax.set(xlabel='max_depth', ylabel='sensitivity',
       title='Sensitivity')
ax.grid()

plt.show()