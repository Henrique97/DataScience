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
#print(__doc__)
from sklearn.metrics import confusion_matrix


LW = 2
RANDOM_STATE = 42
repeater=1

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

# Load the dataset
dftraining = pd.read_csv("all_mean_in_class.csv")
dftraining  = pd.get_dummies(dftraining, columns=['class'])
dftraining = dftraining.drop(['Unnamed: 0','class_neg'], axis=1)

dftest = pd.read_csv("training_set_media_geral.csv")
dftest  = pd.get_dummies(dftest, columns=['class'])
dftest = dftest.drop(['Unnamed: 0','class_neg'], axis=1)

Xtraining = dftraining.drop(['class_pos'], axis = 1)
Ytraining = dftraining['class_pos']

Xtesting = dftest.drop(['class_pos'], axis = 1)
Ytesting = dftest['class_pos']


Xtraining,ytraining = Xtraining.values, Ytraining.values

Xtesting, ytesting = Xtesting.values,Ytesting.values

Xtraining=MinMaxScaler().fit_transform(Xtraining)
Xtesting=MinMaxScaler().fit_transform(Xtesting)

classifiers= [['DT-None', tree.DecisionTreeClassifier()]]

values = np.array([])
values = (np.logspace(1,9,dtype='int'))[3:10]

for i in values:
    classifier =[]
    string = 'RF-{}'.format(i)
    classifier.append(string)
    classifier.append(RandomForestClassifier(n_estimators=i))
    classifiers.append(classifier)

samplers = [
    ['Estimators', DummySampler()]
]

pipelines = [
    ['{}-{}'.format(samplers[0][0], classifier[0]),
     make_pipeline(samplers[0][1], classifier[1])]
    for classifier in classifiers
]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

Allmean_tpr = dict()
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
        classif = pipeline.fit(Xtraining,ytraining)
        probas_ = classif.predict_proba(Xtesting)
        pred = classif.predict(Xtesting)
        cm1 = confusion_matrix(ytesting,pred)
        Allmean_spec[name] += cm1[1,1]/(cm1[1,0]+cm1[1,1])
        Allmean_sens[name] += cm1[0,0]/(cm1[0,0]+cm1[0,1])
        total1=sum(sum(cm1))
        Allmean_accu[name]+=(cm1[0,0]+cm1[1,1])/total1
        fpr, tpr, thresholds = roc_curve(ytesting, probas_[:, 1])
        mean_tpr = interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        mean_tpr[-1] = 1.0
        Allmean_tpr[name]+=mean_tpr
        
for name, pipeline in pipelines: 
    
    print("-----",name,"-----")
    print('Sensitivity : ', Allmean_sens[name]/(repeater))
    print('Specificity : ', Allmean_spec[name]/(repeater))
    print ('Accuracy : ', Allmean_accu[name]/(repeater))

    spec.append(Allmean_spec[name]/(repeater))
    sens.append(Allmean_sens[name]/(repeater))
    accu.append(Allmean_accu[name]/(repeater))
    
    Allmean_tpr[name] /= repeater
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
plt.title('N_estimators')

plt.legend(loc="lower right")

plt.show()

fig, ax = plt.subplots()
ax.plot(values, accu[1:(len(values)+1)])

ax.set(xlabel='N_estimators', ylabel='accuracy',
       title='Accuracy')
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.plot(values, spec[1:(len(values)+1)])

ax.set(xlabel='N_estimators', ylabel='specificity',
       title='Specificity')
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.plot(values, sens[1:(len(values)+1)])

ax.set(xlabel='N_estimators', ylabel='sensitivity',
       title='Sensitivity')
ax.grid()

plt.show()