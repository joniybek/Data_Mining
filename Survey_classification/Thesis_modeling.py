# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pandas.stats.api import ols
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import matplotlib.font_manager
from scipy import stats
from sklearn.preprocessing import scale
import sklearn.preprocessing as prep
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score

# <codecell>

#Read file
df=pd.read_csv('df3.csv')

# <codecell>

# Select features to feed tha algorithm
listOfFeatures2=['NumStud_total','1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']
feature_x=np.array(df[listOfFeatures2],dtype="float64")
#target_y=np.array(df[['numDecision']],dtype="float64")
target_y=df['numDecision'].values

# <codecell>

# Scale features
feature_x=prep.scale(feature_x)

# <codecell>

# Random forest and ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier #ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
forest = ExtraTreesClassifier(criterion='gini',n_estimators=250,random_state=1)
# Feed and evaluate with CV
%%time
scores=cross_val_score(forest,feature_x,target_y,cv=10)
scores.mean()

# <codecell>

# SVM classifier
%%time
svm1 = svm.SVC(C=1,kernel='rbf',gamma=0.0,probability=False)
scores=cross_val_score(svm1,feature_x,target_y,cv=10)
scores.mean()

# <codecell>

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
%%time
scores=cross_val_score(dt,feature_x,target_y,cv=10)
scores.mean()

# <codecell>

# Export graph of DT
tree.export_graphviz(dt,out_file='tree.dot',feature_names=listOfFeatures2)

# <codecell>

# This is for getting confusion matrix, firstly feed with train and test, reolace classifier with forest,svm1 or tree
from sklearn.metrics import confusion_matrix
X_train1, X_test1, y_train1, y_test1 = train_test_split(feature_x, target_y, test_size=0.5, random_state=0)
classifier.fit(X_train1,y_train1)
y_pred=classifier.predict(X_test1)
confusion_matrix(y_test1, y_pred)

# <codecell>

%%time
from sklearn.cross_validation import train_test_split
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(feature_x, target_y, test_size=0.2, random_state=0)

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0,1e-1,1e-2,1e-3],'C': [1, 10, 100]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100]}]


#tuned_parameters = {'n_estimators':[250],"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],
#              "min_samples_leaf": [1, 3, 10],  "bootstrap": [True, False], "criterion": ["gini", "entropy"]}


#tuned_parameters = {"max_depth": [3, None],"max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],
#              "min_samples_leaf": [1, 3, 10], "criterion": ["gini", "entropy"]}

tuned_parameters = {'n_estimators':[250],"max_depth": [3], "max_features": [10], "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [ 3],  "bootstrap": [True], "criterion": [ "entropy"]}



classifierList=list()



scores = ['precision', 'recall','f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print

    clf = GridSearchCV(RandomForestClassifier(n_jobs=3), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print
    print(clf.best_params_)
    print
    print("Grid scores on development set:")
    print
    #for params, mean_score, scores in clf.grid_scores_:
    #    print("%0.3f (+/-%0.03f)  %r"
    #          % (mean_score, scores.std() * 2, params))
    print

    print("Detailed classification report:")
    print
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    classifierList.append(clf.best_estimator_)
    print

# <codecell>

%%time
#this is rechecking avarage score with cross-valid
scores=cross_val_score(classifierList[0],feature_x,target_y,cv=10)

# <codecell>

scores.mean()

# <codecell>

# this is for exporting dot file
tree.export_graphviz(classifierList[0],out_file='tree.dot',feature_names=listOfFeatures2)

# <codecell>

classifierList[0]

# <codecell>

feature_x

# <codecell>


# <codecell>

#this is for plotting feature importance
classifierList[1].fit(feature_x,target_y)
pd.DataFrame(classifierList[1].feature_importances_,index=listOfFeatures2 ).plot(kind='bar')
plt.show()

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>

#this is part with teachers

# <codecell>

listOfFeatures=['Teacher','numDecision','NumStud_total','NumStud_filled','1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']
df2=df[df['numDecision']>1][listOfFeatures].T.to_dict().values()

# <codecell>

vec = DictVectorizer(sparse = False)
npv=vec.fit_transform(df2)
dfb=pd.DataFrame(npv,columns=vec.get_feature_names())
dfb['otherTeacher']=0

# <codecell>

listOfFeatures=['Teacher','numDecision','NumStud_total','NumStud_filled','1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']
dfw=df[df['numDecision']<=1][listOfFeatures]

# <codecell>

#del dfw['Teacher']
dfw['otherTeacher']=1
#notListOfFeatures=[x for x in vec.get_feature_names() if x not in listOfFeatures ]
#dfw[[ [x] for x in notListOfFeatures]]=0
#dfw.head().T
df3=pd.concat([dfb,dfw], axis=0)
df3=df3.fillna(0)

# <codecell>

svm1 = svm.SVC(kernel='rbf')
feature_x2=np.array(df3[[x for x in df3.columns if x!='numDecision']],dtype="float64")
#target_y=np.array(df[['numDecision']],dtype="float64")
target_y2=df['numDecision'].values

# <codecell>

%%time
scores=cross_val_score(svm1,feature_x2,target_y2,cv=10)

# <codecell>

scores.mean()

# <codecell>


