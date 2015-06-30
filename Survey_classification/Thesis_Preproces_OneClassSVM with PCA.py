# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%%time
from pandas.stats.api import ols
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from unidecode import unidecode
import matplotlib.font_manager
from scipy import stats
from sklearn.preprocessing import scale
import sklearn.preprocessing as prep
from sklearn import svm
from sklearn.decomposition import PCA

# <codecell>

#Here we read file, it can be Excel file, csv, database connection, etc
%%time
df=pd.read_csv('thesis_data.csv',sep=',',encoding='cp775')
df["Teacher"] = df["Teacher"].apply(lambda x : unidecode(x))
                                               
df["Course code"] = df["Course code"].apply(lambda x : unidecode(x))

# <codecell>

a=df

# <codecell>


a=a[a['Exclude 1']==0]

# <codecell>


# <codecell>

df.head()

# <codecell>

df=df[df['Exclude 1']!=0]

# <codecell>

#this is thesis stuff
t=np.array(df[['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']],dtype="float64")
df['numDecision']=df.apply(mapDecision,axis=1)
target_y=df['numDecision'].values

# <codecell>

tmp2=prep.scale(t)
tmp=tmp2
#pd.DataFrame(tmp2).plot(kind='box')
#plt.show()

# <codecell>

#minmax=prep.MinMaxScaler()
#tmp2=minmax.fit_transform(t)
#pd.DataFrame(tmp2).plot(kind='box')
#plt.show()
#tmp=tmp2

# <codecell>

#tmp2=prep.normalize(tmp,norm='l2')
#pd.DataFrame(tmp2).plot(kind='box')
#plt.show()
#tmp=tmp2

# <codecell>

#tmp2=prep.Binarizer().fit(t).transform(t)
#pd.DataFrame(tmp2).plot(kind='box')
#plt.show()
#tmp=tmp2
len(x_new[:,])

# <codecell>

clf=PCA(n_components=2)
X = clf.fit_transform(tmp)
x_new=clf.inverse_transform(X)
plt.plot(x_new[:,],x_new[:,1],'og',alpha=0.8)
plt.title('After dropping')
plt.show()

# <codecell>

#x_new.shape
x_new[:,1].shape

# <codecell>

plt.scatter(X[:,0],X[:,1],alpha=0.8,c=target_y)

#a1 = plt.scatter(X[is_inlier == 0, 0], X[is_inlier == 0, 1], c='white')

plt.legend(["Not considered"])
plt.show()

# <codecell>

count_n=tmp.shape[0]
OUTLIER_FRACTION = 0.26

# <codecell>

clf=svm.OneClassSVM(kernel='rbf')
clf.fit(tmp)

# <codecell>

dist_to_border = clf.decision_function(tmp).ravel()
threshold = stats.scoreatpercentile(dist_to_border,
            100 * OUTLIER_FRACTION)

is_inlier = dist_to_border > threshold 

# <codecell>

df[is_inlier == 0].Decision.value_counts()

# <codecell>

pd.DataFrame(dist_to_border).plot()
plt.show()

# <codecell>

df.Decision.value_counts()

# <codecell>

#PCA
X = PCA(n_components=2).fit_transform(tmp)
clf=svm.OneClassSVM(kernel='rbf')
clf.fit(X)
count_n=X.shape[0]
dist_to_border = clf.decision_function(X).ravel()
threshold = stats.scoreatpercentile(dist_to_border,
            100 * OUTLIER_FRACTION)
is_inlier = dist_to_border > threshold

# <codecell>

#graph SVM+PCA
xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
n_inliers = int((1. - OUTLIER_FRACTION) * count_n)
n_outliers = int(OUTLIER_FRACTION * count_n)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.title("Outlier detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
a1 = plt.contour(xx, yy, Z, levels=[threshold],
                            linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
b1 = plt.scatter(X[is_inlier == 0, 0], X[is_inlier == 0, 1], c='white')
c1 = plt.scatter(X[is_inlier == 1, 0], X[is_inlier == 1, 1], c='black')
plt.axis('tight')
plt.legend([a1.collections[0], b1, c1],
           ['learned decision function', 'outliers', 'inliers'],
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlim((-7, 7))
plt.ylim((-7, 7))
plt.show()

# <codecell>

df1=df[is_inlier == 0]
df1=df1[df1['Decision']!='Not considered']
df1=df1.append(df[is_inlier != 0])
#df1[['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']].plot(kind='box')
#plt.show()

# <codecell>

#df1.to_csv('new_dataset.csv',encoding='utf-8')

# <codecell>

def filled(x):
    if x['N of students in the course']>29:
        return 29*x['N of surveys filled']/x['N of students in the course']
    else:
        return x['N of surveys filled']

# <codecell>

def total(x):
    if x['N of students in the course']>29:
        return 29
    else:
        return x['N of students in the course']

# <codecell>

df1['NumStud_filled_created']=df1.apply(filled,axis=1)
df1['Stud_total_created']=df1.apply(total,axis=1)
del df1['N of students in the course']
del df1['N of surveys filled']

# <codecell>

df1.head()

# <codecell>

df1[['Stud_total_created','NumStud_filled_created']].plot(kind='box')
plt.show()

# <codecell>

df1['numstud_total_filled'] = PCA(n_components=1).fit_transform(df1[['Stud_total_created','NumStud_filled_created']])

# <codecell>

df1.head()

# <codecell>

def isOtherTeacher(x):
    try:
        if np.isnan(x['Other teacher']):
            return 0
        else:
            return 1
    except:
        return 1
    

# <codecell>

df1['isThereOtherTeacher']=df1.apply(isOtherTeacher,axis=1)

# <codecell>

del df1['Other teacher']

# <codecell>

def mapDecision(x):
    if x['Decision']=='Act':
        return 3
    elif x['Decision']=='OK':
        return 1
    elif x['Decision']=='Observe':
        return 2
    else:
        return 0

# <codecell>

df1['numDecision']=df1.apply(mapDecision,axis=1)

# <codecell>

df1.to_csv('df1.csv')

# <codecell>

df1.head()

# <codecell>


# <codecell>

# Feature importance
from sklearn.ensemble import ExtraTreesClassifier

# <codecell>

forest = ExtraTreesClassifier(n_estimators=550,random_state=1)

# <codecell>

df2=df1.copy()

# <codecell>

#df2=df2[df2['numDecision']!=0] # this will filter out weather it is preselection (commented) or after selection (uncommented)

# <codecell>

#df2=df3

# <codecell>

feature_importance_x=np.array(df2[['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']],dtype="float64")
feature_importance_y=np.array(df2[['numDecision']],dtype="float64")
feature_importance_y[feature_importance_y<1]=0
feature_importance_y[feature_importance_y>=1]=1

# <codecell>

forest.fit(feature_importance_x,feature_importance_y.ravel())
importances = forest.feature_importances_

# <codecell>

t2=pd.DataFrame(importances)
t2.index=['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']
t2.columns=['importance in %']
t2['importance in %']=t2['importance in %']*100
t2=t2.sort(columns=['importance in %'],ascending=1)
t2.plot(kind='bar')
plt.show()

# <codecell>

#this is for logistic regression
#from sklearn.linear_model import LogisticRegression
#logreg=LogisticRegression(C=1)
#logreg.fit(feature_importance_x,feature_importance_y.ravel())
#t3=pd.DataFrame(logreg.coef_.ravel())
#t3.index=['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']
#t3.columns=['importance in %']
#t3['importance in %']=t3['importance in %']
#t3=t3.sort(columns=['importance in %'],ascending=1)
#t3.plot(kind='bar')
#plt.show()

# <codecell>

df2.info()

# <codecell>

df3=df2.loc[np.random.choice(df2[df2['numDecision']==0].index, 250, replace=False)]
df3=df3.append(df2[df2['numDecision']!=0])

# <codecell>

df3.to_csv('df3.csv',sep=',')

# <codecell>

feature_x=np.array(df3[['1 Q','2 Q','3 Q','4 Q','5 Q','6  Q','7 Q','8 Q','9 Q','10 Q','11 Q']],dtype="float64")
#target_y=np.array(df[['numDecision']],dtype="float64")
target_y=df3['numDecision'].values

# <codecell>

from sklearn.cross_validation import cross_val_score
scores=cross_val_score(forest,feature_x,target_y,cv=10)

# <codecell>

scores.mean()

# <codecell>


