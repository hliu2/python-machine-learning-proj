from sklearn import preprocessing 
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC


data=pd.read_table('bank-additional-full.csv', sep=';')
job=pd.get_dummies(data.job)
categorical_names=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
cat=[]
for categorical in categorical_names:
    cat.append(pd.get_dummies(data[categorical]))
categorical_matrix=pd.merge(cat[0],cat[1],left_index=True,right_index=True)
categorical_dataframe=pd.concat(cat,axis=1)
clean_data=pd.concat([data.drop(categorical_names,axis=1),categorical_dataframe],axis=1)
clean_data.to_csv('clean_data.csv')
y=clean_data.y
X=clean_data.drop('y',axis=1)

#feature importance 
feature_selection_model=ExtraTreesClassifier().fit(X,y)
feature_importance=feature_selection_model.feature_importances_
importance_matrix=np.array([list(X.columns),list(feature_importance)]).T
def sortkey(s):
    return s[1]
sort=zip(list(X.columns),list(feature_importance))
pd.DataFrame(sorted(sort,key=sortkey,reverse=True),,columns=['variabls','importance'])[:10]

#use random foreset based-method to select features
np.random.seed(0)
x_scaled=preprocessing.scale(X)
clf=ExtraTreesClassifier()
x_filtered = clf.fit(x_scaled, y).transform(x_scaled,threshold='0.5*mean')

#split data into two parts
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x_filtered, y, test_size=0.5, random_state=None)
x_train.shape

# cuostomeized criterion to guide the cross-validation 
f2=make_scorer(fbeta_score, beta=2,pos_label='yes')
def f2_score(x):
    return 5.0*x[0,0]/(5.0*x[0,0]+4*x[0,1]+x[1,0])
    
#svm classification 
param_grid = [
  {'C': [100,1000,2000]},
   ]
grid_search = GridSearchCV(LinearSVC(), param_grid, n_jobs=-1, verbose=1,scoring=f2)
svm_fitted=grid_search.fit(x_train,y_train).predict(x_test)
svm_matrix=metrics.confusion_matrix(y_pred=svm_fitted,y_true=y_test,labels=['yes','no'])
print svm_matrix
print f2_score(svm_matrix)

#logistic regression
param_grid = [
  {'class_weight':[{'yes':0.95,'no':0.05},{'yes':0.9,'no':0.1},{'yes':0.8,'no':0.2},{'yes':0.7,'no':0.3},{'yes':0.6,'no':0.4}]},
   ]
grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs=-1, verbose=1,scoring=f2)
log_fitted=grid_search.fit(x_train,y_train).predict(x_test)
log_matrix=metrics.confusion_matrix(y_pred=log_fitted,y_true=y_test,labels=['yes','no'])
print log_matrix
print f2_score(log_matrix)

#knearest neighbors
#tuned_parameters = {'n_neighbors':[1,2,3,5,6,7,8,9,10],'weights':['uniform','distance']}
param_grid = [
  {'n_neighbors': [1,2,3,4,5,6,7,8,9,10]},
   ]
grid_search1 = GridSearchCV(KNeighborsClassifier(), param_grid, n_jobs=-1, verbose=1,scoring=f2)
knn_fitted=grid_search1.fit(x_train,y_train).predict(x_test)
knn_matrix=metrics.confusion_matrix(y_pred=knn_fitted,y_true=y_test,labels=['yes','no'])
print knn_matrix
print f2_score(knn_matrix)

# LDA 
lda_fitted=LDA().fit(x_train,y_train).predict(x_test)
lda_matrix=metrics.confusion_matrix(y_pred=lda_fitted,y_true=y_test,labels=['yes','no'])
print lda_matrix
print f2_score(lda_matrix)

#QDA
QDA_fitted=QDA().fit(x_train,y_train).predict(x_test)
qda_matrix=metrics.confusion_matrix(y_pred=QDA_fitted,y_true=y_test,labels=['yes','no'])
print qda_matrix
print f2_score(qda_matrix)

# random foreset 
param_grid = [
  {'n_estimators': [5,20,50,100]},{'criterion':['gini','entropy']},{'max_depth':[10,100,1000,10000]}
   ]
grid_search1 = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1, verbose=1,scoring=f2)
randomForest_fitted=grid_search1.fit(x_train,y_train).predict(x_test)
randomForest_matrix=metrics.confusion_matrix(y_pred=randomForest_fitted,y_true=y_test,labels=['yes','no'])
print randomForest_matrix
print f2_score(randomForest_matrix)

# Decision tree 
param_grid = [
  {'criterion':['gini','entropy']},{'max_depth':[10,50,100,500,1000,5000,10000,50000]}
   ]
grid_search1 = GridSearchCV(DecisionTreeClassifier (), param_grid, n_jobs=-1, verbose=1,scoring=f2)
tree_fitted=grid_search1.fit(x_train,y_train).predict(x_test)
tree_matrix=metrics.confusion_matrix(y_pred=tree_fitted,y_true=y_test,labels=['yes','no'])
print tree_matrix
print f2_score(tree_matrix)

# bagging tree 
param_grid = [
  {'n_estimators': [5,20,50,100]}
   ]
grid_search1 = GridSearchCV(BaggingClassifier(), param_grid, n_jobs=-1, verbose=1,scoring=f2)
bagging_fitted=grid_search1.fit(x_train,y_train).predict(x_test)
bagging_matrix=metrics.confusion_matrix(y_pred=bagging_fitted,y_true=y_test,labels=['yes','no'])
print bagging_matrix
print f2_score(bagging_matrix)