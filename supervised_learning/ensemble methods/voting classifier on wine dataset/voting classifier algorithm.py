import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import  MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
from sklearn.datasets import load_wine

wine_data=load_wine()
x=wine_data.data
y=wine_data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=7)

nb=MultinomialNB()
knn=KNeighborsClassifier(n_neighbors=5)
lg=LogisticRegression()
dc=DecisionTreeClassifier()
rnf=RandomForestClassifier()
sv=SVC(C=100, degree=1,gamma= 0.0001,kernel='poly')

voting_clf=VotingClassifier(estimators=[('lr',lg),
                                        ('rf',rnf),
                                        ('dt',dc),
                                        ('kn',knn),
                                        ('nby',nb),
                                        ('svm',sv)],voting='hard')
voting_clf.fit(x_train,y_train)
y_pred_v=voting_clf.predict(x_test)
print("voting classifier")
print(accuracy_score(y_test,y_pred_v))

for clf in (nb,knn,lg,dc,rnf,sv):
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
    
    





