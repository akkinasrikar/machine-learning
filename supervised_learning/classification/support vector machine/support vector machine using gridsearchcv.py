import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC



cancer=load_breast_cancer()
x=cancer['data']
y=cancer['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=6)

def support_vector_machine():
    sv=SVC()
    sv.fit(x_train,y_train)
    y_pred=sv.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
def grid_search_svm():
    param_grid={'C':[1,10,0.1,0.01,100,1000],
                'gamma':[1,10,0.1,0.01,0.001,0.0001],
                'degree':[1,2,3,4,5,6]}
    grid_svm=GridSearchCV(SVC(),param_grid,verbose=3)
    grid_svm.fit(x_train,y_train)
    #print(grid_svm.best_estimator_)
    print(grid_svm.best_params_)
    y_pred=grid_svm.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
    
    
grid_search_svm()