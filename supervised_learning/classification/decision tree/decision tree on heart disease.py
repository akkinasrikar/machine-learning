import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

heart_data=pd.read_csv('heart.csv')

x=heart_data.drop('target',axis=1)
y=heart_data['target']

#print(heart_data.info())
#sns.heatmap(heart_data.isnull(),cbar=False,yticklabels=False,cmap='viridis')
#sns.pairplot(heart_data,hue='target')

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4)

def decision_tree():
    dct=DecisionTreeClassifier()
    dct.fit(x_train,y_train)
    y_pred=dct.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

def param_grid():
    paramgrid={
               'min_samples_leaf':[1,2,3,4,5,6],
               'criterion':['gini','entropy'],
               'max_features':[1,2,3,4,9,7,8,11]}
    grid_search=GridSearchCV(DecisionTreeClassifier(),paramgrid,verbose=3)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)
    y_pred=grid_search.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
        

print("grid_search")
param_grid()
print('decision tree')    
decision_tree()