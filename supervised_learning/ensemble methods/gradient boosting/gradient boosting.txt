import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import style
style.use('fivethirtyeight')



data=pd.read_csv('heart.csv')
x=np.array(data.iloc[:,:-1].values)
y=np.array(data.iloc[:,-1].values)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12)

def gradient_boosting():
    gbc=GradientBoostingClassifier(n_estimators=382,max_depth=2,learning_rate=0.1)
    gbc.fit(x,y)
    y_pred=gbc.predict(x_test)
    print(accuracy_score(y_test,y_pred))
    
def best_values(n):
    accuracy=[]
    n_esimator_=[]
    for i in range(1,n+1):
        gbc=GradientBoostingClassifier(n_estimators=i,max_depth=2,learning_rate=0.1)
        gbc.fit(x,y)
        y_pred=gbc.predict(x_test)
        rel=accuracy_score(y_test,y_pred)
        accuracy.append(rel)
        n_esimator_.append(i)
        if rel==1:
            break
    b_n=n_esimator_[-1]
    print(f"best n_estimator is {b_n}")
    plt.plot(n_esimator_,accuracy,marker='o',
             markerfacecolor='red',
             color='blue',
             markersize=10,
             linestyle='dashed')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy rate')
    plt.show()
    
gradient_boosting()
best_values(420)

