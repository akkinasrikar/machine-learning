import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
from sklearn.datasets import load_wine
from matplotlib import style
style.use('fivethirtyeight')

wine_data=load_wine()
x=wine_data.data
y=wine_data.target


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=7)


def bagging():
    bag_clf=BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=500,
                              bootstrap=True,
                              n_jobs=-1,
                              max_samples=50)
    bag_clf.fit(x_train,y_train)
    y_pred=bag_clf.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    
def random_forest():
    rnf=RandomForestClassifier()
    rnf.fit(x_train,y_train)
    y_pred=rnf.predict(x_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    for name,score in zip(wine_data['feature_names'],rnf.feature_importances_):
        print(name , score)
    plt_x=np.array(x_test[:,9:10])
    plt_y=np.array(x_test[:,6:7])
    plt.plot([],[],color='red',label='class 0',linewidth=5)
    plt.plot([],[],color='blue',label='class 1',linewidth=5)
    plt.plot([],[],color='green',label='class 2',linewidth=5)
    for i in range(len(plt_x)):
        if y_pred[i]==0:
            plt.scatter(plt_x[i],plt_y[i],s=100,marker='*',color='red')
        elif y_pred[i]==1:
            plt.scatter(plt_x[i],plt_y[i],s=100,marker='D',color='blue')
        else:
            plt.scatter(plt_x[i],plt_y[i],s=100,marker='o',color='green')
    plt.legend(frameon=True)
    plt.show()
    
print("bagging and pasting")    
bagging()
print("RandomForestClassifier")
random_forest()



