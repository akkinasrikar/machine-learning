import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set_style('whitegrid')


titanic_dataset=pd.read_csv('full.csv')

titanic_dataset.drop(['Cabin','Body','Name_wiki','Destination','Lifeboat',
                      'Class','Age_wiki','Hometown','Boarded'],axis=1,inplace=True)

#exploring the data analysis
#print(titanic_dataset.isnull())
#print(titanic_dataset.info())
#sns.heatmap(titanic_dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#sns.countplot(titanic_dataset['Survived'],hue=titanic_dataset['Sex'])
#sns.distplot(titanic_dataset['Age'].dropna(),bins=30)

#sns.boxplot(x=titanic_dataset['Pclass'],y=titanic_dataset['Age'])
def imputing(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 39
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
titanic_dataset['Age']=titanic_dataset[['Age','Pclass']].apply(imputing,axis=1)          
titanic_dataset.dropna(inplace=True)    
#sns.heatmap(titanic_dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex=pd.get_dummies(titanic_dataset['Sex'],drop_first=True)
embark=pd.get_dummies(titanic_dataset['Embarked'],drop_first=True)
titanic_dataset=pd.concat([titanic_dataset,sex,embark],axis=1)
titanic_dataset.drop(['Sex','Name','Ticket','PassengerId','Embarked'],axis=1,inplace=True)

x=titanic_dataset.iloc[:,1:]
y=titanic_dataset['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5)
x_train=np.array(x_train)
y_train=np.array(y_train)


class logisticregression:
    def intercept(self,x_t):
        intercept_=np.c_[np.ones((x_t.shape[0])),x_t]
        return intercept_
    def sigmoid(self,z):
        r=(1)/(1+np.exp(-z))
        return r
    def fit(self,x_f,y_f):
        n=100000
        lr=0.01
        x_new=self.intercept(x_f)
        self.theta=np.zeros(x_new.shape[1])
        for i in range(n):
            z=np.dot(x_new,self.theta)
            p=self.sigmoid(z)
            gradient=np.dot(x_new.T,p-y_train)/len(x_train)
            self.theta=self.theta-lr*gradient          
    def predict(self,x_p):
        x_new=self.intercept(x_p)
        rel=self.sigmoid(np.dot(x_new,self.theta))
        for i in range(len(x_p)):
            if rel[i]<0.5:
                rel[i]=0
            else:
                rel[i]=1
        return rel

model=logisticregression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)  
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))        
           
        
        
        
