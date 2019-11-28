import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns
sns.set_style('whitegrid')

mnist=load_digits()
x=np.array(mnist.images)
y=np.array(mnist.target)
n=int(input("enter the number btw 1 to 1500>>"))
some_digit=np.array(x[n])
plt.imshow(some_digit,cmap=plt.cm.gray_r,interpolation='nearest')
plt.axis("off")
plt.show
print(f"expected output {y[n]}")


nsamples=len(x)
x=x.reshape((nsamples,-1))


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=6)


def softmax_regression():
    multiclass_reg=LogisticRegression(multi_class='multinomial',solver="lbfgs")
    multiclass_reg.fit(x_train,y_train)
    y_pred=multiclass_reg.predict(x_test)
    rel=multiclass_reg.predict([x[n]])
    print(f"predicted output is {rel}")
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
   
        
softmax_regression()        
        
