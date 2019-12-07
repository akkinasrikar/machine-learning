import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from matplotlib import style
style.use('fivethirtyeight')
iris_data=load_iris()
flowers_data=iris_data.data
flowers_data=flowers_data[:,0:]
labels=iris_data.target

#plt.scatter(flowers_data[:,0],flowers_data[:,1],c=labels,cmap='rainbow')
#plt.show()

def clustering():
    kmeans=KMeans(n_clusters=4)
    kmeans.fit(flowers_data)
    plt.scatter(flowers_data[:,0],flowers_data[:,1],c=kmeans.labels_,cmap='rainbow')
    plt.show()
    
class choosing_n_value:
    def __init__(self,n,data):
        self.n=n
        self.data=data
    def elbow_method(self):
        self.wcss=[]
        for i in range(1,self.n+1):
            kmeans=KMeans(n_clusters=i)
            kmeans.fit(self.data)
            self.wcss.append(kmeans.inertia_)
        #plt.plot(self.wcss,range(1,self.n+1),marker='o',color='blue',linestyle='dashed',markerfacecolor='red',markersize=10)
        #plt.show()
    def silhouette_method(self):
        self.ssc=[]
        for i in range(2,self.n+1):
            kmeans=KMeans(n_clusters=i)
            kmeans.fit(self.data)
            labels=kmeans.labels_
            self.ssc.append(silhouette_score(self.data,labels,metric='euclidean'))
        #plt.plot(range(2,self.n+1),self.ssc,marker='o',color='blue',linestyle='dashed',markerfacecolor='red',markersize=10)
        #plt.show()
    def visualize(self):
        fig,(ax1,ax2)=plt.subplots(1,2,sharey=False,figsize=(10,6))
        ax1.plot(self.wcss,range(1,self.n+1),marker='o',color='blue',linestyle='dashed',markerfacecolor='red',markersize=10)
        ax1.set_title('elbow method')
        ax2.plot(range(2,self.n+1),self.ssc,marker='o',color='blue',linestyle='dashed',markerfacecolor='red',markersize=10)
        ax2.set_title('silhouette method')
    def clustering(self):
        kmeans=KMeans(n_clusters=self.n)
        kmeans.fit(self.data)
        fig,(ax1,ax2)=plt.subplots(1,2,sharey=False,figsize=(10,6))
        ax1.scatter(flowers_data[:,0],flowers_data[:,1],c=kmeans.labels_,cmap='rainbow')
        ax1.set_title('KMeans clustering')
        ax2.scatter(flowers_data[:,0],flowers_data[:,1],c=labels,cmap='rainbow')
        ax2.set_title('original data')
        
        
'''
n_value=choosing_n_value(6,flowers_data)
n_value.elbow_method()  
n_value.silhouette_method()
n_value.visualize() '''

model=choosing_n_value(3,flowers_data)
model.clustering()