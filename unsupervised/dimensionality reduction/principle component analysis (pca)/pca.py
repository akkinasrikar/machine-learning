import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score
import plotly 
import plotly.graph_objects as go

wine_data=load_wine()
x_data=wine_data['data']
y_target=np.array(wine_data['target'])

sc=StandardScaler()
x_data=sc.fit_transform(x_data)
pca=PCA(n_components=2)
x_data=pca.fit_transform(x_data)

class kmeans_clustering:
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
    def clustering(self,nf):
        kmeans=KMeans(n_clusters=nf)
        kmeans.fit(self.data)
        fig,(ax1,ax2)=plt.subplots(1,2,sharey=False,figsize=(10,6))
        ax1.scatter(self.data[:,0],self.data[:,1],c=kmeans.labels_,cmap='rainbow',s=40)
        ax1.set_title('KMeans')
        ax2.scatter(self.data[:,0],self.data[:,1],c=y_target,cmap='rainbow',s=40)
        ax2.set_title('Original')
method=kmeans_clustering(n=10,data=x_data)
method.elbow_method()
method.silhouette_method()
method.visualize()
method.clustering(3)
        
        
        
    
