import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as shc
from sklearn.cluster import AgglomerativeClustering 
from matplotlib import style
style.use('fivethirtyeight')


mall_data=pd.read_csv('mall.csv')

x=mall_data.iloc[:,3:].values

def den():
    dendograms=shc.dendrogram(shc.linkage(x,method='ward'))
    plt.title('Dendograms')
    plt.xlabel('customers')
    plt.ylabel('euclidean distance')
    plt.show()

def hc():
    agc=AgglomerativeClustering(n_clusters=5,linkage='ward',affinity="euclidean")
    agc.fit_predict(x)
    fig,(ax1)=plt.subplots(1,sharey=False,figsize=(10,8))
    scatter1=ax1.scatter(x[:,0],x[:,1],cmap='rainbow',c=agc.labels_,s=40)
    legend1=ax1.legend(*scatter1.legend_elements(),title='class')
    ax1.add_artist(legend1)
    ax1.set_title('HC')
    plt.show()
#den()
hc()