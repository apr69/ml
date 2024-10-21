import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Iris.csv')
df
df.drop('Id', axis=1 , inplace=True)
df

df.info()
df.isnull().sum()

df.nunique()

df['Species'].unique()

df1 = df[df['Species']=='Iris-setosa']
df2 = df[df['Species']=='Iris-versicolor']
df3 = df[df['Species']=='Iris-virginica']
plt.scatter(df1['PetalLengthCm'],df1['PetalWidthCm'], color='r' , label='Iris-setosa')
plt.scatter(df2['PetalLengthCm'],df2['PetalWidthCm'], color='b', label='Iris-versicolor')
plt.scatter(df3['PetalLengthCm'],df3['PetalWidthCm'], color='g' , label='Iris-virginica ')
plt.legend()
plt.show()


df_imp = df.iloc[:,0:4]
from sklearn.cluster import KMeans
k_meansclus = range(1,10)
sse = []

for k in k_meansclus :
  km = KMeans(n_clusters =k)
  km.fit(df_imp)
  sse.append(km.inertia_)


plt.title('The Elbow Method')
plt.plot(k_meansclus,sse)
plt.show()

km1 = KMeans(n_clusters=3,max_iter=300 , random_state=0)
km1.fit(df_imp)

y_means = km1.fit_predict(df_imp)
y_means

df_imp = np.array(df_imp)
df_imp

plt.scatter(df_imp[y_means==1,2 ],df_imp[y_means==1,3 ], color='r' , label='Iris-setosa')
plt.scatter(df_imp[y_means==0,2 ],df_imp[y_means==0,3 ], color='b' , label='Iris-versicolor')
plt.scatter(df_imp[y_means==2,2 ],df_imp[y_means==2,3 ], color='g', label='Iris-virginica')
plt.legend()
plt.show()
