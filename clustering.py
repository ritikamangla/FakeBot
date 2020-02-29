import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler , OneHotEncoder ,LabelEncoder
from datetime import timedelta
from sklearn.decomposition import PCA


dataset = pd.read_csv('FinalDS_AdditionalFeatures.csv')
x=dataset.iloc[:,[7,8,12,13,16,17,20,39,40,41,44]].values
encoder = LabelEncoder()
x[:,3] = encoder.fit_transform(x[:,3])
#CALCULATE TIME DIFFERENCE


datetimeFormat = '%d-%m-%Y %H:%M'
date1 = '09-02-2020 10:01'
difference =[]
for i in range(1600):
    date2 = x[i,2]
    diff = datetime.datetime.strptime(date1, datetimeFormat) \
           - datetime.datetime.strptime(date2, datetimeFormat)
    difference.append(diff.days)

print("Difference:", difference)

x = np.append(arr =x  ,values = np.ones((1601,1)).astype(list), axis= 1)
for i in range(1600):
    x[i][11] = difference[i]

x= x[:,[0,1,3,4,5,6,7,8,9,10,11]]

pca =PCA(n_components=2)
x= pca.fit_transform(x)

wcss = []
from sklearn.cluster import KMeans

"""for i in range (1,11):
    kmeans = KMeans(n_clusters= i , init = 'k-means++',max_iter=300 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters= 5 , init = 'k-means++',max_iter=300 ,random_state=0)
y_means = kmeans.fit_predict(x)"""

kmeans = KMeans(n_clusters= 3 , init = 'k-means++',max_iter=300 ,random_state=0)
y_means = kmeans.fit_predict(x)
#q = kmeans.

groups= pd.DataFrame(y_means)
gk = groups.groupby(0)
print(gk)

l0=[]
l1=[]
l2=[]

for i in range(1600):
    if(y_means[i]==0):
        l0.append(i)

    if (y_means[i] == 1):
        l1.append(i)

    if (y_means[i] == 2):
        l2.append(i)



print(x[y_means == 0, 1])
#visualising data
plt.scatter(x[y_means == 0,0],x[y_means == 0, 1], s = 100 ,c= 'red',label ='cluster 1',alpha=0.2)
plt.scatter(x[y_means == 1,0],x[y_means == 1, 1], s = 100 ,c= 'cyan',label ='cluster 2',alpha=0.2)
plt.scatter(x[y_means == 2,0],x[y_means == 2, 1], s = 100 ,c= 'green',label ='cluster 3',alpha=0.2)
plt.legend()
plt.show()