import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
# print(cancer.keys()) 
print(cancer['DESCR'])
# print(cancer['target_names'])
db=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#print(db.head())
from sklearn.preprocessing import StandardScaler as SS
scalar=SS()
scalar.fit(db)
scaled_data=scalar.transform(db)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
# plt.figure(figsize=(8,6))
# plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
# plt.xlabel("1st PCA")
# plt.ylabel("2nd PCA")
# plt.show()
print(pca.components_)
db_com=pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(db_com,cmap='plasma',)
plt.show()