from sklearn import datasets
import pandas as pd

data = datasets.load_breast_cancer().data



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data = sc.fit_transform(data)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pc = pca.fit_transform(data)
data_pc = pd.DataFrame(data_pc)
data_pc.columns=['P1', "P2"]

from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=2, affinity="rbf")
labels = model.fit_predict(data)



import matplotlib.pyplot as plt

plt.scatter(data_pc["P1"], data_pc["P2"], c=model.fit_predict(data_pc), cmap=plt.cm.winter)
plt.show()