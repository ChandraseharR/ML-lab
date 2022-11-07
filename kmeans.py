from sklearn import datasets
import random
import pandas as pd
import numpy as np


def dist(x, y):
    n = len(x)
    dist = 0
    for i in range(n):
        dist += np.square(x[i] - y[i])
    return np.sqrt(dist)


def KMeans(data, n_clusters=3):
    n = len(data[0])
    print(data[0])
    cluster_centres_old = [data[random.randint(0, len(data) - 1)] for i in range(n_clusters)]
    print(cluster_centres_old)
    cluster_centres_new = list(cluster_centres_old)

    while True:
        labels = []
        count = np.zeros(n_clusters)

        for i in range(len(data)):
            distances = []
            for j in range(len(cluster_centres_old)):
                distances.append(dist(data[i], cluster_centres_old[j]))
            label = distances.index(min(distances))
            labels.append(label)
            count[label] += 1

        for i in range(n_clusters):
            cluster_centres_new[i] = np.zeros(n)

        for i in range(len(data)):
            cluster_centres_new[labels[i]] += data[i] / count[labels[i]]

        if np.array_equal(cluster_centres_old, cluster_centres_new):
            break
        else:
            cluster_centres_old = list(cluster_centres_new)

    return {"labels": labels, "cluster_centres": cluster_centres_old}



df=pd.read_csv("iris.csv")
features = list(df.columns)
print(features)
X=features[0:-1]
data = np.array(df[X])
# print(data)
# iris = datasets.load_iris()
# print(iris)
# data = iris.data
# print(data)
actual_labels = df[features[-1]]
# print(actual_labels)
print(KMeans(data, 3))