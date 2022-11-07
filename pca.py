import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = np.array([[90,90,25,95,100],[80,95,40,85,77],[50,30,95,87,27],[27,37,25,68,25],[25,41,88,63,36],[41,42,45,61,78]])
df = pd.DataFrame(data,columns = ['Maths','Science','Social','English','Lang-II'])
print(df)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scale=scaler.fit_transform(data)
print(scale)

cov=np.cov(scale.T)
print(cov)

eval,evec=np.linalg.eig(cov)
evec
eigpar=[[eval[i],evec[i]] for i in range(len(eval))]
eigpar
sorted(eigpar,key= lambda x:x[0],reverse=True)
for i in eigpar:
  print(i[0])
n=2
wc= np.array([eigpar[i][1] for i in range(n)]).T
wc
fd=np.dot(scale,wc)
fd

df=pd.DataFrame(fd,columns=['pc1','pc2'])
df.cov()

