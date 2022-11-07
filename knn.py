data=[[7,7],[7,4],[3,4],[1,4]]
do=[[7,7,'bad'],[7,4,'bad'],[3,4,'good'],[1,4,'good']]
df=pd.DataFrame(do,columns=['dur','strength','y'])
df
def dist(x, y):
    n = len(x)
    dist=0
    for i in range(n):
        dist += np.square(x[i]-y[i])
    return np.sqrt(dist)
point=[2,2]
dic={}
for i in range(len(data)):
  dic[str(data[i])]=[dist(point,data[i]),do[i][2]]
d=sorted(dic.items(),key=lambda x: x[1])
print(d)

k=3
lst=[]
for i in range(k):
  lst.append(d[i])
print(lst)

for i in range(len(lst)):
  lst[i]=list(lst[i])
c1,c2=0,0
for i in lst:
   if i[1][1]=='good':
     c1+=1
   else:
     c2+=1
if c1>c2:
  print('point is good')
else:
  print('point is bad')