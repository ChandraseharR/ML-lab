import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data= pd.read_csv("tic-tac-toe-endgame.csv")
print(data.head())
df = pd.get_dummies(data, columns=data.columns.tolist()[:-1])
print(df.head())

cols= df.columns.tolist()

ip = cols[1:]
op = cols[0]

X = df[ip].to_numpy()
Y = df[op].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#fitting the model
clf = tree.DecisionTreeClassifier().fit(X_train, Y_train)

#save plot
plt.figure(figsize=(20, 20))
tree.plot_tree(clf, filled=True)
plt.savefig('tree.png')
plt.show()

#tree height = 10 for now

#prediction
Y_pred = clf.predict(X_test)

#confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#accuracy
print("Accuracy = ", accuracy_score(Y_test, Y_pred))

#training error
Y_pred_train = clf.predict(X_train)
print("Training error rate = ", 1 - accuracy_score(Y_train, Y_pred_train))



Decision Tree - Iris:
 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# dataset from sklearn (readily available X and Y for decision tree)
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# Y = iris.target

iris_dataset = pd.read_csv("Iris.csv")
cols = iris_dataset.columns.tolist()

input_features = cols[1:5]
class_label = cols[5]

X = iris_dataset[input_features].to_numpy()
Y = iris_dataset[class_label].to_numpy()

#train test splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting the model
clf = tree.DecisionTreeClassifier().fit(X_train, Y_train)

#save plot
plt.figure(figsize=(20, 20))
tree.plot_tree(clf, filled=True)
plt.savefig('tree.png')
plt.show()

#prediction
Y_pred = clf.predict(X_test)

#confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

#accuracy
print("Accuracy: ", accuracy_score(Y_test, Y_pred))